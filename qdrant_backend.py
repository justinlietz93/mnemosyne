"""Simple Qdrant adapter for Mnemosyne.

Provides upsert/search/update/delete/count operations with a small, stable API used
by `mnemosyne_core.Mnemosyne` when a Qdrant backend is requested.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Iterable
import logging
import threading
import queue
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception as e:
    raise RuntimeError("qdrant-client is required for qdrant_backend. pip install qdrant-client") from e

log = logging.getLogger(__name__)


class QdrantIndex:
    def __init__(
        self,
        collection_name: str,
        url: str = "http://127.0.0.1:6333",
        api_key: Optional[str] = None,
        vector_size: int = 1024,
        distance: str = "Cosine",
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)
        dist_enum = qmodels.Distance.COSINE if distance.lower().startswith("cos") else qmodels.Distance.EUCLID
        # Try to ensure the collection exists non-destructively. If Qdrant is unreachable
        # mark the adapter as unavailable and let callers fall back.
        try:
            self._ensure_collection(vector_size, dist_enum)
            self.available = True
        except Exception:
            self.available = False
        # Background queue for async payload updates (start only when adapter initialized)
        try:
            self._update_q: "queue.Queue[tuple[str, Dict]]" = queue.Queue()
            self._worker_thread = threading.Thread(target=self._payload_worker, daemon=True)
            self._worker_thread.start()
            log.debug("Qdrant async payload worker started")
        except Exception:
            # Best-effort: if background worker cannot start, remain functional synchronously
            log.exception("Failed to start background payload worker; falling back to sync updates")

    def _ensure_collection(self, vector_size: int, distance: qmodels.Distance):
        # Non-destructive create-if-missing. Avoid recreate_collection which wipes data.
        try:
            cols = self.client.get_collections()
            exists = [c.name for c in getattr(cols, 'collections', [])]
            if self.collection_name not in exists:
                # create_collection is idempotent for existing collections in recent clients
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
                )
        except Exception:
            # As a last resort attempt to create the collection; if that fails, surface the error to caller
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
            )

    def upsert(
        self,
        ids: Iterable[str],
        embeddings: Iterable[List[float]],
        payloads: Optional[Iterable[Dict]] = None,
        batch_size: int = 256,
    ) -> List[str]:
        """
        Upsert points. Returns the list of Qdrant point IDs (UUID strings) in the same order as 'ids'.
        Note: To keep the index 'pure embeddings', callers should pass payloads=None.
        """
        it_ids = list(ids)
        it_emb = list(embeddings)
        it_payloads = list(payloads) if payloads is not None else [None] * len(it_ids)
        assert len(it_ids) == len(it_emb) == len(it_payloads)
        out_qids: List[str] = []
        n = len(it_ids)
        for i in range(0, n, batch_size):
            chunk_ids = it_ids[i : i + batch_size]
            chunk_vecs = it_emb[i : i + batch_size]
            chunk_payloads = it_payloads[i : i + batch_size]
            points = []
            qids_batch: List[str] = []
            for pid, vec, pay in zip(chunk_ids, chunk_vecs, chunk_payloads):
                # Qdrant point IDs must be integer or UUID; derive deterministic UUID5 from original id
                try:
                    qid = str(uuid.UUID(pid)) if isinstance(pid, str) and len(pid) == 36 else str(uuid.uuid5(uuid.NAMESPACE_DNS, pid))
                except Exception:
                    qid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(pid)))
                qids_batch.append(qid)
                # Keep payload None by default for pure-embedding operation
                p = None if pay is None else dict(pay)
                points.append(qmodels.PointStruct(id=qid, vector=vec, payload=p))
            self.client.upsert(self.collection_name, points=points)
            out_qids.extend(qids_batch)
        return out_qids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_payload: Optional[Dict] = None,
        with_payload: bool = False,
    ) -> Dict[str, List]:
        qfilter = None
        if filter_payload:
            must = []
            for k, v in filter_payload.items():
                must.append(
                    qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                )
            qfilter = qmodels.Filter(must=must)

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qfilter,
            with_payload=with_payload,
        )
        # Return Qdrant point IDs as strings; callers may map to original ids externally.
        ids = [[str(h.id) for h in hits]]
        distances = [[(1.0 - h.score) if h.score is not None else 1.0 for h in hits]]
        # No payloads returned; keep shape compatibility with Chroma
        documents = [[None for _ in hits]]
        metadatas = [[{} for _ in hits]]
        return {"ids": ids, "distances": distances, "documents": documents, "metadatas": metadatas}

    def update_payload(self, doc_id: str, payload: Dict):
        # Keep synchronous API for compatibility but delegate heavy work to async path
        try:
            # Enqueue for async processing; fallback to sync if queue put fails
            try:
                # Convert original id to qdrant id
                try:
                    qid = str(uuid.UUID(doc_id)) if isinstance(doc_id, str) and len(doc_id) == 36 else str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
                except Exception:
                    qid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))
                self._update_q.put_nowait((qid, payload))
                return
            except Exception:
                pass
            # Fallback synchronous update using set_payload/overwrite_payload
            try:
                self.client.set_payload(collection_name=self.collection_name, payload=payload, points=[qid])
            except Exception:
                try:
                    self.client.overwrite_payload(collection_name=self.collection_name, payload=payload, points=[qid])
                except Exception:
                    log.exception("sync payload update failed for %s", qid)
        except Exception as e:
            log.exception("payload update failed for %s: %s", doc_id, e)

    def _payload_worker(self):
        while True:
            try:
                doc_id, payload = self._update_q.get()
                try:
                    try:
                        self.client.set_payload(collection_name=self.collection_name, payload=payload, points=[doc_id])
                    except Exception:
                        self.client.overwrite_payload(collection_name=self.collection_name, payload=payload, points=[doc_id])
                except Exception:
                    log.exception("async payload update failed for %s", doc_id)
                finally:
                    self._update_q.task_done()
            except Exception:
                # Thread should continue running; avoid dying on unexpected errors
                continue

    def delete(self, ids: Iterable[str]):
        try:
            # Convert provided ids to qdrant-compatible ids
            qids = []
            for idv in ids:
                try:
                    qid = str(uuid.UUID(idv)) if isinstance(idv, str) and len(idv) == 36 else str(uuid.uuid5(uuid.NAMESPACE_DNS, idv))
                except Exception:
                    qid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(idv)))
                qids.append(qid)
            self.client.delete(collection_name=self.collection_name, points=qids)
        except Exception as e:
            log.exception("delete failed: %s", e)

    def count(self) -> int:
        st = self.client.count(self.collection_name)
        return int(st.count if st is not None else 0)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
