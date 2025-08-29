# ==============================================================================
# Project Prometheus: Minimum Viable Mnemosyne (MVM)
# Version 0.7
#
# Agent: PrometheusAI
# Mission: To create the foundational memory-injection pipeline for our AGI.
#
# Description:
# This script provides an interactive Command-Line Interface (CLI) to interact
# with the Mnemosyne memory system, allowing for dynamic injection and retrieval.
#
# Changelog:
# v0.7 - Hardcoded embedding model to a dedicated, fast model for efficiency.
# ==============================================================================

import ollama
import time
import re
import os
from void_memory import VoidMemoryManager
try:
    from qdrant_backend import QdrantIndex
except Exception:
    QdrantIndex = None
# Local pure-text store (keeps Qdrant payloads empty)
try:
    from local_doc_store import LocalDocStore
except Exception:
    LocalDocStore = None  # type: ignore

# --- Configuration ---
EMBEDDING_MODEL = 'mxbai-embed-large' # Use a dedicated, fast model for embeddings
DB_PATH = "./mvm_db"
COLLECTION_NAME = "mnemosyne_core"
STORAGE_PREFIX = "memory-passage: "
QUERY_PREFIX = "query: "

class Mnemosyne:
    """
    The foundational memory system. Handles memory injection and retrieval.
    """
    def __init__(self, db_path: str, collection_name: str, model: str, backend: str = 'qdrant', qdrant_url: str = 'http://127.0.0.1:6333', *, enable_void_memory: bool = True, force_qdrant: bool = True):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = model
        self.storage_prefix = STORAGE_PREFIX
        self.query_prefix = QUERY_PREFIX
        self.enable_void_memory = enable_void_memory
        self._void_memory = VoidMemoryManager(capacity=5000) if enable_void_memory else None
        self._backend = backend.lower()
        self._qdrant_url = qdrant_url
        # Ensure DB path exists for local doc store
        try:
            os.makedirs(self.db_path, exist_ok=True)
        except Exception:
            pass
        
        print(f"Initializing Mnemosyne Core...")
        print(f"  - DB Path: {self.db_path}")
        print(f"  - Collection: {self.collection_name}")
        print(f"  - Embedding Model: {self.model}")

        # Enforce Qdrant-only operation. Fail fast if Qdrant adapter is missing or unreachable.
        try:
            if QdrantIndex is None:
                raise RuntimeError("QdrantIndex adapter not available (missing import)")
            # Use vector dim matching mxbai-embed-large (1024)
            self._index = QdrantIndex(collection_name=self.collection_name, url=self._qdrant_url, vector_size=1024)
            if not getattr(self._index, 'available', True):
                raise RuntimeError("Qdrant adapter initialization failed or Qdrant unreachable")
            # Explicitly mark no local Chroma client will be used
            self.client = None
            # Initialize local doc store for pure-embedding setup (no payloads in Qdrant)
            try:
                if LocalDocStore is None:
                    raise RuntimeError("LocalDocStore not available")
                self._doc_store = LocalDocStore(os.path.join(self.db_path, "doc_store.sqlite3"))
            except Exception as e:
                print(f"[Mnemosyne] Warning: LocalDocStore unavailable: {e}")
                self._doc_store = None

            # Chroma-shaped adapter backed by Qdrant + LocalDocStore
            class _Q:
                def __init__(self, idx, doc_store, parent_ref):
                    self._idx = idx
                    self._doc = doc_store
                    self._parent = parent_ref
                def add(self, embeddings, documents, ids, metadatas):
                    # Upsert pure embeddings; store texts externally
                    try:
                        qids = self._idx.upsert(ids=ids, embeddings=embeddings, payloads=None)
                        if self._doc is not None:
                            try:
                                self._doc.upsert_mappings(qids, ids, documents)
                            except Exception:
                                pass
                    except Exception:
                        pass
                def query(self, query_embeddings, n_results=3, where=None):
                    # Pure-embedding search; then map qids->orig_ids and texts from LocalDocStore
                    out = self._idx.search(query_embedding=query_embeddings[0], top_k=max(1, int(n_results)), filter_payload=None, with_payload=False)
                    qids = out.get("ids", [[]])[0] if out else []
                    dists = out.get("distances", [[]])[0] if out else []
                    if self._doc is not None:
                        try:
                            orig_ids = self._doc.get_orig_ids_by_qids(qids)
                        except Exception:
                            orig_ids = [None] * len(qids)
                        try:
                            texts = self._doc.get_texts_by_qids(qids)
                        except Exception:
                            texts = [None] * len(qids)
                    else:
                        orig_ids, texts = [None] * len(qids), [None] * len(qids)
                    ids = [[(orig_ids[i] if orig_ids[i] is not None else qids[i]) for i in range(len(qids))]]
                    documents = [[(texts[i] if i < len(texts) else None) for i in range(len(qids))]]
                    metadatas = [[{} for _ in range(len(qids))]]
                    results = {"ids": ids, "distances": [dists], "documents": documents, "metadatas": metadatas}
                    # Lifecycle reinforcement stays internal (no payload writes)
                    try:
                        if hasattr(self._parent, "_process_reinforce_and_mirror"):
                            self._parent._process_reinforce_and_mirror(results)
                    except Exception:
                        pass
                    return results
                def count(self):
                    return self._idx.count()

            self.collection = _Q(self._index, getattr(self, "_doc_store", None), self)
        except Exception:
            # Always fail fast for Qdrant-only configuration
            raise
        # Print a safe memory count depending on backend
        try:
            count = self._index.count() if getattr(self, '_index', None) is not None else 0
        except Exception:
            count = 0
        print(f"Mnemosyne Core initialized. Current memory count: {count}")
        if self._void_memory:
            print("[VoidMemory] Lifecycle manager ENABLED.")
        else:
            print("[VoidMemory] Lifecycle manager DISABLED.")

    def _get_embedding(self, prompt: str):
        """Generates an embedding for a given prompt."""
        return ollama.embeddings(model=self.model, prompt=prompt)['embedding']

    def _generate_query_embedding(self, query: str):
        prefixed_query = self.query_prefix + query
        return self._get_embedding(prefixed_query)

    def _search_backend(self, query_embedding, n_results: int = 3, where_filter: dict = None):
        # Return a standardized results dict with keys: ids, distances, documents, metadatas
        # Qdrant-only search path
        if getattr(self, '_index', None) is None:
            raise RuntimeError("No Qdrant index attached; cannot perform search")
        return self._index.search(query_embedding, top_k=max(n_results, 8), filter_payload=where_filter)

    def _process_reinforce_and_mirror(self, results: dict):
        # Reinforce void memory and, if using Qdrant, mirror lifecycle fields back to payloads
        if not self._void_memory or not results:
            return
        try:
            self._void_memory.reinforce(results)
        except Exception as e:
            print(f"[VoidMemory] Reinforce error: {e}")
        try:
            retrieved_ids = results.get('ids', [[]])[0] if results else []
            self._void_memory.tick_post_retrieval(retrieved_ids)
        except Exception as e:
            print(f"[VoidMemory] Post-retrieval hook error: {e}")

        # No payload mirroring: keep Qdrant pure embeddings (all lifecycle stays internal)
        return

    def _chunk_text(self, text: str, chunk_size: int = 5, overlap: int = 1) -> list[str]:
        """
        A configurable text chunker that splits text into groups of sentences with overlap.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ').strip())
        if not sentences or (len(sentences) == 1 and sentences[0] == ''):
            return []
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            end = i + chunk_size
            chunk_sentences = sentences[max(0, i - overlap):end]
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
        
        return chunks

    def inject(self, document: str, source_id: str, chunk_size: int = 5, overlap: int = 1, metadata: dict = None):
        """
        Injects a large document into memory by chunking it first with configurable parameters and optional metadata.
        """
        print(f"\n--- Injecting Document: {source_id} ---")
        chunks = self._chunk_text(document, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            print("Warning: Document is empty or contains no valid sentences. Nothing to inject.")
            return

        print(f"Document split into {len(chunks)} chunks.")

        chunk_ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
        prefixed_chunks = [self.storage_prefix + chunk for chunk in chunks]

        print("Generating embeddings for all chunks...")
        try:
            embeddings = [
                self._get_embedding(p_chunk)
                for p_chunk in prefixed_chunks
            ]
            print("Embeddings generated successfully.")
        except Exception as e:
            print(f"Error during bulk embedding generation: {e}")
            return

        if metadata is None:
            metadata = {}

        # Add timestamp if not provided
        if 'timestamp' not in metadata:
            metadata['timestamp'] = time.time()

        # Add source if not provided
        if 'source' not in metadata:
            metadata['source'] = source_id

        metadatas = [metadata.copy() for _ in chunks]  # Same metadata for all chunks of this document
        # Qdrant-only injection
        if getattr(self, '_index', None) is None:
            raise RuntimeError("No Qdrant index attached; cannot inject data")
        # Upsert pure embeddings to Qdrant; manage text externally in LocalDocStore
        try:
            qids = self._index.upsert(ids=chunk_ids, embeddings=embeddings, payloads=None)
        except Exception as e:
            print(f"[Qdrant] Upsert failed: {e}")
            return
        # Persist mapping qid -> (orig_id, text) outside of Qdrant
        try:
            if getattr(self, "_doc_store", None) is not None:
                self._doc_store.upsert_mappings(qids, chunk_ids, chunks)
        except Exception as e:
            print(f"[DocStore] Upsert mapping failed: {e}")
        if self._void_memory:
            try:
                self._void_memory.register_chunks(chunk_ids, chunks, embeddings=embeddings)
            except TypeError:
                # backward-compatible: old signature
                self._void_memory.register_chunks(chunk_ids, chunks)
        print(f"All chunks for document '{source_id}' injected successfully with metadata.")
        # Optional: mark autosave hook externally

    # ---------------- Void Memory Persistence & Diagnostics ----------------
    def save_void_state(self, path: str) -> bool:
        if not self._void_memory:
            return False
        try:
            return bool(self._void_memory.save_json(path))
        except Exception as e:  # pragma: no cover
            print(f"[VoidMemory] Save failed: {e}")
            return False

    def load_void_state(self, path: str) -> bool:
        if not self.enable_void_memory:
            return False
        try:
            if not path or not os.path.exists(path):
                return False
            loaded = VoidMemoryManager.load_json(path)
            if loaded:
                self._void_memory = loaded
                print(f"[VoidMemory] Loaded state from {path} (tick={loaded.stats().get('tick',0)})")
                return True
            return False
        except Exception as e:  # pragma: no cover
            print(f"[VoidMemory] Load failed: {e}")
            return False

    def void_stats(self):
        if not self._void_memory:
            return {}
        return self._void_memory.stats()

    def void_top(self, k: int = 10):
        if not self._void_memory:
            return []
        return self._void_memory.top(k)

    def void_events(self, limit: int = 50, consume: bool = False):
        if not self._void_memory:
            return []
        if consume:
            return self._void_memory.consume_events()[-limit:]
        return self._void_memory.peek_events(limit)

    def retrieve(self, query: str, n_results: int = 3, where_filter: dict = None):
        """
        Retrieves the most relevant memory chunks based on a query, with optional metadata filter.
        """
        print(f"\n--- Retrieving Memories ---")
        print(f"Query: \"{query}\"")
 
        try:
            query_embedding = self._generate_query_embedding(query)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return

        # Use collection adapter to fetch texts from local store and reinforce lifecycle
        try:
            results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where_filter)
        except Exception as e:
            print(f"[Mnemosyne] Query failed: {e}")
            return

        print("\nTop matching memories (chunks):")
        if not results['documents'] or not results['documents'][0]:
            print("No matching memories found.")
            return

        # Exploration-aware re-ranking
        ids = results['ids'][0]
        docs = results['documents'][0]
        dists = results['distances'][0]
        metas = results['metadatas'][0]
        if self._void_memory:
            base = []
            for cid, doc, dist, meta in zip(ids, docs, dists, metas):
                sim = 1 - dist
                score = self._void_memory.composite_score_for(cid) or 0.0
                expw = self._void_memory.exploratory_weight(cid)
                temp = self._void_memory.exploration_temperature()
                # Blend: exploitation (similarity + lifecycle score) and exploration (exp weight * temp)
                blended = 0.6 * sim + 0.3 * score + 0.1 * expw * (0.5 + 0.5 * temp)
                # Small stochastic jitter scaled by temp
                jitter = (temp ** 2) * 0.05 * (hash(cid) % 997 / 997.0 - 0.5)
                base.append((blended + jitter, cid, doc, meta, sim))
            base.sort(key=lambda x: x[0], reverse=True)
            display = base[:n_results]
        else:
            display = []
            for cid, doc, dist, meta in zip(ids, docs, dists, metas):
                sim = 1 - dist
                display.append((sim, cid, doc, meta, sim))
            display.sort(key=lambda x: x[0], reverse=True)
            display = display[:n_results]

        for rank, (_, cid, doc, meta, sim) in enumerate(display, 1):
            print(f"  {rank}. Chunk ID: {cid}")
            print(f"      Similarity: {sim:.4f}")
            print(f"      Metadata: {meta}")
            print(f"      Content: \"{doc}\"")
        if self._void_memory:
            stats = self._void_memory.stats()
            print(f"[VoidMemory] Stats: count={int(stats['count'])} avg_conf={stats['avg_conf']:.3f} avg_mass={stats['avg_mass']:.2f} avg_heat={stats['avg_heat']:.2f} territories={int(stats['territories'])} avg_boredom={stats.get('avg_boredom',0):.2f} exploration_temp={stats.get('exploration_temp',0):.2f}")
            pending = self._void_memory.consume_pending_condensations()
            if pending:
                # Register internal engram (no semantic injection)
                self._void_memory.register_engram(pending)
                self._void_memory.degrade(pending, ttl_floor=int(self._void_memory.base_ttl/4))
                print(f"[VoidMemory] Internal engram formed over {len(pending)} chunks.")

# --- Main Execution: Interactive CLI ---
if __name__ == "__main__":
    mnemosyne = Mnemosyne(db_path=DB_PATH, collection_name=COLLECTION_NAME, model=EMBEDDING_MODEL)

    print("\n--- Mnemosyne Interactive CLI ---")
    print("Commands: inject, retrieve, count, vstats, vevents, vtop, vsave <path>, vload <path>, quit")

    while True:
        raw = input("\nEnter command > ").strip()
        parts = raw.split()
        command = parts[0].lower() if parts else ''

        if command == "quit":
            print("Shutting down Mnemosyne.")
            break

        elif command == "count":
            try:
                if getattr(mnemosyne, '_index', None) is not None:
                    cnt = mnemosyne._index.count()
                elif getattr(mnemosyne, 'collection', None) is not None:
                    cnt = mnemosyne.collection.count()
                else:
                    cnt = 0
            except Exception:
                cnt = 0
            print(f"Current memory count: {cnt}")

        elif command == "inject":
            source_id = input("Enter a unique source ID for this document > ").strip()
            print("Enter the document text. Press Ctrl+D (Linux/Mac) or Ctrl+Z then Enter (Windows) when done.")
            document_lines = []
            try:
                while True:
                    line = input()
                    document_lines.append(line)
            except EOFError:
                pass
            document = "\n".join(document_lines)
            if document:
                mnemosyne.inject(document, source_id)
            else:
                print("No document text entered.")

        elif command == "retrieve":
            query = input("Enter your query > ").strip()
            if query:
                mnemosyne.retrieve(query)
            else:
                print("No query entered.")
        elif command == "vstats":
            stats = mnemosyne.void_stats()
            if not stats:
                print("[VoidMemory] Disabled.")
            else:
                print("[VoidMemory] Stats:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
        elif command == "vevents":
            events = mnemosyne.void_events(limit=50, consume=False)
            if not events:
                print("[VoidMemory] No events.")
            else:
                for tick, etype, payload in events:
                    print(f"  tick={tick} type={etype} payload={payload}")
        elif command == "vtop":
            topn = 10
            if len(parts) > 1:
                try:
                    topn = int(parts[1])
                except ValueError:
                    pass
            tops = mnemosyne.void_top(topn)
            if not tops:
                print("[VoidMemory] No data.")
            else:
                print("[VoidMemory] Top composite scores:")
                for cid, score in tops:
                    print(f"  {cid}: {score:.4f}")
        elif command == "vsave":
            if len(parts) < 2:
                print("Usage: vsave <path>")
            else:
                path = parts[1]
                ok = mnemosyne.save_void_state(path)
                print(f"[VoidMemory] Save {'OK' if ok else 'FAILED'} -> {path}")
        elif command == "vload":
            if len(parts) < 2:
                print("Usage: vload <path>")
            else:
                path = parts[1]
                ok = mnemosyne.load_void_state(path)
                print(f"[VoidMemory] Load {'OK' if ok else 'FAILED'} <- {path}")

        else:
            print("Unknown command. Available commands: inject, retrieve, count, quit")