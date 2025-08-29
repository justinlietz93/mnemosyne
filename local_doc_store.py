"""
LocalDocStore: minimal persistent id->text store (SQLite, stdlib-only).

Purpose:
- Keep Qdrant "pure embeddings" by removing content-bearing payloads.
- Preserve chunk text externally so the Kernel can still build augmented prompts.
- Provide deterministic id->text lookup for retrieval results (ids).

Schema:
- Table docs(id TEXT PRIMARY KEY, text TEXT NOT NULL)

Usage:
- upsert_many(ids, texts): insert or replace many rows atomically
- get_many(ids): returns texts aligned to input order (None if missing)
- upsert_mappings(qids, orig_ids, texts): store qid→orig_id mapping and orig_id→text
- get_orig_ids_by_qids(qids): map qids back to original ids (aligned)
- get_texts_by_qids(qids): fetch texts aligned to qids via mapping
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, List, Optional, Tuple


class LocalDocStore:
    def __init__(self, path: str) -> None:
        """
        Initialize the document store at 'path'. Creates parent directories and
        table schema as needed. Uses WAL for safe concurrent readers.
        """
        self._path = os.path.abspath(path)
        d = os.path.dirname(self._path)
        if d and not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                # best-effort; rely on sqlite to error if still missing
                pass
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        try:
            # journaling mode to allow concurrent readers
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        self._init_schema()

    def _init_schema(self) -> None:
        try:
            cur = self._conn.cursor()
            # Original id -> text content
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS docs (
                    id   TEXT PRIMARY KEY,
                    text TEXT NOT NULL
                );
                """
            )
            # Qdrant id (UUID string) -> original id
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS qmap (
                    qid      TEXT PRIMARY KEY,
                    orig_id  TEXT NOT NULL
                );
                """
            )
            try:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_qmap_orig ON qmap(orig_id);")
            except Exception:
                pass
            self._conn.commit()
        except Exception:
            # leave DB in best-effort state; caller will see failures on use
            pass

    def upsert_many(self, ids: Iterable[str], texts: Iterable[str]) -> None:
        """
        Insert or replace many (id, text) rows atomically.
        """
        rows: List[Tuple[str, str]] = []
        for i, t in zip(ids, texts):
            # store literal text; do not sanitize to preserve content fidelity
            rows.append((str(i), str(t)))
        if not rows:
            return
        try:
            cur = self._conn.cursor()
            cur.executemany(
                "INSERT INTO docs(id, text) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET text=excluded.text",
                rows,
            )
            self._conn.commit()
        except Exception:
            # try to rollback on failure
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

    def upsert_mappings(self, qids: Iterable[str], orig_ids: Iterable[str], texts: Iterable[str]) -> None:
        """
        Upsert qid→orig_id mappings and orig_id→text in a single transaction.
        """
        qrows: List[Tuple[str, str]] = []
        drows: List[Tuple[str, str]] = []
        for q, o, t in zip(qids, orig_ids, texts):
            qrows.append((str(q), str(o)))
            drows.append((str(o), str(t)))
        if not qrows:
            return
        try:
            cur = self._conn.cursor()
            cur.executemany(
                "INSERT INTO qmap(qid, orig_id) VALUES(?, ?) ON CONFLICT(qid) DO UPDATE SET orig_id=excluded.orig_id",
                qrows,
            )
            cur.executemany(
                "INSERT INTO docs(id, text) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET text=excluded.text",
                drows,
            )
            self._conn.commit()
        except Exception:
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

    def get_orig_ids_by_qids(self, qids: Iterable[str]) -> List[Optional[str]]:
        """
        Map Qdrant ids (UUID strings) back to original chunk ids, aligned to input order.
        """
        ql = [str(q) for q in qids]
        if not ql:
            return []
        uniq = list(dict.fromkeys(ql).keys())
        placeholders = ",".join("?" for _ in uniq)
        try:
            cur = self._conn.cursor()
            cur.execute(f"SELECT qid, orig_id FROM qmap WHERE qid IN ({placeholders})", uniq)
            rows = cur.fetchall()
        except Exception:
            return [None for _ in ql]
        found = {str(r[0]): str(r[1]) for r in rows}
        return [found.get(k) for k in ql]

    def get_texts_by_qids(self, qids: Iterable[str]) -> List[Optional[str]]:
        """
        Fetch texts aligned to input qids using qmap then docs.
        """
        orig = self.get_orig_ids_by_qids(qids)
        return self.get_many([o if o is not None else "" for o in orig])

    def get_many(self, ids: Iterable[str]) -> List[Optional[str]]:
        """
        Fetch texts for the provided ids. Output order aligns to input ids.
        Missing ids return None.
        """
        id_list = [str(i) for i in ids]
        if not id_list:
            return []
        # Deduplicate for compact query; preserve original order via map
        uniq = list(dict.fromkeys(id_list).keys())
        placeholders = ",".join("?" for _ in uniq)
        try:
            cur = self._conn.cursor()
            cur.execute(f"SELECT id, text FROM docs WHERE id IN ({placeholders})", uniq)
            rows = cur.fetchall()
        except Exception:
            # on failure, return Nones of same length
            return [None for _ in id_list]
        found = {str(r[0]): str(r[1]) for r in rows}
        return [found.get(k) for k in id_list]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


__all__ = ["LocalDocStore"]