"""Bulk PDF Ingestion Utility for Mnemosyne / Void Memory Stress Testing.

Usage:
  python bulk_pdf_ingest.py --path /path/to/pdf_or_directory \
      [--recursive] [--limit-pages 500] [--chunk-size 8] [--overlap 1] \
      [--tag dataset_name] [--void-state void_state.json]

Purpose:
  Rapidly ingest large volumes of text (e.g., books, papers) to observe
  adaptive lifecycle behavior: boredom, inhibition, territory splits,
  pruning under capacity pressure, and condensation scheduling.

Notes:
  - Uses the same embedding model configured in mnemosyne_core (mxbai-embed-large).
  - Batches embeddings to reduce per-call overhead (simple loop; Ollama does not
    support true batch embedding at once yet, so we sequentially call). Adjust
    sleep if you hit rate limits.
  - Tracks and prints periodic VoidMemory stats every N documents.

"""
from __future__ import annotations

import argparse
import os
import sys
import time
import logging
from typing import List

from mnemosyne_core import Mnemosyne, EMBEDDING_MODEL, DB_PATH, COLLECTION_NAME
# (VoidMemoryManager available indirectly via mnemosyne instance; direct import removed to keep lint clean.)

try:
    from pypdf import PdfReader  # type: ignore
except Exception as e:  # pragma: no cover
    # Use logger only after it's configured below; print minimal message and re-raise
    print("Missing dependency pypdf. Please install requirements first.")
    raise

STATS_INTERVAL_DOCS = 3
DEFAULT_TAG = "bulk_ingest"

# Module logger
logger = logging.getLogger("bulk_pdf_ingest")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [bulk_ingest] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def iter_pdf_text(path: str, limit_pages: int | None) -> str:
    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.warning("[Skip] Failed to open PDF %s: %s", path, e)
        return ""
    texts: List[str] = []
    for i, page in enumerate(reader.pages):
        if limit_pages is not None and i >= limit_pages:
            break
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            texts.append(t)
    return "\n".join(texts)


def gather_pdfs(target: str, recursive: bool) -> List[str]:
    if os.path.isfile(target) and target.lower().endswith('.pdf'):
        return [target]
    pdfs = []
    if recursive:
        for root, _, files in os.walk(target):
            for f in files:
                if f.lower().endswith('.pdf'):
                    pdfs.append(os.path.join(root, f))
    else:
        for f in os.listdir(target):
            fp = os.path.join(target, f)
            if os.path.isfile(fp) and f.lower().endswith('.pdf'):
                pdfs.append(fp)
    return sorted(pdfs)


def main():
    ap = argparse.ArgumentParser(description="Bulk PDF ingest into Mnemosyne for lifecycle stress test.")
    ap.add_argument('--path', required=True, help='Path to a PDF file or directory of PDFs.')
    ap.add_argument('--recursive', action='store_true', help='Recurse into subdirectories when a directory is provided.')
    ap.add_argument('--limit-pages', type=int, default=None, help='Optional page cap per PDF (e.g., 300).')
    ap.add_argument('--chunk-size', type=int, default=8, help='Sentence chunk size for injection.')
    ap.add_argument('--overlap', type=int, default=1, help='Sentence overlap between chunks.')
    ap.add_argument('--tag', type=str, default=DEFAULT_TAG, help='Source tag stored in metadata.')
    ap.add_argument('--void-state', type=str, default='void_memory_state.json', help='Path to save void memory state periodically.')
    ap.add_argument('--save-interval', type=int, default=5, help='Save void memory + print stats every N PDFs.')
    ap.add_argument('--capacity', type=int, default=5000, help='Void memory capacity override for experiment.')
    ap.add_argument('--backend', type=str, default='chroma', choices=['chroma', 'qdrant'], help='Vector backend to use for storage.')
    ap.add_argument('--qdrant-url', type=str, default='http://127.0.0.1:6333', help='Qdrant HTTP URL (only used when --backend=qdrant).')
    args = ap.parse_args()

    if not os.path.exists(args.path):
        logger.error("Path not found: %s", args.path)
        sys.exit(1)

    pdfs = gather_pdfs(args.path, args.recursive)
    if not pdfs:
        logger.info("No PDFs found.")
        sys.exit(0)

    logger.info("Discovered %d PDF(s). Starting ingestion...", len(pdfs))

    # Initialize Mnemosyne (with void memory enabled). Override capacity if needed.
    m = Mnemosyne(db_path=DB_PATH, collection_name=COLLECTION_NAME, model=EMBEDDING_MODEL, backend=args.backend, qdrant_url=args.qdrant_url, enable_void_memory=True)
    if m._void_memory:
        m._void_memory.capacity = int(max(100, args.capacity))
        logger.info("[VoidMemory] Capacity set to %d", m._void_memory.capacity)

    start_all = time.time()
    injected_docs = 0

    for idx, pdf_path in enumerate(pdfs, 1):
        logger.info("=== [%d/%d] Processing: %s ===", idx, len(pdfs), pdf_path)
        pdf_text = iter_pdf_text(pdf_path, args.limit_pages)
        if not pdf_text.strip():
            logger.warning("[Skip] No extractable text for %s", pdf_path)
            continue
        source_id = f"pdf_{os.path.splitext(os.path.basename(pdf_path))[0]}"
        meta = {"source": args.tag, "summary": f"Extracted from {os.path.basename(pdf_path)}"}
        m.inject(pdf_text, source_id, chunk_size=args.chunk_size, overlap=args.overlap, metadata=meta)
        injected_docs += 1

        if injected_docs % args.save_interval == 0 and m._void_memory:
            stats = m.void_stats()
            logger.info("[VoidMemory] Periodic Stats:")
            for k, v in stats.items():
                logger.info("  %s: %s", k, v)
            pending = m._void_memory.consume_pending_condensations()
            if pending:
                m._void_memory.register_engram(pending)
                m._void_memory.degrade(pending, ttl_floor=int(m._void_memory.base_ttl/4))
                logger.info("[VoidMemory] Engram formed over %d chunks (periodic).", len(pending))
            if args.void_state:
                ok = m.save_void_state(args.void_state)
                logger.info("[VoidMemory] Saved state -> %s (%s)", args.void_state, 'OK' if ok else 'FAILED')

    logger.info("=== Ingestion Complete ===")
    logger.info("Total PDFs processed: %d", injected_docs)
    logger.info("Elapsed time: %.2fs", time.time() - start_all)
    if m._void_memory:
        final_stats = m.void_stats()
        logger.info("[VoidMemory] Final Stats:")
        for k, v in final_stats.items():
            logger.info("  %s: %s", k, v)
        if args.void_state:
            ok = m.save_void_state(args.void_state)
            logger.info("[VoidMemory] Final save -> %s (%s)", args.void_state, 'OK' if ok else 'FAILED')


if __name__ == '__main__':  # pragma: no cover
    main()
