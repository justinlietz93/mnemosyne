"""Inspect void memory state and storage (Chroma/Qdrant), and run a smoke retrieval.

Outputs:
 - VoidMemory stats
 - Top N composite-scored chunks
 - Storage counts and backend availability
 - A sample retrieval result
"""
import json
import os
import sys
import time

# Make sure repository root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mnemosyne_core import Mnemosyne, EMBEDDING_MODEL, DB_PATH, COLLECTION_NAME

VOID_STATE = 'void_memory_state.json'

print("\n=== Void Memory State File ===")
if os.path.exists(VOID_STATE):
    try:
        with open(VOID_STATE, 'r') as f:
            js = json.load(f)
        print(f"Found {VOID_STATE}: keys={list(js.keys())[:10]}...")
        # try printing a small summary
        s = js.get('stats') or js.get('meta') or {}
        print("Summary stats snippet:", {k: s.get(k) for k in ('count','tick','territories') if k in s})
    except Exception as e:
        print("Failed to read void state:", e)
else:
    print(f"No {VOID_STATE} found in cwd={os.getcwd()}")

print("\n=== Instantiate Mnemosyne (Chroma) ===")
try:
    m = Mnemosyne(db_path=DB_PATH, collection_name=COLLECTION_NAME, model=EMBEDDING_MODEL, backend='chroma', enable_void_memory=True)
    backend_used = m._backend
    print("Backend in use:", backend_used)
    try:
        if getattr(m, '_index', None) is not None:
            print("Qdrant index present. Count:", m._index.count())
        elif getattr(m, 'collection', None) is not None:
            try:
                print("Chroma collection count:", m.collection.count())
            except Exception as e:
                print("Failed to get chroma count:", e)
    except Exception as e:
        print("Storage count check failed:", e)

    if m._void_memory:
        print("Void stats:")
        vs = m.void_stats()
        for k, v in vs.items():
            print(f"  {k}: {v}")
        print("Top 20 composite scores (id,score):")
        for cid, score in m.void_top(20):
            print(f"  {cid}: {score:.4f}")
    else:
        print("Void memory disabled on Mnemosyne instance.")

    print("\nRunning a sample retrieval for 'software design' (n_results=5):")
    m.retrieve("software design", n_results=5)
except Exception as e:
    print("Failed to instantiate Mnemosyne or run inspections:", e)

print("\n=== Qdrant Availability Check ===")
try:
    mq = Mnemosyne(db_path=DB_PATH, collection_name=COLLECTION_NAME, model=EMBEDDING_MODEL, backend='qdrant', qdrant_url='http://127.0.0.1:6333', enable_void_memory=False)
    print("Requested qdrant backend; effective backend:", mq._backend)
    if getattr(mq, '_index', None) is not None:
        print("Qdrant reports count:", mq._index.count())
    else:
        print("No qdrant index attached (probably fell back to chroma).")
except Exception as e:
    print("Qdrant check failed:", e)

print("\n=== Done ===")
