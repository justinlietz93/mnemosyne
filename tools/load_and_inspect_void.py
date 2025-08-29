"""Load saved void state into VoidMemoryManager and inspect top entries and territories.

Outputs:
 - Void state loaded? keys summary
 - Top N composite scores with small payload
 - Territory histogram and counts
 - Sample MemoryState dict for top item
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from void_memory import VoidMemoryManager

STATE_PATH = 'void_memory_state.json'
TOP_N = 30

print('\n=== Load void state from file ===')
if not os.path.exists(STATE_PATH):
    print(f'Missing {STATE_PATH} in cwd={os.getcwd()}')
    sys.exit(1)

try:
    manager = VoidMemoryManager.load_json(STATE_PATH)
except Exception as e:
    print('Failed to load void state:', e)
    sys.exit(1)

print('Loaded VoidMemoryManager: tick=', manager._tick, 'count=', len(manager._mem))

print('\nTop composite scores:')
for cid, score in manager.top(TOP_N):
    print(f'  {cid}: {score:.4f}')

# Territory histogram
from collections import Counter
territories = Counter()
for mid, ms in manager._mem.items():
    territories[int(ms.territory)] += 1

print('\nTerritory counts (sample sorted desc):')
for t, c in territories.most_common(20):
    print(f'  territory {t}: {c}')

# Print sample MemoryState for highest-scoring id
if manager.top(TOP_N):
    top_id, _ = manager.top(1)[0]
    ms = manager.get_state(top_id)
    if ms:
        print('\nSample MemoryState for top id:')
        try:
            print(ms.to_dict())
        except Exception:
            # fallback to repr
            print(repr(ms))

print('\nDone')
