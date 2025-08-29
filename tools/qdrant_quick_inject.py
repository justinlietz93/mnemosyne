"""Quick test: inject one short doc into Qdrant via Mnemosyne and show counts.
"""
import time
import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mnemosyne_core import Mnemosyne, EMBEDDING_MODEL, DB_PATH, COLLECTION_NAME

print('\n=== Qdrant quick inject test ===')
# Use a dedicated small collection name to avoid touching main data
col = 'mnemosyne_qdrant_test'
try:
    m = Mnemosyne(db_path=DB_PATH, collection_name=col, model=EMBEDDING_MODEL, backend='qdrant', qdrant_url='http://127.0.0.1:6333', enable_void_memory=False)
    print('Effective backend:', m._backend)
    if not hasattr(m, '_index') or m._index is None:
        print('No Qdrant index attached; aborting test')
        sys.exit(1)
    before = m._index.count()
    print('Count before:', before)
    doc = 'This is a tiny test document about software design and architecture.'
    m.inject(doc, source_id='qdrant_test_doc', chunk_size=2, overlap=0, metadata={'source':'quick_test'})
    # brief pause to let async payload worker finish
    time.sleep(1.0)
    after = m._index.count()
    print('Count after:', after)
    print('Injected increase:', after - before)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Qdrant quick inject failed:', e)
