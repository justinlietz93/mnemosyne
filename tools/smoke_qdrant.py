import sys
import os
# Ensure repo root is on sys.path when running from tools/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mnemosyne_core import Mnemosyne, EMBEDDING_MODEL, DB_PATH, COLLECTION_NAME

text = "This is a tiny test document about AI. It mentions Mnemosyne and Qdrant integration."

m = Mnemosyne(db_path=DB_PATH, collection_name=COLLECTION_NAME, model=EMBEDDING_MODEL, backend='qdrant', qdrant_url='http://127.0.0.1:6333', enable_void_memory=True)

print('Injecting...')
m.inject(text, source_id='smoke_test_1', chunk_size=4, overlap=1, metadata={'source': 'smoke'})

print('Retrieving...')
m.retrieve('Mnemosyne Qdrant integration test', n_results=3)

print('Done')
