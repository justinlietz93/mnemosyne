import time
from mnemosyne_core import Mnemosyne

DB_PATH = "./mvm_db"
COLLECTION_NAME = "mnemosyne_core"
EMBEDDING_MODEL = 'nomic-embed-text'

mnemosyne = Mnemosyne(db_path=DB_PATH, collection_name=COLLECTION_NAME, model=EMBEDDING_MODEL)

# Test inject
source_id = f"test_inject_{int(time.time())}"
document = "This is a test document for Mnemosyne injection. It contains sample information about AI memory systems."
metadata = {"source": "test", "summary": "Test document"}
mnemosyne.inject(document, source_id, metadata=metadata)

# Test retrieve
query = "AI memory systems"
results = mnemosyne.retrieve(query, n_results=1)

print("Test completed. Check for any errors above.")