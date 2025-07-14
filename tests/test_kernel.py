import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kernel import PrometheusKernel, Mnemosyne, EMBEDDING_MODEL, MAIN_MODEL, UTILITY_MODEL

# Initialize for testing
mnemosyne = Mnemosyne(db_path="./mvm_db", collection_name="mnemosyne_core", model=EMBEDDING_MODEL)
kernel = PrometheusKernel(mnemosyne, llm_model=MAIN_MODEL, utility_model=UTILITY_MODEL, aegis_enabled=True, clean_start=True)

# Sample queries
queries = [
    "What is the capital of France?",
    "Tell me about the sun.",
    "Recent quantum computing advances 2025"
]

print("Starting Kernel test...")
for query in queries:
    print(f"\n--- Processing query: '{query}' ---")
    kernel.process_prompt(query)
    print("--- End of query processing ---\n")

print("Kernel test completed. Check for errors above.")