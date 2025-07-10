# ==============================================================================
# Project Prometheus: Minimum Viable Mnemosyne (MVM)
# Version 0.5
#
# Agent: PrometheusAI
# Mission: To create the foundational memory-injection pipeline for our AGI.
#
# Description:
# This script provides an interactive Command-Line Interface (CLI) to interact
# with the Mnemosyne memory system, allowing for dynamic injection and retrieval.
#
# Changelog:
# v0.2 - Corrected the distance metric to 'cosine'.
# v0.3 - Added task-specific prefixes to embedding prompts.
# v0.4 - Refactored into Mnemosyne class; added text chunking.
# v0.5 - Implemented an interactive CLI for dynamic memory management.
# ==============================================================================

import ollama
import chromadb
import time
import re

# --- Configuration ---
OLLAMA_MODEL = 'nemotron:70b'
DB_PATH = "./mvm_db"
COLLECTION_NAME = "mnemosyne_core"
STORAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

class Mnemosyne:
    """
    The core memory system for our AGI. Handles memory injection and retrieval.
    """
    def __init__(self, db_path: str, collection_name: str, model: str):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = model
        self.storage_prefix = STORAGE_PREFIX
        self.query_prefix = QUERY_PREFIX
        
        print(f"Initializing Mnemosyne Core...")
        print(f"  - DB Path: {self.db_path}")
        print(f"  - Collection: {self.collection_name}")
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Mnemosyne Core initialized. Current memory count: {self.collection.count()}")

    def _chunk_text(self, text: str, chunk_size: int = 5) -> list[str]:
        """
        A simple text chunker that splits text into paragraphs or groups of sentences.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ').strip())
        if not sentences or (len(sentences) == 1 and sentences[0] == ''):
            return []
            
        chunks = []
        current_chunk = []
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def inject(self, document: str, source_id: str):
        """
        Injects a large document into memory by chunking it first.
        """
        print(f"\n--- Injecting Document: {source_id} ---")
        chunks = self._chunk_text(document)
        if not chunks:
            print("Warning: Document is empty or contains no valid sentences. Nothing to inject.")
            return

        print(f"Document split into {len(chunks)} chunks.")
        
        chunk_ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
        prefixed_chunks = [self.storage_prefix + chunk for chunk in chunks]
        
        print("Generating embeddings for all chunks...")
        try:
            embeddings = [
                ollama.embeddings(model=self.model, prompt=p_chunk)['embedding']
                for p_chunk in prefixed_chunks
            ]
            print("Embeddings generated successfully.")
        except Exception as e:
            print(f"Error during bulk embedding generation: {e}")
            return
            
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=chunk_ids
        )
        print(f"All chunks for document '{source_id}' injected successfully.")

    def retrieve(self, query: str, n_results: int = 3):
        """
        Retrieves the most relevant memory chunks based on a query.
        """
        print(f"\n--- Retrieving Memories ---")
        print(f"Query: \"{query}\"")

        prefixed_query = self.query_prefix + query

        try:
            query_embedding = ollama.embeddings(model=self.model, prompt=prefixed_query)['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        print("\nTop matching memories (chunks):")
        if not results['documents'] or not results['documents'][0]:
            print("No matching memories found.")
            return
            
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i]
            similarity_score = 1 - distance
            chunk_id = results['ids'][0][i]
            print(f"  {i+1}. Chunk ID: {chunk_id}")
            print(f"     Similarity: {similarity_score:.4f}")
            print(f"     Content: \"{doc}\"")

# --- Main Execution: Interactive CLI ---
if __name__ == "__main__":
    mnemosyne = Mnemosyne(db_path=DB_PATH, collection_name=COLLECTION_NAME, model=OLLAMA_MODEL)
    
    print("\n--- Mnemosyne Interactive CLI ---")
    print("Commands: inject, retrieve, count, quit")

    while True:
        command = input("\nEnter command > ").lower().strip()

        if command == "quit":
            print("Shutting down Mnemosyne.")
            break
        
        elif command == "count":
            print(f"Current memory count: {mnemosyne.collection.count()}")

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
        
        else:
            print("Unknown command. Available commands: inject, retrieve, count, quit")

