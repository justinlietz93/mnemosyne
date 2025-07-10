# ==============================================================================
# Project Prometheus: Prometheus Kernel
# Version 0.4
#
# Agent: PrometheusAI
# Mission: To act as the central reasoning agent, utilizing the Mnemosyne
#          memory system to generate context-aware responses.
#
# Description:
# This script defines the PrometheusKernel class. The kernel takes user input,
# autonomously queries the Mnemosyne memory core for relevant context,
# constructs an augmented prompt, and then uses a language model to generate
# a final, informed answer. This is the first step toward true autonomy.
#
# Changelog:
# v0.2 - Updated system prompt to synthesize memory with internal knowledge.
# v0.3 - Implemented streaming for the final LLM response for better UX.
# v0.4 - System prompt updated to allow the agent to ignore irrelevant memories.
# ==============================================================================

import ollama
from mnemosyne_core import Mnemosyne # NOTE: Requires renaming mvm_v1.py

# --- Configuration ---
OLLAMA_MODEL = 'nemotron:70b'

class PrometheusKernel:
    """
    The core reasoning agent. It orchestrates memory and language models.
    """
    def __init__(self, mnemosyne_instance: Mnemosyne, llm_model: str):
        print("Initializing Prometheus Kernel...")
        self.mnemosyne = mnemosyne_instance
        self.llm_model = llm_model
        # The system prompt defines the agent's core behavior
        self.system_prompt = (
            "You are Prometheus, a helpful AI assistant with a vast internal knowledge base and a long-term memory system. "
            "When you receive a query, included will be relevant memories to act as a refresher or additional context. "
            "Treat these memories as a human would, casually recalling them to inform your response without needing to explicitly mention them out loud. "
            "Your task is to answer the user's query accurately and naturally. "
            "Synthesize the retrieved memories with your own general knowledge to provide a comprehensive answer. "
            "If the memories provide a direct answer, prioritize that information. If a retrieved memory is completely irrelevant to the user's query, "
            "you're allowed to just ignore it and rely on your internal knowledge. IMPORTANT: DO NOT mention these memories UNLESS the user ASKS. "
        )
        print("Prometheus Kernel initialized successfully.")

    def _format_augmented_prompt(self, query: str, context_chunks: list) -> str:
        """
        Formats the final prompt for the LLM, combining the original query
        with the retrieved context.
        """
        if not context_chunks:
            return f"User Query: {query}\n\nContext from Memory: No relevant context found in memory."

        context_str = "\n\n---\n\n".join(context_chunks)
        
        prompt = (
            f"User Query: {query}\n\n"
            f"Context from Memory:\n"
            f"======================\n"
            f"{context_str}\n"
            f"======================\n\n"
            f"Based on the provided context and your own knowledge, please answer the user's query."
        )
        return prompt

    def process_prompt(self, query: str):
        """
        The main autonomous loop for processing a single user prompt.
        """
        print(f"\n[Kernel] Received query: \"{query}\"")
        
        print("[Kernel] Querying Mnemosyne for relevant context...")
        retrieved_memories = self.mnemosyne.collection.query(
            query_embeddings=[ollama.embeddings(model=self.mnemosyne.model, prompt=self.mnemosyne.query_prefix + query)['embedding']],
            n_results=3
        )
        
        context_docs = retrieved_memories['documents'][0]
        
        if context_docs:
             print(f"[Kernel] Found {len(context_docs)} relevant memory chunks.")
        else:
            print("[Kernel] No relevant memories found.")

        augmented_prompt = self._format_augmented_prompt(query, context_docs)
        
        print("[Kernel] Generating final response with augmented context (streaming)...")
        try:
            # NEW: Set stream=True to get a generator of response chunks
            stream = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': augmented_prompt}
                ],
                stream=True,
            )

            print("\n--- Prometheus Response ---")
            # Iterate through the stream and print each chunk as it arrives
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            
            print("\n---------------------------\n")

        except Exception as e:
            print(f"Error during final response generation: {e}")


# --- Main Execution: Interactive Kernel Loop ---
if __name__ == "__main__":
    mnemosyne_instance = Mnemosyne(db_path="./mvm_db", collection_name="mnemosyne_core", model=OLLAMA_MODEL)
    kernel = PrometheusKernel(mnemosyne_instance=mnemosyne_instance, llm_model=OLLAMA_MODEL)

    print("\n--- Prometheus Kernel Interactive Loop ---")
    print("Enter your query to the agent. Type 'quit' to exit.")
    
    if mnemosyne_instance.collection.count() == 0:
        print("\n[Kernel] Memory is empty. Injecting sample data for demonstration...")
        mnemosyne_instance.inject("The sun is a star at the center of our solar system. It is composed primarily of hydrogen and helium.", "sun_facts")
        mnemosyne_instance.inject("The capital of France is Paris. Paris is famous for the Eiffel Tower and the Louvre museum.", "paris_facts")
        print("[Kernel] Sample data injected.")

    while True:
        user_query = input("\nUser Query > ").strip()
        if user_query.lower() == 'quit':
            print("Shutting down kernel.")
            break
        if user_query:
            kernel.process_prompt(user_query)

