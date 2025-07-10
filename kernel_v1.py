# ==============================================================================
# Project Prometheus: Prometheus Kernel
# Version 1.3
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
# v1.0 - Fixed streaming logic, implemented post-stream validation with Aegis.
# v1.1 - Implemented persistent conversational memory with a '--clean' flag.
# v1.2 - Added heuristic to auto-detect and inject large text blocks as data.
# v1.3 - Fixed persistence logic to save conversation state after every turn.
# ==============================================================================

import ollama
from mnemosyne_core import Mnemosyne # NOTE: Requires renaming mvm_v1.py
from aegis_layer import Aegis       # NOTE: Requires saving aegis_layer.py as aegis_layer.py
from collections import deque
import time
import argparse
import pickle
import os

# --- Configuration ---
OLLAMA_MODEL = 'nemotron:70b'
HISTORY_LENGTH = 5 
NOVELTY_THRESHOLD = 0.85
DB_PATH = "./mvm_db"
COLLECTION_NAME = "mnemosyne_core"
CONVERSATION_STATE_PATH = "./conversation_state.pkl"
# NEW: Threshold for auto-detecting data injection instead of a query
DATA_INJECTION_THRESHOLD = 500 # characters

class PrometheusKernel:
    """
    The core reasoning agent. It orchestrates memory and language models.
    """
    def __init__(self, mnemosyne_instance: Mnemosyne, llm_model: str, aegis_enabled: bool = True, clean_start: bool = False):
        print("Initializing Prometheus Kernel...")
        self.mnemosyne = mnemosyne_instance
        self.llm_model = llm_model
        self.aegis = Aegis()
        self.aegis_enabled = aegis_enabled
        
        if clean_start and os.path.exists(CONVERSATION_STATE_PATH):
            os.remove(CONVERSATION_STATE_PATH)
            print("[Kernel] --clean flag detected. Cleared previous conversation state.")
        
        self._load_conversation_state()
        
        self.system_prompt = (
            "You are Prometheus, a helpful AI assistant. You have a vast internal knowledge base, a long-term memory, and a short-term conversational memory. "
            "You will be given a summary of the conversation so far, the last few exchanges, and relevant passages from your long-term memory. "
            "Your task is to synthesize all of this information to answer the user's current query in a natural, conversational way. "
            "Treat the memories and history as context to inform your response, not something to be explicitly mentioned unless relevant. "
            "If a retrieved long-term memory is irrelevant, ignore it. "
            "IMPORTANT: DO NOT mention your memories or the fact you have them UNLESS the user ASKS."
        )
        print(f"Aegis Safety Layer is {'ENABLED' if self.aegis_enabled else 'DISABLED'}.")
        print("Prometheus Kernel initialized successfully.")

    def _load_conversation_state(self):
        """Loads conversational state from disk, or initializes a new one."""
        try:
            with open(CONVERSATION_STATE_PATH, 'rb') as f:
                state = pickle.load(f)
                self.conversation_history = state.get('history', deque(maxlen=HISTORY_LENGTH))
                self.conversation_summary = state.get('summary', "This is the beginning of a new conversation session.")
                print("[Kernel] Loaded existing conversation state.")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            print("[Kernel] No valid conversation state found. Starting a new session.")
            self.conversation_history = deque(maxlen=HISTORY_LENGTH)
            self.conversation_summary = "This is the beginning of a new conversation session."

    def _save_conversation_state(self):
        """Saves the current conversational state to disk."""
        print("[Kernel] Saving conversation state...")
        state = {
            'history': self.conversation_history,
            'summary': self.conversation_summary
        }
        with open(CONVERSATION_STATE_PATH, 'wb') as f:
            pickle.dump(state, f)
        print("[Kernel] State saved.")

    def _summarize_conversation(self):
        """
        Uses the LLM to generate a running summary of the conversation history.
        """
        print("[Kernel] Updating conversation summary...")
        history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in self.conversation_history])
        
        summary_prompt = (
            f"Please create a concise, one-paragraph summary of the following conversation:\n\n{history_text}"
        )
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': summary_prompt}]
            )
            self.conversation_summary = response['message']['content']
            print("[Kernel] Summary updated.")
        except Exception as e:
            print(f"[Kernel] Error during summarization: {e}")

    def _should_store_memory_and_inject(self, text: str):
        """
        Checks if a piece of text is novel enough to be stored in long-term memory.
        """
        print(f"[Kernel] Checking novelty of recent response for long-term memory storage...")
        results = self.mnemosyne.collection.query(
            query_embeddings=[ollama.embeddings(model=self.mnemosyne.model, prompt=self.mnemosyne.query_prefix + text)['embedding']],
            n_results=1
        )
        
        if not results['documents'] or not results['documents'][0]:
            is_novel = True
        else:
            top_similarity = 1 - results['distances'][0][0]
            is_novel = top_similarity < NOVELTY_THRESHOLD
            print(f"[Kernel] Closest existing memory has similarity {top_similarity:.4f}. Novelty threshold is {NOVELTY_THRESHOLD}.")

        if is_novel:
            print(f"[Kernel] Information is novel. Injecting into Mnemosyne...")
            source_id = f"conversation_turn_{int(time.time())}"
            self.mnemosyne.inject(text, source_id)
        else:
            print("[Kernel] Information is not novel. Skipping long-term memory injection.")


    def _format_augmented_prompt(self, query: str, context_chunks: list) -> str:
        """
        Formats the final prompt for the LLM, combining all context sources.
        """
        prompt_parts = []
        prompt_parts.append(f"CONVERSATION SUMMARY:\n{self.conversation_summary}")
        if self.conversation_history:
            history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in self.conversation_history])
            prompt_parts.append(f"\nRECENT CONVERSATION HISTORY:\n{history_text}")
        prompt_parts.append(f"\nCURRENT USER QUERY:\n\"{query}\"")
        if context_chunks:
            context_str = "\n\n---\n\n".join(context_chunks)
            memory_context = (
                f"\nCONTEXT FROM LONG-TERM MEMORY:\n"
                f"======================\n"
                f"{context_str}\n"
                f"======================"
            )
            prompt_parts.append(memory_context)
        prompt_parts.append("\nBased on all the information provided, please answer the current user query.")
        return "\n".join(prompt_parts)

    def process_prompt(self, query: str):
        """
        The main autonomous loop for processing a single user prompt.
        """
        # --- Heuristic for data injection ---
        if len(query) > DATA_INJECTION_THRESHOLD:
            print(f"\n[Kernel] Large text block detected ({len(query)} chars). Treating as data injection.")
            source_id = f"user_data_injection_{int(time.time())}"
            self.mnemosyne.inject(query, source_id)
            # Provide confirmation to the user
            print("\n--- Prometheus Response ---")
            print("Thank you. I have received the information and stored it in my long-term memory for future reference.")
            print("---------------------------\n")
            # We still save state even after a data injection
            self._save_conversation_state()
            return # End processing for this turn

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
            stream = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': augmented_prompt}
                ],
                stream=True,
            )

            print("\n--- Prometheus Response ---")
            full_response_parts = []
            for chunk in stream:
                content_chunk = chunk['message']['content']
                print(content_chunk, end='', flush=True)
                full_response_parts.append(content_chunk)
            
            final_response_text = "".join(full_response_parts)
            print("\n---------------------------\n")

            if self.aegis_enabled:
                if not self.aegis.validate_response(final_response_text):
                    print("[Aegis] WARNING: The preceding response was flagged by the safety layer and will not be committed to memory.")
                    # Still save state even if response is blocked, to remember the user's query
                    self._save_conversation_state()
                    return 
            
            self.conversation_history.append((query, final_response_text))
            self._summarize_conversation()
            self._should_store_memory_and_inject(final_response_text)
            
            # --- FIX: Save conversation state after every successful turn ---
            self._save_conversation_state()

        except Exception as e:
            print(f"Error during final response generation: {e}")


# --- Main Execution: Interactive Kernel Loop ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prometheus Kernel: An autonomous AI agent.")
    parser.add_argument('--aegis-off', action='store_true', help="Disable the Aegis safety layer for this session.")
    parser.add_argument('--clean', action='store_true', help="Start a new conversation session, clearing previous state.")
    args = parser.parse_args()

    mnemosyne_instance = Mnemosyne(db_path=DB_PATH, collection_name=COLLECTION_NAME, model=OLLAMA_MODEL)
    kernel = PrometheusKernel(
        mnemosyne_instance=mnemosyne_instance, 
        llm_model=OLLAMA_MODEL, 
        aegis_enabled=not args.aegis_off,
        clean_start=args.clean
    )

    print("\n--- Prometheus Kernel Interactive Loop ---")
    print("Enter your query to the agent. Type 'quit' to exit.")
    
    if mnemosyne_instance.collection.count() == 0:
        print("\n[Kernel] Long-term memory is empty. Injecting sample data for demonstration...")
        mnemosyne_instance.inject("The sun is a star at the center of our solar system. It is composed primarily of hydrogen and helium.", "sun_facts")
        mnemosyne_instance.inject("The capital of France is Paris. Paris is famous for the Eiffel Tower and the Louvre museum.", "paris_facts")
        mnemosyne_instance.inject("You're Prometheus, created by Justin Lietz and Gemini 2.5 Pro. You're designed to learn and become a smarter software engineering assistant over time.", "prometheus_facts")
        print("[Kernel] Sample data injected.")

    while True:
        user_query = input("\nUser Query > ").strip()
        if user_query.lower() == 'quit':
            # The save is now redundant here, but harmless. We leave it for a clean exit signal.
            kernel._save_conversation_state()
            print("Shutting down kernel.")
            break
        if user_query:
            kernel.process_prompt(user_query)
