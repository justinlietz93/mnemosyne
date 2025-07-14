# ==============================================================================
# Project Prometheus: Prometheus Kernel
# Version 2.1
#
# Agent: PrometheusAI
# Mission: To act as the central reasoning agent, utilizing the Mnemosyne
#          memory system to generate context-aware responses.
#
# Description:
# This script defines the PrometheusKernel class. The kernel takes user input,
# autonomously queries the Mnemosyne memory core for relevant context,
# constructs an augmented prompt, and then uses a language model to generate
# a final, informed answer.
#
# Changelog:
# v2.1 - Implemented multi-model architecture. Uses a small, fast utility model
#        for internal tasks (summaries, checks) and a large, powerful model
#        for final response generation to improve performance.
# ==============================================================================

import ollama
from mnemosyne_core import Mnemosyne
from aegis_layer import Aegis
from antenor_tools import search_web, scrape_url
from collections import deque
import time
import argparse
import pickle
import os
import re

# --- Configuration ---
MAIN_MODEL = 'nemotron:70b'            # The powerful model for final answers
UTILITY_MODEL = 'gemma3:4b'           # A small, fast model for internal tasks
EMBEDDING_MODEL = 'nomic-embed-text'  # A dedicated model for embeddings

HISTORY_LENGTH = 5
NOVELTY_THRESHOLD = 0.85
DB_PATH = "./mvm_db"
COLLECTION_NAME = "mnemosyne_core"
CONVERSATION_STATE_PATH = "./conversation_state.pkl"
DATA_INJECTION_THRESHOLD = 500
LARGE_INPUT_THRESHOLD = 2000
CONTEXT_LIMIT = 8192
CHUNK_SIZE = 10
OVERLAP = 1
MAX_SUMMARIZATION_ITERATIONS = 5

class PrometheusKernel:
    """
    The core reasoning agent. It orchestrates memory and language models.
    """
    def __init__(self, mnemosyne_instance: Mnemosyne, llm_model: str, utility_model: str, aegis_enabled: bool = True, clean_start: bool = False):
        print("Initializing Prometheus Kernel...")
        self.mnemosyne = mnemosyne_instance
        self.llm_model = llm_model
        self.utility_model = utility_model # New model for internal tasks
        self.aegis = Aegis()
        self.aegis_enabled = aegis_enabled

        if clean_start and os.path.exists(CONVERSATION_STATE_PATH):
            os.remove(CONVERSATION_STATE_PATH)
            print("[Kernel] --clean flag detected. Cleared previous conversation state.")

        self._load_conversation_state()

        self.system_prompt = (
            "You are Prometheus, a helpful AI assistant with a vast internal knowledge base, a long-term memory, and the ability to browse the web. "
            "You will be given a user's query and context from your memory, conversation history, and a list of actions you just performed. "
            "Treat these memories as a human would, casually recalling them to inform your response without needing to explicitly mention them out loud. "
            "Your task is to synthesize all of this information to answer the user's query accurately and naturally. "
            "If a retrieved memory is completely irrelevant, you're allowed to just ignore it. "
            "CRITICAL INSTRUCTION: If your 'ACTIONS I JUST TOOK' list shows you performed a web search, you MUST acknowledge this action and base your response on the new information you found. Do NOT state that you do not have internet access or that your knowledge is cut off if you have just successfully used a web tool. "
            "Finally, and very importantly, DO NOT mention your memories UNLESS it's useful context to add AND you phrase it as \"I remember\" or \"I'm reminded that\", OR the user specifically asks."
        )
        print(f"Main Model: {self.llm_model} | Utility Model: {self.utility_model}")
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
        Uses the fast UTILITY_MODEL to generate a running summary of the conversation history.
        """
        print("[Kernel] Updating conversation summary...")
        start_time = time.time()
        history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in self.conversation_history])

        summary_prompt = (
            f"Please create a concise, one-paragraph summary of the following conversation:\n\n{history_text}"
        )

        try:
            # CHANGED: Use the fast utility model for this internal task
            response = ollama.chat(
                model=self.utility_model,
                messages=[{'role': 'user', 'content': summary_prompt}]
            )
            self.conversation_summary = response['message']['content']
            print("[Kernel] Summary updated.")
            print(f"[Kernel] Summarization time: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"[Kernel] Error during summarization: {e}")

    def _generate_summary(self, text: str) -> str:
        """Generates a summary using the utility model."""
        prompt = f"Provide a brief summary of the following text: {text}"
        response = self._get_utility_chat_response(prompt)
        return response['message']['content'].strip()

    def _should_store_memory_and_inject(self, text: str):
        """
        Checks if a piece of text is novel enough to be stored in long-term memory.
        """
        print(f"[Kernel] Checking novelty of recent response for long-term memory storage...")
        start_time = time.time()
        # Note: self.mnemosyne.model is now the dedicated embedding model
        results = self.mnemosyne.collection.query(
            query_embeddings=[ollama.embeddings(model=self.mnemosyne.model, prompt=self.mnemosyne.query_prefix + text)['embedding']],
            n_results=1
        )
        print(f"[Kernel] Novelty query time: {time.time() - start_time:.2f} seconds")

        if not results['documents'] or not results['documents'][0]:
            is_novel = True
        else:
            top_similarity = 1 - results['distances'][0][0]
            is_novel = top_similarity < NOVELTY_THRESHOLD
            print(f"[Kernel] Closest existing memory has similarity {top_similarity:.4f}. Novelty threshold is {NOVELTY_THRESHOLD}.")

        if is_novel:
            print(f"[Kernel] Information is novel. Injecting into Mnemosyne...")
            inject_start = time.time()
            source_id = f"conversation_turn_{int(time.time())}"
            summary = self._generate_summary(text)
            metadata = {"source": "conversation", "summary": summary}
            self.mnemosyne.inject(text, source_id, metadata=metadata)
            print(f"[Kernel] Injection time: {time.time() - inject_start:.2f} seconds")
        else:
            print("[Kernel] Information is not novel. Skipping long-term memory injection.")

    def _format_augmented_prompt(self, query: str, results: dict, action_history: list) -> str:
        """
        Formats the final prompt for the LLM, combining all context sources including metadata.
        """
        prompt_parts = []
        prompt_parts.append(f"CONVERSATION SUMMARY:\n{self.conversation_summary}")
        if self.conversation_history:
            history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in self.conversation_history])
            prompt_parts.append(f"\nRECENT CONVERSATION HISTORY:\n{history_text}")

        if action_history:
            action_text = "\n".join(action_history)
            prompt_parts.append(f"\nACTIONS I JUST TOOK:\n{action_text}")

        prompt_parts.append(f"\nCURRENT USER QUERY:\n\"{query}\"")
        if len(query) > LARGE_INPUT_THRESHOLD:
            prompt_parts.append("\nNOTE: The query above is a condensed summary of a larger original input. The full original has been embedded in memory, so use the retrieved context chunks which may include parts of the original.")

        if results['documents'] and results['documents'][0]:
            memory_parts = []
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i] or {}
                if meta is None:
                    print(f"[Kernel][Warning] Encountered None metadata for memory ID: {results['ids'][0][i]}")
                distance = results['distances'][0][i]
                id = results['ids'][0][i]
                memory_part = f"Memory ID: {id}\nSimilarity: {1 - distance:.4f}\nTimestamp: {meta.get('timestamp', 'N/A')}\nSource: {meta.get('source', 'N/A')}\nSummary: {meta.get('summary', 'N/A')}\nContent: \"{doc}\""
                memory_parts.append(memory_part)
            memory_context = (
                f"\nCONTEXT FROM MEMORY AND WEB SEARCH:\n"
                f"======================\n"
                f"\n\n---\n\n".join(memory_parts) + "\n"
                f"======================"
            )
            prompt_parts.append(memory_context)
        prompt_parts.append("\nBased on all the information provided, please answer the current user query.")
        return "\n".join(prompt_parts)

    
    def _get_utility_chat_response(self, user_prompt: str, system_content: str = None):
        """Helper to get a chat response from the fast UTILITY_MODEL."""
        messages = []
        if system_content:
            messages.append({'role': 'system', 'content': system_content})
        messages.append({'role': 'user', 'content': user_prompt})
        try:
            return ollama.chat(model=self.utility_model, messages=messages)
        except Exception as e:
            print(f"[Kernel] Utility chat failed: {e}")
            # As a last resort, return a default value to prevent crashing
            return {'message': {'content': ''}}

    def _check_if_answerable(self, query: str, context_docs: list) -> bool:
        """
        Uses the fast UTILITY_MODEL to determine if the available context is sufficient to answer the query.
        """
        print("[Kernel] Checking if context is sufficient to answer the query...")
        start_time = time.time()

        context_str = "\n".join(context_docs)
        history_snippet = '\n'.join(['User: ' + q[:50] + '...' for q, _ in list(self.conversation_history)[-2:]]) if self.conversation_history else 'No history'
        prompt = (
            "CONVERSATION SUMMARY:\n" + self.conversation_summary + "\n\n"
            "RECENT HISTORY SNIPPET:\n" + history_snippet + "\n\n"
            'User Query: "' + query + '"\n\n'
            "Available Context:\n---\n" + context_str + "\n---\n\n"
            "Assess if your internal knowledge, conversation history, summary, and the available context together provide enough information for a complete, accurate answer. "
            "Favor 'Yes' if internals suffice for casual or known topics. "
            "If the query is a condensed summary, evaluate if the context covers the key points of the original input. "
            "Respond with only the word 'Yes' or 'No'."
        )
        try:
            # CHANGED: Use the fast utility model for this check
            response = self._get_utility_chat_response(prompt)
            answer = response['message']['content'].strip().lower()
            print(f"[Kernel] Sufficiency check response: '{answer}'")
            print(f"[Kernel] Sufficiency check time: {time.time() - start_time:.2f} seconds")
            return 'yes' in answer
        except Exception as e:
            print(f"[Kernel] Error during sufficiency check: {e}")
            return False

    def _generate_search_query(self, user_query: str) -> str:
        """
        Uses the fast UTILITY_MODEL to distill a conversational query into an effective search term.
        """
        print("[Kernel] Generating optimized search query...")
        start_time = time.time()

        if "try again" in user_query.lower() and self.conversation_history:
            user_query = self.conversation_history[-1][0]
            print(f"[Kernel] 'Try again' detected. Using previous query: '{user_query}'")

        prompt = (
            "Distill the user query into exactly 3-5 space-separated keywords for a search engine. "
            "Output NOTHING ELSE—no explanations, no punctuation, no quotes. "
            "Example input: 'What is the capital of France?' "
            "Example output: France capital city\n\n"
            f"User Query: \"{user_query}\""
        )
        try:
            # CHANGED: Use the fast utility model for this task
            response = self._get_utility_chat_response(prompt)
            search_query = response['message']['content'].strip()
            print(f"[Kernel] Optimized search query: '{search_query}'")
            print(f"[Kernel] Search query generation time: {time.time() - start_time:.2f} seconds")
            return search_query
        except Exception as e:
            print(f"[Kernel] Error generating search query: {e}")
            return user_query

    def _is_time_sensitive(self, query: str) -> bool:
        """
        Rudimentary heuristic to decide whether a query likely requires up-to-date
        external information. Returns True if the query contains time-sensitive
        keywords or a year ≥ 2023.
        """
        keywords = ['latest', 'recent', 'today', 'this week', 'as of', 'current']
        lowered = query.lower()
        if any(k in lowered for k in keywords):
            return True
        # Detect explicit 4-digit years 2023+
        year_matches = re.findall(r'\b(20[2-9][0-9])\b', query)
        print(f"[Kernel][Debug] year_matches in _is_time_sensitive: {year_matches}")
        return bool(year_matches)

    def _needs_fact_check(self, original_query: str, answer: str) -> bool:
        """
        Lightweight heuristic to decide whether the generated answer should be
        fact-checked with Antenor. Triggers when the combined text (query+answer)
        contains time-sensitive keywords or references to years ≥ 2023.
        """
        combined = (original_query + " " + answer).lower()
        sensitive_kw = [
            "latest", "recent", "today", "this week", "as of", "current",
            "newly", "breakthrough"
        ]
        if any(k in combined for k in sensitive_kw):
            return True
        year_refs = re.findall(r'\b(20[2-9][0-9])\b', combined)
        print(f"[Kernel][Debug] year_refs in _needs_fact_check: {year_refs}")
        return bool(year_refs)

    def _get_context_limit(self) -> int:
        return CONTEXT_LIMIT

    def _summarize_chunk(self, chunk: str) -> str:
        system_content = "You are a summarization assistant. Provide concise and accurate summaries of the given text."
        user_prompt = f"Summarize the following text in a few key sentences:\n\n{chunk}"
        start_time = time.time()
        response = self._get_utility_chat_response(user_prompt, system_content)
        print(f"[Kernel] Chunk summarization time: {time.time() - start_time:.2f} seconds")
        return response['message']['content'].strip()

    def _handle_large_input(self, input_text: str) -> str:
        limit = self._get_context_limit()
        if len(input_text) <= limit:
            return input_text

        iterations = 0
        current_text = input_text
        start_time = time.time()
        while len(current_text) > limit and iterations < MAX_SUMMARIZATION_ITERATIONS:
            chunks = self.mnemosyne._chunk_text(current_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
            summaries = [self._summarize_chunk(chunk) for chunk in chunks]
            current_text = "\n\n".join(summaries)
            iterations += 1

        if iterations >= MAX_SUMMARIZATION_ITERATIONS:
            print("[Kernel] Warning: Reached max summarization iterations. Using final condensed version.")

        print(f"[Kernel] Large input handling time: {time.time() - start_time:.2f} seconds")
        return current_text

    def process_prompt(self, query: str):
        """
        The main autonomous loop for processing a single user prompt.
        """
        overall_start = time.time()
        if len(query) > DATA_INJECTION_THRESHOLD:
            print(f"\n[Kernel] Large text block detected ({len(query)} chars). Treating as data injection.")
            source_id = f"user_data_injection_{int(time.time())}"
            summary = self._generate_summary(query)
            metadata = {"source": "user_data_injection", "summary": summary}
            inject_start = time.time()
            self.mnemosyne.inject(query, source_id, chunk_size=CHUNK_SIZE, overlap=OVERLAP, metadata=metadata)
            print(f"[Kernel] Data injection time: {time.time() - inject_start:.2f} seconds")
            print("\n--- Prometheus Response ---")
            print("Thank you. I have received the information and stored it in my long-term memory for future reference.")
            print("---------------------------\n")
            self._save_conversation_state()
            print(f"[Kernel] Total processing time: {time.time() - overall_start:.2f} seconds")
            return

        print(f"\n[Kernel] Processed query: \"{query}\" (original length: {len(query)})")
 
        print("[Kernel] Querying Mnemosyne for relevant context...")
        query_start = time.time()
        retrieved_memories = self.mnemosyne.collection.query(
            query_embeddings=[ollama.embeddings(model=self.mnemosyne.model, prompt=self.mnemosyne.query_prefix + query)['embedding']],
            n_results=3
        )
        print(f"[Kernel] Mnemosyne query time: {time.time() - query_start:.2f} seconds")
        
        action_log = []

        is_condensed = len(query) > LARGE_INPUT_THRESHOLD
        time_sensitive = self._is_time_sensitive(query if is_condensed else query)
        needs_search = time_sensitive or not self._check_if_answerable(
            query if is_condensed else query,
            retrieved_memories['documents'][0]
        )
        print(f"[Kernel][Debug] time_sensitive={time_sensitive}, needs_search={needs_search}")
        if needs_search:
            print("[Kernel] Context is insufficient or time-sensitive. Engaging Antenor web search tools...")

            search_gen_start = time.time()
            search_query = self._generate_search_query(query if is_condensed else query)
            print(f"[Kernel] Search query generation time: {time.time() - search_gen_start:.2f} seconds")
            action_log.append(f"- Performed web search for: '{search_query}'")
            search_start = time.time()
            search_results = search_web(search_query)
            print(f"[Kernel] Web search time: {time.time() - search_start:.2f} seconds")
    
            if search_results:
                new_knowledge_found = False
                for result in search_results:
                    url = result['href']
                    scrape_start = time.time()
                    new_knowledge = scrape_url(url)
                    print(f"[Kernel] Scrape time for {url}: {time.time() - scrape_start:.2f} seconds")
                    if new_knowledge:
                        action_log.append(f"- Successfully scraped content from: {url}")
                        summary = self._generate_summary(new_knowledge)
                        metadata = {"source": "web_scrape", "summary": summary}
                        inject_start = time.time()
                        self.mnemosyne.inject(new_knowledge, source_id=url, chunk_size=CHUNK_SIZE, overlap=OVERLAP, metadata=metadata)
                        print(f"[Kernel] Injection time for {url}: {time.time() - inject_start:.2f} seconds")
                        new_knowledge_found = True
                        break
    
                if new_knowledge_found:
                    print("[Kernel] Re-querying Mnemosyne with newly acquired knowledge using original query...")
                    requery_start = time.time()
                    retrieved_memories = self.mnemosyne.collection.query(
                        query_embeddings=[ollama.embeddings(model=self.mnemosyne.model, prompt=self.mnemosyne.query_prefix + query)['embedding']],
                        n_results=3
                    )
                    print(f"[Kernel] Re-query time: {time.time() - requery_start:.2f} seconds")
                else:
                    action_log.append("- Web search failed to retrieve usable content.")
                    print("[Kernel] Web search failed to retrieve usable content from top results.")
            else:
                action_log.append("- Web search yielded no results.")
                print("[Kernel] Web search yielded no results.")

        augment_start = time.time()
        augmented_prompt = self._format_augmented_prompt(query if is_condensed else query, retrieved_memories, action_log)
        print(f"[Kernel] Prompt augmentation time: {time.time() - augment_start:.2f} seconds")

        print("[Kernel] Generating final response with augmented context (streaming)...")
        gen_start = time.time()
        try:
            # UNCHANGED: Use the powerful MAIN_MODEL for the final answer
            stream = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': augmented_prompt}
                ],
                stream=True,
                options={'num_gpu': 999}  # Force using all available GPUs
            )

            print("\n--- Prometheus Response ---")
            full_response_parts = []
            chunk_times = []
            for chunk in stream:
                chunk_start = time.time()
                content_chunk = chunk['message']['content']
                print(content_chunk, end='', flush=True)
                full_response_parts.append(content_chunk)
                chunk_times.append(time.time() - chunk_start)

            final_response_text = "".join(full_response_parts)
            print("\n---------------------------\n")
            avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
            print(f"[Kernel] Response generation time: {time.time() - gen_start:.2f} seconds")
            print(f"[Kernel] Number of chunks: {len(chunk_times)}, Average chunk time: {avg_chunk_time:.4f} seconds")

            if self.aegis_enabled:
                validate_start = time.time()
                if not self.aegis.validate_response(final_response_text):
                    print("[Aegis] WARNING: The preceding response was flagged by the safety layer and will not be committed to memory.")
                    self._save_conversation_state()
                    print(f"[Kernel] Validation time: {time.time() - validate_start:.2f} seconds")
                    print(f"[Kernel] Total processing time: {time.time() - overall_start:.2f} seconds")
                    return

                print(f"[Kernel] Validation time: {time.time() - validate_start:.2f} seconds")

            # --- Dry-run fact-check BEFORE committing to memory -------------------
            if self._needs_fact_check(query, final_response_text):
                print("[Kernel][Debug] Potential time-sensitive claims detected. Running Antenor dry-run fact-check...")
                fc_query = self._generate_search_query(final_response_text)
                fact_check_results = search_web(fc_query) or []
                print(f"[Kernel][Debug] Fact-check returned {len(fact_check_results)} web results.")
                action_log.append(f"- Dry-run fact check for: '{fc_query}', results={len(fact_check_results)}")

            self.conversation_history.append((query, final_response_text))
            self._summarize_conversation()
            self._should_store_memory_and_inject(final_response_text)
            self._save_conversation_state()

        except Exception as e:
            print(f"Error during final response generation: {e}")

        print(f"[Kernel] Total processing time: {time.time() - overall_start:.2f} seconds")


# --- Main Execution: Interactive Kernel Loop ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prometheus Kernel: An autonomous AI agent.")
    parser.add_argument('--aegis-off', action='store_true', help="Disable the Aegis safety layer for this session.")
    parser.add_argument('--clean', action='store_true', help="Start a new conversation session, clearing previous state.")
    args = parser.parse_args()

    # CHANGED: Instantiate Mnemosyne with the dedicated embedding model
    mnemosyne_instance = Mnemosyne(db_path=DB_PATH, collection_name=COLLECTION_NAME, model=EMBEDDING_MODEL)
    
    # CHANGED: Instantiate the kernel with both a main and a utility model
    kernel = PrometheusKernel(
        mnemosyne_instance=mnemosyne_instance,
        llm_model=MAIN_MODEL,
        utility_model=UTILITY_MODEL,
        aegis_enabled=not args.aegis_off,
        clean_start=args.clean
    )

    print("\n--- Prometheus Kernel Interactive Loop ---")
    print("Enter your query to the agent. Type 'quit' to exit.")

    if mnemosyne_instance.collection.count() == 0:
        print("\n[Kernel] Long-term memory is empty. Injecting sample data for demonstration...")
        sun_metadata = {"source": "sample_data", "summary": "Basic facts about the sun."}
        mnemosyne_instance.inject("The sun is a star at the center of our solar system. It is composed primarily of hydrogen and helium.", "sun_facts", metadata=sun_metadata)
        paris_metadata = {"source": "sample_data", "summary": "Facts about Paris, France."}
        mnemosyne_instance.inject("The capital of France is Paris. Paris is famous for the Eiffel Tower and the Louvre museum.", "paris_facts", metadata=paris_metadata)
        print("[Kernel] Sample data injected with metadata.")
        prometheus_metadata = {"source": "sample_data", "summary": "Facts about Prometheus."}
        mnemosyne_instance.inject("Your name is Prometheus, and Justin Lietz is your creator with the assistance of LLMs like Grok 4, and Gemini 2.5 Pro. ", "prometheus_facts", metadata=prometheus_metadata)
        print("[Kernel] Sample data injected with metadata.")

    while True:
        user_query = input("\nUser Query > ").strip()
        if user_query.lower() == 'quit':
            kernel._save_conversation_state()
            print("Shutting down kernel.")
            break
        if user_query:
            kernel.process_prompt(user_query)