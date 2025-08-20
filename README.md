# Mnemosyne

**Status:** 🚧 _Active Development - Not Yet Production Ready_ 🚧

Mnemosyne is the memory module for the Project Prometheus AGI, providing robust, persistent storage and retrieval of knowledge via vector embeddings. It is designed to integrate with AGI systems, enabling them to "remember" and leverage past experiences and context.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Interface (CLI)](#command-line-interface-cli)
- [Configuration](#configuration)
- [Development & Contribution](#development--contribution)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Mnemosyne is a core component of the Prometheus AGI project, responsible for storing, chunking, and retrieving textual information using advanced embedding models and a vector database backend. It enables AGI systems to query and inject knowledge into a persistent memory store, supporting richer, context-aware interactions.

---

## Features

- **Document Injection:** Add new documents to the memory system, automatically chunked and embedded.
- **Contextual Retrieval:** Query the memory for relevant information using semantic similarity.
- **Pluggable Embedding Models:** Integrate with embedding model services (e.g., Ollama).
- **Vector Database Backend:** Uses ChromaDB for fast, persistent vector storage.
- **Interactive CLI:** Command-line interface for manual testing and operations.
- **Modular Design:** Easily extendable for new embedding models or database backends.

---

## Architecture

```mermaid
graph TD
  A[Prometheus AGI] --> B[Mnemosyne]
  B --> C[ChromaDB (Vector DB)]
  B --> D[Embedding Service (Ollama)]
  E[User/CLI] --> B
```

- **mnemosyne_core.py:** Main orchestrator class for memory operations.
- **Ollama Model:** Generates vector embeddings for both documents and queries.
- **ChromaDB:** Stores and indexes document embeddings for retrieval.
- **CLI:** Provides an interface for manual injection and retrieval.

---

## Requirements

- Python 3.9+
- [ChromaDB](https://www.trychroma.com/) (Python package)
- Ollama embedding service (running and accessible)
- Other dependencies as listed in `requirements.txt`

---

## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/justinlietz93/mnemosyne.git
    cd mnemosyne
    ```

2. **Install dependencies**

    It's recommended to use a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Set up ChromaDB**

    Ensure ChromaDB is installed and accessible. Install via pip if needed:

    ```bash
    pip install chromadb
    ```

4. **Set up the Ollama embedding service**

    - [Ollama](https://ollama.ai/) must be installed and running.
    - Configure the model endpoint URL and key as needed (see [Configuration](#configuration)).

---

## Quick Start

Once installed, you can run the CLI for basic memory injection and retrieval:

```bash
python mnemosyne/mnemosyne_core.py
```

- You will see an interactive prompt.
- Use provided commands to inject text or perform a retrieval query.

**Example:**

```text
> inject "Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals."
Memory injected successfully.

> retrieve "What is artificial intelligence?"
Top result: Artificial intelligence is intelligence demonstrated by machines, in contrast to...
```

---

## Command-Line Interface (CLI)

The CLI supports the following commands:

- `inject "<text>"`: Add a new document to memory.
- `retrieve "<query>"`: Search memory for relevant information.
- `exit` or `quit`: Exit the CLI.

_The CLI is still experimental – feedback and bug reports are welcome!_

---

## Configuration

Configuration values (such as the embedding model, database path, etc.) are set in `mnemosyne_core.py`:

- `OLLAMA_MODEL`: Name of the embedding model used by Ollama.
- `DB_PATH`: Location for ChromaDB storage.
- `COLLECTION_NAME`: Name of the vector collection.

You may also use environment variables or a `.env` file (feature coming soon).

---

## Development & Contribution

**This project is under active development and contributions are welcome!**

### To run tests (if present):

```bash
pytest
```

### To contribute:

1. Fork the repo and create your branch (`git checkout -b feature/fooBar`)
2. Commit your changes (`git commit -am 'Add some fooBar'`)
3. Push to the branch (`git push origin feature/fooBar`)
4. Create a new Pull Request

Please see [CONTRIBUTING.md](CONTRIBUTING.md) (to be created) for more details.

---

## Troubleshooting

- **ChromaDB connection errors:** Ensure the `DB_PATH` is writable and ChromaDB is installed.
- **Ollama not responding:** Verify the embedding service is running and the endpoint/model name matches.
- **CLI not starting:** Check Python version and that all dependencies are installed.

If you're stuck, please open an issue or discussion on GitHub.

---

## Roadmap

- [ ] Add REST API for remote memory operations
- [ ] Support for additional embedding providers
- [ ] Advanced chunking and pre-processing strategies
- [ ] Persistent configuration via YAML/JSON files
- [ ] Integration tests and CI/CD
- [ ] Docker support

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Built as part of Project Prometheus AGI
- Vector database: [ChromaDB](https://www.trychroma.com/)
- Embedding service: [Ollama](https://ollama.ai/)
- Visualizations: [Mermaid](https://mermaid-js.github.io/)

---

**NOTE:** Mnemosyne is _not_ production-ready. Expect breaking changes and incomplete features as development continues!
