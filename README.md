# Mnemosyne Void Learning Demo — Quickstart

This repository demonstrates event‑driven learning over a sparse memory field (VoidMemory) built on top of a pure embeddings index (Qdrant). The main demo script logs learning metrics over rounds of retrieval reinforcement, renders plots, and performs an A/B evaluation using a small LLM on a synthetic QA set. Two falsification controls are included to make results compelling beyond reasonable doubt.

Core demo script:
- tools/void_learning_report.py

Key functions (for reference):
- Drive learning via retrieval: def drive_reinforcement(...)
- Build retrieval context for A/B: def build_context_from_results(...)
- Controls: mismatch queries and reinforcement‑OFF


## Prerequisites

1) Python
- Python 3.10+ recommended

2) Qdrant (vector DB)
- Docker (recommended):
  docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
- Or native installation: https://qdrant.tech

3) Ollama (embeddings + small LLM)
- Install: https://ollama.com
- Pull models (example choices; adjust as needed):
  ollama pull mxbai-embed-large
  ollama pull gemma3:4b

4) Python dependencies
- From repo root:
  pip install -r requirements.txt


## TL;DR — One‑liner to run the demo

Synthetic mode with A/B QA and both controls:
python tools/void_learning_report.py --mode synthetic --rounds 12 --topk 5 --plots-out output/void_report --do-qa --do-control-mismatch --do-control-off

Artifacts will be in output/void_report/ (and subfolders control_mismatch/ and control_off/).


## What the demo does

- Ingests a small synthetic corpus across four clusters (graphs, neural, physics, bridges).
- Repeatedly retrieves with relevant queries to drive reinforcement in the void memory field.
- Logs time‑series metrics and renders plots (avg_conf, avg_mass, avg_heat, churn_pairs, engrams, territories).
- Performs small‑LLM A/B on synthetic QA: pre vs post reinforcement.
- Runs two controls:
  - Mismatch queries (learn ON, but irrelevant prompts).
  - Reinforcement OFF (bypass learning).


## How to run

1) Ensure services are running
- Qdrant: http://127.0.0.1:6333
- Ollama: embeddings model and small LLM are available

2) Install Python deps
pip install -r requirements.txt

3) Run synthetic learning demo (recommended)
python tools/void_learning_report.py --mode synthetic --rounds 12 --topk 5 --plots-out output/void_report --do-qa

4) Add controls for falsification
- Mismatch (irrelevant prompts):
  python tools/void_learning_report.py --mode synthetic --rounds 12 --topk 5 --plots-out output/void_report --do-control-mismatch
- Reinforcement OFF (flat expected):
  python tools/void_learning_report.py --mode synthetic --rounds 12 --topk 5 --plots-out output/void_report --do-control-off

5) Optional: Books mode (PDF ingestion)
- Put PDFs under ./books or specify --books to a PDF file or directory:
  python tools/void_learning_report.py --mode books --books ./books --rounds 8 --topk 5 --plots-out output/void_report
- Note: A/B QA is only defined for synthetic mode (known ground truth keywords).


## Outputs and where to look

Default: output/void_report/

- metrics.json, metrics.csv
- Plots:
  - avg_conf.png — average confidence (should trend upward with learning)
  - avg_mass.png — average mass (growth indicates consolidation)
  - avg_heat.png — activity proxy
  - churn_pairs.png — co‑retrieval churn (association dynamics)
  - engrams.png — engram counts (internal consolidations)
  - territories.png — territory count (splits/merges)
- spearman.json — Spearman ρ between metrics and round
- A/B (synthetic only):
  - qa_pre.json, qa_post.json
  - ab_accuracy.png
  - qa_ab_summary.json (pre, post, delta)

Controls:
- control_mismatch/ — same set of metrics/plots for mismatch queries
- control_off/ — same set of metrics/plots for reinforcement‑OFF


## Acceptance criteria (evidence thresholds)

To be “compelling beyond a shadow of a doubt,” check:

1) A/B lift on synthetic QA
- Let M be QA items, a_i the per‑item keyword accuracy (0..1).
- Pre mean A_pre = (1/M)∑ a_i (pre), Post mean A_post (post).
- Accept if:
  - ΔA = A_post − A_pre ≥ 0.15 absolute
  - A_post ≥ 0.45

2) Learning dynamics corroboration
- Spearman ρ across rounds:
  - ρ(avg_conf, round) ≥ 0.4
  - ρ(avg_mass, round) ≥ 0.3
  - churn_pairs shows ρ ≥ 0.2 (associations evolving)
  - territories is non‑decreasing with at least one split; ideally observe ≥ 1 split and ≥ 1 merge

3) Controls falsify spurious effects
- Mismatch queries:
  - ΔA_mismatch ≤ 0.05 and |ρ(metric, round)| ≤ 0.2
- Reinforcement‑OFF:
  - ΔA_off ≤ 0.05 and |ρ(metric, round)| ≤ 0.2


## Tuning and options

- Rounds and Top‑k:
  --rounds N (default 12)
  --topk K (default 5)
- Small LLM for A/B:
  --small-model gemma3:4b (default); choose any local small LLM in Ollama
  --k-context (default 5) — how many retrieved chunks to include in A/B context
- Books mode:
  --books PATH (directory, PDF, or mixed)
  --limit-pages N — cap pages per PDF during extraction (speed)


## Troubleshooting

- Qdrant not reachable
  - Ensure container is running and port 6333 is open:
    docker ps
    curl http://127.0.0.1:6333/ready

- Ollama embeddings/LLM not available
  - Pull the models:
    ollama pull mxbai-embed-large
    ollama pull gemma3:4b

- Dimension mismatch errors
  - If you previously created a collection with a different vector size, drop or recreate it in Qdrant, or start with a fresh DB directory for this demo.
  - The embeddings model (e.g., mxbai-embed-large) defines the vector dimension; the Qdrant collection must match that size.

- No visible splits/engrams
  - Increase --rounds (e.g., 20–30).
  - Ensure synthetic mode is used for strongest signals initially.
  - Consider that engrams may require sufficient reinforcement cycles to emerge.

- Flat curves in main (non‑control) run
  - Verify your queries are the synthetic ones (the demo’s default retrieval list).
  - Check that m.retrieve(...) is being called (it is in the demo) and that learning is enabled (enable_void_memory=True in the Mnemosyne instance).
  - Confirm Qdrant/Ollama are healthy (no silent failures in logs).


## Guardrails and design constraints

The demo respects the following:
- Sparse‑only: Learning is event‑driven via retrieval calls; no dense scans are introduced.
- No schedulers: Cycles are driven by explicit retrieval rounds only.
- No scans in core/ or maps/: The demo operates in tools/ using public APIs.
- Maps/frame v1/v2 contracts: Not modified by the demo.
- Control impact: Reporting utilities do not affect “golden run” behavior (effects limited to retrieval and logging).


## Notes

- The synthetic corpus is intentionally small to demonstrate learning dynamics quickly.
- The A/B QA uses keyword‑recall as a simple, fast proxy for answer content. It is sufficient to show reliable deltas under this setup; you can integrate richer QA scoring if desired.
- For fresh evaluations, you may want to clear prior storage (DB path) if you have residual state from earlier experiments.


## Common commands

- Full synthetic with QA and both controls:
  python tools/void_learning_report.py --mode synthetic --rounds 12 --topk 5 --plots-out output/void_report --do-qa --do-control-mismatch --do-control-off

- Quick synthetic (QA only):
  python tools/void_learning_report.py --mode synthetic --rounds 12 --topk 5 --plots-out output/void_report --do-qa

- Books mode:
  python tools/void_learning_report.py --mode books --books ./books --rounds 8 --topk 5 --plots-out output/void_report


## Contact

Authored for demonstration by Justin K. Lietz.
