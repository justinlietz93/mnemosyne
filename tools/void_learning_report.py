
"""
Void Learning Report: logs metrics, draws plots, and runs small-LLM A/B on synthetic QA.

Features
- Ingest synthetic cross-domain corpus (graphs/neural/physics) or PDFs from ./books
- Drive reinforcement via retrieval; log time-series learning metrics (JSON+CSV)
- Generate PNG plots (avg_conf, avg_mass, avg_heat, churn_pairs, engrams, territories)
- Optional small-LLM A/B (before vs after reinforcement) on synthetic QA to estimate lift

Requirements
- matplotlib, pandas (for plotting)
- pypdf (only if --mode books)
- Qdrant running (default http://127.0.0.1:6333)
- Ollama running with embedding model 'mxbai-embed-large' and an LLM (e.g., 'gemma3:4b')

Usage
  python tools/void_learning_report.py --mode synthetic --rounds 12 --topk 5 --plots-out output/void_report --do-qa
  python tools/void_learning_report.py --mode books --books ./books --rounds 8 --topk 5 --plots-out output/void_report

Notes
- Uses the pure-embeddings adapter (Qdrant payloads off). Texts are resolved from the LocalDocStore.
- A/B QA is synthetic-only (known ground truth). Books mode omits QA unless a task file is provided.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import csv
from typing import List, Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

try:
    from pypdf import PdfReader  # only when --mode books
except Exception:
    PdfReader = None  # type: ignore

import ollama  # for small-LLM A/B

# Import core memory (Qdrant pure-embeddings + LocalDocStore wiring)
# Ensure project root on sys.path when executed directly as a script from tools/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from mnemosyne_core import Mnemosyne, EMBEDDING_MODEL, DB_PATH, COLLECTION_NAME


# ---------------- Synthetic corpus and QA ----------------

def cluster_texts() -> List[Tuple[str, str]]:
    G = [
        ("g_0", "Graphs have nodes and edges. Depth-first search explores along edges."),
        ("g_1", "Breadth-first search visits neighbors layer by layer across the graph."),
        ("g_2", "Message passing on graphs aggregates neighbor features to update node states."),
        ("g_3", "Graph connectivity and shortest paths relate to traversal and diffusion over edges."),
        ("g_4", "Graph embeddings capture structure; neighborhoods induce similarity."),
    ]
    N = [
        ("n_0", "Neural networks contain layers of neurons with weights and activations."),
        ("n_1", "Backpropagation propagates gradients across layers to update weights."),
        ("n_2", "Residual connections pass information forward, improving gradient flow."),
        ("n_3", "Graph neural networks generalize message passing with learned filters."),
        ("n_4", "Attention aggregates context via weighted neighbors, similar to message passing."),
    ]
    P = [
        ("p_0", "Diffusion spreads heat over a domain; the heat equation involves the Laplacian."),
        ("p_1", "Gradients flow from high concentration to low; smoothing occurs over time."),
        ("p_2", "The Laplace operator on a mesh approximates graph Laplacians on discrete nodes."),
        ("p_3", "Green's functions describe propagation; kernels spread signals across space."),
        ("p_4", "Conservation and symmetry lead to useful invariants under diffusion."),
    ]
    BR = [
        ("b_0", "Message passing on graphs is analogous to diffusion where neighbor states mix."),
        ("b_1", "Backpropagation resembles information flow across a computational graph."),
        ("b_2", "Graph Laplacians approximate continuous Laplacians connecting graphs and physics."),
        ("b_3", "Attention weights act like adaptive diffusion coefficients focusing relevant neighbors."),
        ("b_4", "Neural message passing can implement physics-informed diffusion on meshes."),
    ]
    return G + N + P + BR


def synthetic_qa() -> List[Dict[str, Any]]:
    """
    Each QA item has:
      - q: question
      - ref_keywords: set of keywords expected in a correct answer (orderless)
    """
    return [
        {"q": "What is message passing on graphs?", "ref_keywords": {"message", "passing", "graphs", "neighbors", "update"}},
        {"q": "How are graph Laplacians related to diffusion?", "ref_keywords": {"laplacian", "graph", "diffusion", "approximate"}},
        {"q": "What does backpropagation do?", "ref_keywords": {"backpropagation", "gradients", "layers", "weights", "update"}},
        {"q": "How can attention be seen as diffusion?", "ref_keywords": {"attention", "weights", "diffusion", "neighbors"}},
        {"q": "Describe physics-informed message passing.", "ref_keywords": {"physics", "message", "passing", "meshes", "diffusion"}},
    ]


# ---------------- Helpers ----------------

def mkdir_p(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def books_iter_texts(path: str, limit_pages: Optional[int] = None) -> List[Tuple[str, str]]:
    if PdfReader is None:
        print("[Warn] pypdf missing; cannot use --mode books.")
        return []
    out: List[Tuple[str, str]] = []
    if not os.path.exists(path):
        return out
    if os.path.isfile(path) and path.lower().endswith(".pdf"):
        targets = [path]
    else:
        targets = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".pdf"):
                    targets.append(os.path.join(root, f))
    targets = sorted(targets)
    for i, pdf in enumerate(targets):
        try:
            rd = PdfReader(pdf)
            texts: List[str] = []
            for pi, p in enumerate(rd.pages):
                if limit_pages is not None and pi >= limit_pages:
                    break
                try:
                    t = p.extract_text() or ""
                except Exception:
                    t = ""
                if t.strip():
                    texts.append(t)
            if texts:
                sid = f"pdf_{i}"
                out.append((sid, "\n".join(texts)))
        except Exception as e:
            print(f"[books] Skip {pdf}: {e}")
    return out


def configure_void_for_fast_learning(m: Mnemosyne, fast: bool = True) -> None:
    vm = getattr(m, "_void_memory", None)
    if vm is None:
        return
    if fast:
        try:
            vm.frontier_patience = 4     # default 20
            vm.habituation_start = 4     # default 12
            vm.habituation_scale = 10.0  # faster boredom rise
            vm.condensation_boredom = 0.5
            vm.condensation_conf = 0.4
            vm.condensation_mass = 5.0
            vm.diffusion_interval = 10
        except Exception:
            pass


def ingest(m: Mnemosyne, mode: str, books_path: str, chunk_size: int, overlap: int, limit_pages: Optional[int] = None) -> int:
    n = 0
    if mode == "synthetic":
        for sid, text in cluster_texts():
            try:
                m.inject(text, sid, chunk_size=chunk_size, overlap=overlap, metadata={"source": "report_synth"})
                n += 1
            except Exception as e:
                print(f"[Ingest][{sid}] {e}")
    elif mode == "books":
        pairs = books_iter_texts(books_path, limit_pages=limit_pages)
        for sid, text in pairs:
            try:
                m.inject(text, sid, chunk_size=chunk_size, overlap=overlap, metadata={"source": "report_books"})
                n += 1
            except Exception as e:
                print(f"[Ingest][{sid}] {e}")
    else:
        print(f"[Ingest] Unknown mode: {mode}")
    return n


def retrieval_queries(mode: str) -> List[str]:
    if mode == "synthetic":
        return [
            "message passing on graphs",
            "graph laplacian diffusion",
            "backpropagation across layers",
            "attention as adaptive diffusion",
            "physics informed neural message passing",
            "shortest paths and diffusion relation",
            "gradients over graph embeddings",
        ]
    else:
        # Books mode: use generic queries to touch diverse content
        return [
            "software architecture patterns",
            "testing strategies and tdd",
            "refactoring techniques",
            "data intensive applications concepts",
            "delivery pipelines and devops",
        ]


def drive_reinforcement(m: Mnemosyne, rounds: int, topk: int, mode: str, log_series: List[Dict[str, Any]]) -> None:
    qs = retrieval_queries(mode)
    for r in range(rounds):
        for q in qs:
            try:
                # Use high-level API that triggers reinforcement dynamics
                m.retrieve(q, n_results=topk)
            except Exception as e:
                print(f"[Retrieve]['{q}'] {e}")
        # Record stats after each round
        st = m.void_stats() or {}
        st["round"] = int(r + 1)
        log_series.append(st)


def save_metrics_json_csv(series: List[Dict[str, Any]], json_out: str, csv_out: str) -> None:
    mkdir_p(os.path.dirname(json_out))
    try:
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump(series, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Save] JSON failed: {e}")
    try:
        keys = sorted({k for row in series for k in row.keys()})
        with open(csv_out, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            for row in series:
                wr.writerow(row)
    except Exception as e:
        print(f"[Save] CSV failed: {e}")


def draw_plots(series: List[Dict[str, Any]], out_dir: str) -> None:
    mkdir_p(out_dir)
    if not series:
        print("[Plot] No data.")
        return
    df = pd.DataFrame(series)
    # Select common metrics if present
    metrics = ["avg_conf", "avg_mass", "avg_heat", "churn_pairs", "engrams", "territories"]
    x = df["round"] if "round" in df.columns else pd.Series(range(1, len(df) + 1))
    for mname in metrics:
        if mname in df.columns:
            try:
                plt.figure(figsize=(7.5, 4.5))
                plt.plot(x, df[mname], marker="o")
                plt.xlabel("round")
                plt.ylabel(mname)
                plt.title(mname)
                plt.grid(True, alpha=0.3)
                out_path = os.path.join(out_dir, f"{mname}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=160)
                plt.close()
            except Exception as e:
                print(f"[Plot] {mname}: {e}")


# ---------------- Small-LLM A/B (synthetic only) ----------------

def build_context_from_results(results: Dict[str, Any], k: int) -> str:
    """
    Build a simple textual context from top-k documents in a Chroma-shaped result.
    """
    try:
        docs = list(results.get("documents", [[]])[0] or [])
        top = [d for d in docs[:max(1, int(k))] if isinstance(d, str) and d.strip()]
        return "\n\n---\n\n".join(top)
    except Exception:
        return ""


def _embed_query(m: Mnemosyne, q: str) -> List[float]:
    """
    Generate a query embedding using the backend's embedding model. Robust to failures.
    """
    try:
        emb = m._generate_query_embedding(q)  # type: ignore[attr-defined]
        return emb or []
    except Exception:
        return []


def retrieval_no_learn(m: Mnemosyne, q: str, topk: int = 5) -> Dict[str, Any]:
    """
    Retrieve without triggering reinforcement.
    This bypasses collection.query (which reinforces) and queries the raw Qdrant index,
    then maps Qdrant IDs -> original IDs and texts from the LocalDocStore.
    """
    try:
        emb = _embed_query(m, q)
    except Exception:
        emb = []
    if not emb:
        return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
    try:
        raw = getattr(m, "_index", None)
        if raw is None:
            # Fallback to collection.query if index missing (may reinforce)
            res = m.collection.query(query_embeddings=[emb], n_results=int(max(1, topk)), where=None)
            return {
                "ids": res.get("ids", [[]]),
                "distances": res.get("distances", [[]]),
                "documents": res.get("documents", [[]]),
                "metadatas": res.get("metadatas", [[]]),
            }
        out = raw.search(query_embedding=emb, top_k=int(max(1, topk)), filter_payload=None, with_payload=False)
    except Exception:
        return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
    try:
        qids = out.get("ids", [[]])[0] if out else []
        dists = out.get("distances", [[]])[0] if out else []
    except Exception:
        qids, dists = [], []
    # Map qids -> orig_ids and texts from LocalDocStore
    try:
        ds = getattr(m, "_doc_store", None)
        if ds is not None:
            try:
                orig_ids = ds.get_orig_ids_by_qids(qids)
            except Exception:
                orig_ids = [None] * len(qids)
            try:
                texts = ds.get_texts_by_qids(qids)
            except Exception:
                texts = [None] * len(qids)
        else:
            orig_ids = [None] * len(qids)
            texts = [None] * len(qids)
    except Exception:
        orig_ids, texts = [None] * len(qids), [None] * len(qids)
    ids = [[(orig_ids[i] if orig_ids[i] is not None else qids[i]) for i in range(len(qids))]]
    docs = [[(texts[i] if i < len(texts) else None) for i in range(len(qids))]]
    metas = [[{} for _ in range(len(qids))]]
    return {"ids": ids, "distances": [dists], "documents": docs, "metadatas": metas}


def control_mismatch_queries() -> List[str]:
    """
    Mismatched queries intended not to align with the synthetic corpus.
    """
    return [
        "cooking recipes for sourdough starter",
        "gardening tips for tomato plants",
        "music theory chords and progressions",
        "travel itineraries in Europe",
        "painting techniques for oils",
    ]


def drive_reinforcement_with_queries(
    m: Mnemosyne,
    rounds: int,
    topk: int,
    queries: List[str],
    log_series: List[Dict[str, Any]],
    learn: bool = True,
) -> None:
    """
    Drive retrieval using a provided query list.
    If learn=False, bypass reinforcement (no retrieve calls).
    """
    for r in range(int(max(1, rounds))):
        for q in queries:
            try:
                if learn:
                    m.retrieve(q, n_results=int(max(1, topk)))
                else:
                    _ = retrieval_no_learn(m, q, topk=int(max(1, topk)))
            except Exception as e:
                print(f"[Retrieve]['{q}'] {e}")
        st = m.void_stats() or {}
        st["round"] = int(r + 1)
        log_series.append(st)


def ask_small_llm(model: str, question: str, context: str, temperature: float = 0.1) -> str:
    """
    Query a small LLM via ollama with provided context.
    """
    sys_msg = "You are a concise assistant. Answer strictly using the provided CONTEXT. If unknown, say 'I don't know'."
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            options={"temperature": temperature},
        )
        return str(resp.get("message", {}).get("content", "")).strip()
    except Exception as e:
        print(f"[LLM] ollama.chat failed: {e}")
        return ""


def keyword_accuracy(ans: str, ref_keywords: set[str]) -> float:
    """
    Simple keyword recall proxy: fraction of reference keywords found in the answer.
    """
    try:
        a = ans.lower()
        found = sum(1 for kw in ref_keywords if kw.lower() in a)
        return float(found) / float(max(1, len(ref_keywords)))
    except Exception:
        return 0.0


def evaluate_small_llm(
    m: Mnemosyne,
    qa_items: List[Dict[str, Any]],
    model: str,
    k: int,
    retrieval_fn,
    label: str,
) -> Dict[str, Any]:
    """
    Run QA using a retrieval_fn (pre/post) and compute per-item keyword accuracy and mean.
    """
    results = []
    for it in qa_items:
        q = str(it.get("q", ""))
        ref = set(list(it.get("ref_keywords", []) or []))
        try:
            res = retrieval_fn(m, q, topk=int(max(1, k)))
        except Exception:
            res = {"documents": [[]], "ids": [[]]}
        ctx = build_context_from_results(res, k=int(max(1, k)))
        ans = ask_small_llm(model=model, question=q, context=ctx)
        acc = keyword_accuracy(ans, ref)
        results.append({"q": q, "acc": acc, "answer": ans})
    mean_acc = float(sum(x["acc"] for x in results) / max(1, len(results)))
    return {"label": str(label), "mean_acc": mean_acc, "items": results}


def draw_ab_bar(results_pre: Dict[str, Any], results_post: Dict[str, Any], out_path: str) -> None:
    """
    Draw a simple bar chart comparing per-item accuracy pre vs post.
    """
    try:
        mkdir_p(os.path.dirname(out_path))
        pre = results_pre.get("items", [])
        post = results_post.get("items", [])
        n = max(len(pre), len(post))
        xs = list(range(1, n + 1))
        pre_acc = [pre[i]["acc"] if i < len(pre) else 0.0 for i in range(n)]
        post_acc = [post[i]["acc"] if i < len(post) else 0.0 for i in range(n)]
        plt.figure(figsize=(8.0, 4.5))
        plt.bar([x - 0.2 for x in xs], pre_acc, width=0.4, label="pre")
        plt.bar([x + 0.2 for x in xs], post_acc, width=0.4, label="post")
        plt.xlabel("QA item")
        plt.ylabel("keyword accuracy (0..1)")
        plt.title("Small-LLM QA accuracy: pre vs post reinforcement")
        plt.xticks(xs)
        plt.ylim(0.0, 1.0)
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except Exception as e:
        print(f"[Plot][A/B] {e}")


def compute_spearman(series: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Optional[float]]:
    """
    Compute Spearman correlation between each metric and round, coercing types and
    returning None when correlation is undefined (e.g., constant series).
    """
    try:
        if not series:
            return {m: None for m in metrics}
        df = pd.DataFrame(series)
        if "round" not in df.columns:
            df["round"] = list(range(1, len(df) + 1))
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        out: Dict[str, Optional[float]] = {}
        for m in metrics:
            try:
                if m not in df.columns:
                    out[m] = None
                    continue
                s = pd.to_numeric(df[m], errors="coerce")
                rho = df["round"].corr(s, method="spearman")
                out[m] = None if pd.isna(rho) else float(rho)
            except Exception:
                out[m] = None
        return out
    except Exception:
        return {m: None for m in metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description="Void learning report: logs metrics, plots, and optional small-LLM A/B.")
    ap.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "books"], help="Corpus mode")
    ap.add_argument("--books", type=str, default="./books", help="Path to books directory or PDF file")
    ap.add_argument("--limit-pages", type=int, default=None, help="Optional per-PDF page limit")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--plots-out", type=str, default="output/void_report")
    ap.add_argument("--qdrant-url", type=str, default="http://127.0.0.1:6333")
    ap.add_argument("--do-qa", action="store_true", help="Run small-LLM A/B on synthetic QA")
    ap.add_argument("--small-model", type=str, default="gemma3:4b", help="Small LLM model for A/B")
    ap.add_argument("--k-context", type=int, default=5)
    ap.add_argument("--run-tag", type=str, default=None, help="Optional suffix to isolate collection/db for this run")
    ap.add_argument("--fresh", action="store_true", help="Use a timestamped run tag for a fresh collection/db")
    ap.add_argument("--do-control-mismatch", action="store_true", help="Run mismatch-queries control")
    ap.add_argument("--do-control-off", action="store_true", help="Run reinforcement-off control (flat curves)")
    args = ap.parse_args()

    # Derive isolated storage for this run if requested
    run_tag = args.run_tag if getattr(args, "run_tag", None) else (time.strftime("%Y%m%d-%H%M%S") if getattr(args, "fresh", False) else None)
    db_path = DB_PATH if not run_tag else f"{DB_PATH}_{run_tag}"
    collection_name = COLLECTION_NAME if not run_tag else f"{COLLECTION_NAME}_{run_tag}"

    # Initialize memory (pure embeddings Qdrant + LocalDocStore)
    print("[Report] Initializing Mnemosyne ...")
    print(f"[Report] Using db_path={db_path} collection={collection_name}")
    try:
        m = Mnemosyne(
            db_path=db_path,
            collection_name=collection_name,
            model=EMBEDDING_MODEL,
            backend="qdrant",
            qdrant_url=args.qdrant_url,
            enable_void_memory=True,
        )
    except Exception as e:
        print(f"[Fatal] Mnemosyne init failed: {e}")
        print("Ensure Qdrant and Ollama are running and the embedding model is available.")
        return 2

    configure_void_for_fast_learning(m, fast=True)

    # Ingest
    print(f"[Report] Ingesting mode={args.mode} ...")
    n_docs = ingest(m, args.mode, args.books, args.chunk_size, args.overlap, limit_pages=args.limit_pages)
    print(f"[Report] Ingested {n_docs} documents.")

    out_dir = str(args.plots_out)
    mkdir_p(out_dir)

    # Pre QA (only synthetic QA makes sense for ground-truth keywords)
    qa_items = synthetic_qa() if args.mode == "synthetic" else []
    pre_qa: Dict[str, Any] = {}
    if args.do_qa and qa_items:
        print("[Report] Running small-LLM QA (pre) ...")
        pre_qa = evaluate_small_llm(m, qa_items, model=args.small_model, k=int(max(1, args.k_context)), retrieval_fn=retrieval_no_learn, label="pre")
        try:
            with open(os.path.join(out_dir, "qa_pre.json"), "w", encoding="utf-8") as fh:
                json.dump(pre_qa, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Learning run (reinforcement ON)
    print("[Report] Driving reinforcement ...")
    series: List[Dict[str, Any]] = []
    drive_reinforcement(m, rounds=int(max(1, args.rounds)), topk=int(max(1, args.topk)), mode=args.mode, log_series=series)

    # Save metrics and plots
    print("[Report] Saving metrics and plots ...")
    save_metrics_json_csv(series, os.path.join(out_dir, "metrics.json"), os.path.join(out_dir, "metrics.csv"))
    draw_plots(series, out_dir)
    spearman = compute_spearman(series, ["avg_conf", "avg_mass", "avg_heat", "churn_pairs", "engrams", "territories"])
    try:
        with open(os.path.join(out_dir, "spearman.json"), "w", encoding="utf-8") as fh:
            json.dump(spearman, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Post QA (evaluation retrieval without additional learning)
    post_qa: Dict[str, Any] = {}
    if args.do_qa and qa_items:
        print("[Report] Running small-LLM QA (post) ...")
        post_qa = evaluate_small_llm(m, qa_items, model=args.small_model, k=int(max(1, args.k_context)), retrieval_fn=retrieval_no_learn, label="post")
        try:
            with open(os.path.join(out_dir, "qa_post.json"), "w", encoding="utf-8") as fh:
                json.dump(post_qa, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # A/B bar chart
        draw_ab_bar(pre_qa, post_qa, os.path.join(out_dir, "ab_accuracy.png"))
        try:
            delta = float(post_qa.get("mean_acc", 0.0)) - float(pre_qa.get("mean_acc", 0.0))
            with open(os.path.join(out_dir, "qa_ab_summary.json"), "w", encoding="utf-8") as fh:
                json.dump({"pre": pre_qa.get("mean_acc", 0.0), "post": post_qa.get("mean_acc", 0.0), "delta": delta}, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Control: mismatch queries (learn ON but irrelevant prompts)
    if args.do_control_mismatch:
        print("[Report][Control] Mismatch queries (learn ON) ...")
        series_ctrl: List[Dict[str, Any]] = []
        drive_reinforcement_with_queries(m, rounds=int(max(1, args.rounds)), topk=int(max(1, args.topk)), queries=control_mismatch_queries(), log_series=series_ctrl, learn=True)
        ctrl_dir = os.path.join(out_dir, "control_mismatch")
        mkdir_p(ctrl_dir)
        save_metrics_json_csv(series_ctrl, os.path.join(ctrl_dir, "metrics.json"), os.path.join(ctrl_dir, "metrics.csv"))
        draw_plots(series_ctrl, ctrl_dir)
        spearman_ctrl = compute_spearman(series_ctrl, ["avg_conf", "avg_mass", "avg_heat", "churn_pairs", "engrams", "territories"])
        try:
            with open(os.path.join(ctrl_dir, "spearman.json"), "w", encoding="utf-8") as fh:
                json.dump(spearman_ctrl, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Control: reinforcement OFF (flat expected)
    if args.do_control_off:
        print("[Report][Control] Reinforcement OFF ...")
        series_off: List[Dict[str, Any]] = []
        drive_reinforcement_with_queries(m, rounds=int(max(1, args.rounds)), topk=int(max(1, args.topk)), queries=retrieval_queries(args.mode), log_series=series_off, learn=False)
        off_dir = os.path.join(out_dir, "control_off")
        mkdir_p(off_dir)
        save_metrics_json_csv(series_off, os.path.join(off_dir, "metrics.json"), os.path.join(off_dir, "metrics.csv"))
        draw_plots(series_off, off_dir)
        spearman_off = compute_spearman(series_off, ["avg_conf", "avg_mass", "avg_heat", "churn_pairs", "engrams", "territories"])
        try:
            with open(os.path.join(off_dir, "spearman.json"), "w", encoding="utf-8") as fh:
                json.dump(spearman_off, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    print("[Report] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
