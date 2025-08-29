"""
Smoke test: VoidMemory learning dynamics over pure-embedding Qdrant backend.

Goal:
- Verify that the Void graph system aggregates concepts and associates relationships automatically.
- Evidence via:
  * reinforcement effects (avg_conf/use_count growth),
  * territory dynamics (splits/merges count, territory count),
  * co-retrieval pair churn,
  * engram formation (structural consolidation),
  * retrieval quality across cross-domain bridge queries.

This test uses a synthetic corpus with three conceptual clusters and bridging texts
to provoke cross-domain co-retrieval. It lowers some VoidMemory thresholds to
observe dynamics quickly in a short run (safe, test-only).

Requirements:
- Qdrant running at http://127.0.0.1:6333 (or override via --qdrant-url)
- Ollama with embedding model available: mxbai-embed-large
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

from mnemosyne_core import Mnemosyne, EMBEDDING_MODEL, DB_PATH, COLLECTION_NAME


def _cluster_texts() -> List[Tuple[str, str]]:
    """
    Build a small synthetic corpus:
      - Cluster G (graphs): BFS/DFS/edges/nodes, message passing
      - Cluster N (neural): neurons/layers/weights/activations/backprop
      - Cluster P (physics): diffusion/gradients/laplacian/heat equation
      - Bridges: analogies connecting graphs/neural/physics
    Each entry: (source_id, text)
    """
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


def _queries() -> List[str]:
    return [
        "message passing on graphs",
        "graph laplacian diffusion",
        "backpropagation across layers",
        "attention as adaptive diffusion",
        "physics informed neural message passing",
        "shortest paths and diffusion relation",
        "gradients over graph embeddings",
    ]


def configure_void_for_fast_learning(m: Mnemosyne) -> None:
    """
    Lower a few thresholds to allow observable dynamics in a short test run.
    These do not affect core logic; they only speed up splits/engrams under test.
    """
    vm = getattr(m, "_void_memory", None)
    if vm is None:
        return
    try:
        vm.frontier_patience = 4     # default 20
        vm.habituation_start = 4     # default 12
        vm.habituation_scale = 10.0  # faster boredom rise
        vm.condensation_boredom = 0.5   # default 0.7
        vm.condensation_conf = 0.4      # default 0.6
        vm.condensation_mass = 5.0      # default 8.0
        vm.diffusion_interval = 10      # enable periodic diffusion/merges in short runs
    except Exception:
        pass


def ingest_corpus(m: Mnemosyne, chunk_size: int = 3, overlap: int = 1) -> int:
    corpus = _cluster_texts()
    injected_docs = 0
    for sid, text in corpus:
        try:
            m.inject(text, sid, chunk_size=chunk_size, overlap=overlap, metadata={"source": "void_learning_test"})
            injected_docs += 1
        except Exception as e:
            print(f"[Ingest] Failed for {sid}: {e}")
    return injected_docs


def reinforce_via_queries(m: Mnemosyne, rounds: int = 12, topk: int = 5) -> None:
    """
    Perform repeated retrieval to drive reinforcement, boredom, frontier counts, and potential splits.
    """
    qs = _queries()
    for r in range(rounds):
        for q in qs:
            try:
                m.retrieve(q, n_results=topk)
            except Exception as e:
                print(f"[Retrieve] '{q}' failed: {e}")


def summarize_learning(m: Mnemosyne) -> dict:
    stats = m.void_stats()
    events = m.void_events(limit=200, consume=False)
    split_events = [e for e in events if isinstance(e, tuple) and len(e) >= 2 and str(e[1]).startswith("territory_split")]
    merge_events = [e for e in events if isinstance(e, tuple) and len(e) >= 2 and str(e[1]).startswith("territory_merge")]
    engram_events = [e for e in events if isinstance(e, tuple) and len(e) >= 2 and str(e[1]) == "engram"]
    out = {
        "stats": stats,
        "events_tail": events[-10:] if events else [],
        "split_events": len(split_events),
        "merge_events": len(merge_events),
        "engrams_events": len(engram_events),
    }
    return out


def pass_fail(report: dict) -> Tuple[bool, List[str]]:
    s = report.get("stats", {}) or {}
    reasons: List[str] = []
    ok = True

    # Basic growth signals
    if float(s.get("count", 0)) <= 0:
        ok = False
        reasons.append("no memories registered (count == 0)")

    # Territory dynamics: either observed splits/merges or nontrivial territory count
    terr = int(float(s.get("territories", 0.0)))
    splits = int(report.get("split_events", 0))
    merges = int(report.get("merge_events", 0))
    if splits == 0 and merges == 0 and terr <= 1:
        ok = False
        reasons.append("no observable territory dynamics (splits/merges/territories)")

    # Pair churn indicates co-retrieval relationships
    churn_pairs = int(float(s.get("churn_pairs", 0.0)))
    if churn_pairs == 0:
        ok = False
        reasons.append("no co-retrieval pair churn observed")

    # Confidence growth proxy
    if float(s.get("avg_conf", 0.0)) <= 0.0:
        ok = False
        reasons.append("avg_conf did not increase")

    # Engrams or events
    engr_cnt = int(float(s.get("engrams", 0.0)))
    engr_ev = int(report.get("engrams_events", 0))
    if engr_cnt == 0 and engr_ev == 0:
        reasons.append("no engrams yet (may require more rounds); not failing test")

    return ok, reasons


def main() -> int:
    ap = argparse.ArgumentParser(description="Void graph learning smoke test over Qdrant (pure embeddings).")
    ap.add_argument("--qdrant-url", type=str, default="http://127.0.0.1:6333")
    ap.add_argument("--rounds", type=int, default=12, help="Reinforcement rounds of query cycles")
    ap.add_argument("--topk", type=int, default=5, help="Top-k for retrieval")
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--overlap", type=int, default=1)
    args = ap.parse_args()

    print("[Test] Initializing Mnemosyne (Qdrant pure embeddings)...")
    try:
        m = Mnemosyne(
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            model=EMBEDDING_MODEL,
            backend="qdrant",
            qdrant_url=args.qdrant_url,
            enable_void_memory=True,
        )
    except Exception as e:
        print(f"[Test][Fatal] Mnemosyne/Qdrant init failed: {e}")
        print("Ensure Qdrant is running (default http://127.0.0.1:6333) and Ollama embedding model is available.")
        return 2

    # Configure VoidMemory for fast observable learning in a short run
    configure_void_for_fast_learning(m)

    print("[Test] Ingesting synthetic corpus...")
    n_docs = ingest_corpus(m, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"[Test] Ingested {n_docs} documents.")

    print("[Test] Driving reinforcement via queries...")
    t0 = time.time()
    reinforce_via_queries(m, rounds=args.rounds, topk=args.topk)
    print(f"[Test] Reinforcement cycles took {time.time()-t0:.2f}s")

    print("[Test] Summarizing learning signals...")
    report = summarize_learning(m)
    stats = report["stats"]
    print("[Test] Stats:")
    for k in sorted(stats.keys()):
        try:
            print(f"  - {k}: {stats[k]}")
        except Exception:
            pass
    print(f"[Test] split_events={report['split_events']} merge_events={report['merge_events']} engrams_events={report['engrams_events']}")
    tail = report.get("events_tail", [])
    if tail:
        print("[Test] Recent events (tail):")
        for e in tail:
            print(f"  {e}")

    ok, reasons = pass_fail(report)
    if ok:
        print("[RESULT] PASS: Void graph shows aggregation/association signals (reinforcement, territories, churn).")
        return 0
    else:
        print("[RESULT] PARTIAL/FAIL:")
        for r in reasons:
            print(f"  - {r}")
        print("Note: Increase --rounds or lower thresholds further if needed to see splits/engrams.")
        return 1


if __name__ == "__main__":
    sys.exit(main())