"""Void Dynamics inspired emergent memory layer for Mnemosyne.

This module adds an organic lifecycle to vector memories by tracking per-chunk
state (ttl, confidence, mass, heat, novelty) and performing lightweight
homeostasis (decay + pruning + reinforcement) on every update.

Goals (Phase 1):
  - O(1) per registered/reinforced chunk (amortized) using sampling for decay.
  - No external deps (pure stdlib) to keep integration simple.
  - Deterministic territory bucketing (hash) for future clustering extensions.

Future extension hooks are marked with TODO comments (territory splits, boundary churn, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Iterable, List, Tuple
import time
import math
import random
import heapq
import threading
from collections import deque
import json

@dataclass
class MemoryState:
    id: str
    created: float
    ttl: int
    confidence: float = 0.0
    mass: float = 1.0
    heat: float = 0.0
    last_touch_tick: int = 0  # tick index of last reinforcement or registration
    novelty: float = 0.5  # heuristic 0..1
    territory: int = 0
    use_count: int = 0
    boredom: float = 0.0  # habituation 0..1
    inhibition: float = 0.0  # redundancy / over-exposure penalty

    def reinforce(self,
                  sim: float,
                  tick: int,
                  ttl_boost: int,
                  heat_gain: float,
                  conf_gain: float,
                  habituation_start: int,
                  habituation_scale: float):
        self.last_touch_tick = tick
        self.use_count += 1
        self.heat += heat_gain

        # Habituation: after habituation_start uses, boredom rises toward 1 with an exponential approach.
        if self.use_count > habituation_start:
            over = self.use_count - habituation_start
            # 1 - exp(-over / scale)
            self.boredom = max(self.boredom, 1.0 - math.exp(-over / max(1.0, habituation_scale)))

        # Confidence saturates in [0,1]; boredom dampens incremental gain
        eff_conf_gain = conf_gain * (1.0 - 0.5 * self.boredom)
        self.confidence = min(1.0, self.confidence + eff_conf_gain * max(0.0, sim))

        # Mass grows sublinearly and is also dampened by boredom
        self.mass += (0.25 + 0.75 * max(0.0, sim)) * (1.0 - 0.4 * self.boredom)
        self.ttl = max(self.ttl, ttl_boost)

        # Novelty decays with repeated exposure, more aggressively under boredom
        self.novelty *= (1.0 - 0.10 * (1.0 + self.boredom))
        self.novelty = max(0.0, min(1.0, self.novelty))

    def composite_score(self, tick: int, recency_hl: int, boredom_weight: float) -> float:
        # Tick-based recency half-life: recency = 0.5 ^ (Δticks / recency_hl)
        dt = max(0, tick - self.last_touch_tick)
        recency = 0.5 ** (dt / max(1.0, recency_hl))
        strength = self.mass / (self.mass + 10.0)
        heat_term = self.heat / (self.heat + 5.0)
        # Boredom penalizes score directly and dampens novelty contribution
        effective_novelty = self.novelty * (1.0 - 0.6 * self.boredom)
        base = (
            0.30 * self.confidence
            + 0.25 * recency
            + 0.18 * strength
            + 0.12 * effective_novelty
            + 0.15 * heat_term
        )
        penalty = boredom_weight * self.boredom + 0.20 * (self.inhibition / (self.inhibition + 5.0))
        return max(0.0, base - penalty)


class VoidMemoryManager:
    """Lifecycle manager for Mnemosyne chunks.

    Parameters
    -----------
    capacity : int
        Soft cap; pruning triggers when size exceeds this.
    base_ttl : int
        Initial / reinforcement ttl horizon (ticks = manager updates).
    decay_half_life : int
        Ticks for heat exponential half-life.
    prune_sample : int
        Number of candidates sampled from bottom of score distribution when pruning.
    prune_target_ratio : float
        Fraction of over-capacity to drop during a prune pass.
    thread_safe : bool
        Enable a lock for multi-threaded usage.
    """
    def __init__(self,
                 capacity: int = 5000,
                 base_ttl: int = 240,
                 decay_half_life: int = 120,
                 prune_sample: int = 256,
                 prune_target_ratio: float = 0.50,
                 thread_safe: bool = False,
                 seed: int = 0,
                 recency_half_life_ticks: int = 50,
                 habituation_start: int = 12,
                 habituation_scale: float = 25.0,
                 boredom_weight: float = 0.20,
                 frontier_novelty_threshold: float = 0.55,
                 frontier_patience: int = 20,
                 condensation_boredom: float = 0.70,
                 condensation_conf: float = 0.60,
                 condensation_mass: float = 8.0,
                 diffusion_interval: int = 50,
                 diffusion_kappa: float = 0.05,
                 exploration_churn_window: int = 200):
        self.capacity = int(max(10, capacity))
        self.base_ttl = int(max(10, base_ttl))
        self.decay_half_life = int(max(1, decay_half_life))
        self.prune_sample = int(max(16, prune_sample))
        self.prune_target_ratio = float(min(1.0, max(0.05, prune_target_ratio)))
        self._rng = random.Random(int(seed))
        self._mem: Dict[str, MemoryState] = {}
        self._tick: int = 0
        self._lock = threading.Lock() if thread_safe else None
        self.recency_half_life_ticks = int(max(1, recency_half_life_ticks))
        self.habituation_start = int(max(0, habituation_start))
        self.habituation_scale = float(max(1.0, habituation_scale))
        self.boredom_weight = float(max(0.0, min(1.0, boredom_weight)))
        # Frontier & territory dynamics
        self.frontier_novelty_threshold = float(min(1.0, max(0.0, frontier_novelty_threshold)))
        self.frontier_patience = int(max(2, frontier_patience))
        self._territory_frontier: Dict[int, int] = {}
        self._next_territory_id: int = 10_000  # new splits start high to avoid collisions
        # Condensation scheduling
        self.condensation_boredom = float(condensation_boredom)
        self.condensation_conf = float(condensation_conf)
        self.condensation_mass = float(condensation_mass)
        self._pending_condense: List[str] = []
        self._condense_callback = None  # optional external function(summary_ids->summary)
        # Diffusion
        self.diffusion_interval = int(max(5, diffusion_interval))
        self.diffusion_kappa = float(max(0.0, min(1.0, diffusion_kappa)))
        # Intrinsic reward
        self._reward_ema = 0.0
        self._reward_alpha = 0.05
        self._last_retrieved_territories: List[int] = []
        # Co-retrieval boundary churn
        self._pair_churn: Dict[tuple, float] = {}
        self._pair_last_tick: Dict[tuple, int] = {}
        self.exploration_churn_window = int(max(10, exploration_churn_window))
        self._exploration_temp = 0.0
        # Internal structural consolidations (engrams) - non-semantic clusters
        self._engrams: List[List[str]] = []
        # Event ring buffer for lightweight introspection
        self._events = deque(maxlen=1024)  # (tick, type, payload)
        self._persistence_version = 1
        # Embedding-driven emergent territories
        # Centroids stored as plain lists (means) and counts for incremental updates
        self._territory_centroids: Dict[int, List[float]] = {}
        self._territory_counts: Dict[int, int] = {}
        self._territory_centroid_norms: Dict[int, float] = {}
        # Per-territory recent distances (member -> centroid) for radius estimation
        self._territory_member_dists: Dict[int, deque] = {}
        # Running sample of nearest-neighbor distances used to set adaptive tau
        self._nn_distances: deque = deque(maxlen=5000)
        self._territory_tau: float = 0.25
        self._territory_warmup: int = 1000
        self._territory_nn_sample_max: int = 5000
        # Merge/split telemetry
        self._split_counter: int = 0
        self._merge_counter: int = 0

    # ---------------- Internal helpers ----------------
    def _territory_of(self, text: str) -> int:
        # Deterministic stable hash using blake2b to avoid Python's randomized hash across runs.
        # Return a large integer derived from the hash so territories can emerge organically
        # and split/merge over time instead of being constrained to a fixed bucket count.
        try:
            import hashlib
            h = hashlib.blake2b(text.encode('utf-8'), digest_size=16).digest()
            # Convert bytes to a large int; territory ids are allowed to grow organically
            val = int.from_bytes(h, 'big')
            return val
        except Exception:
            # Fallback to builtin hash for very unlikely failure modes (positive int)
            return abs(hash(text))

    def _estimate_novelty(self, text: str) -> float:
        # Simple heuristic: normalized Shannon-esque char diversity ratio.
        if not text:
            return 0.0
        chars = set(text.lower())
        return min(1.0, len(chars) / 64.0)

    def _decay_pass(self, now: float):
        # Exponential decay of heat; TTL decrement & eviction for expired low-value items.
        drop: List[str] = []
        for ms in self._mem.values():
            ms.ttl -= 1
            if ms.heat > 0.0:
                # Heat half-life
                ms.heat *= 0.5 ** (1.0 / self.decay_half_life)
            if ms.ttl <= 0 and ms.confidence < 0.05 and ms.mass < 3.0:
                drop.append(ms.id)
        for did in drop:
            self._mem.pop(did, None)

    def _prune_if_needed(self, now: float):
        size = len(self._mem)
        if size <= self.capacity:
            return
        over = size - self.capacity
        # Adaptive prune intensity (Phase B): adjust based on avg boredom & inhibition
        avg_boredom = 0.0
        avg_inhib = 0.0
        if self._mem:
            for ms in self._mem.values():
                avg_boredom += ms.boredom
                avg_inhib += ms.inhibition
            n = len(self._mem)
            avg_boredom /= n
            avg_inhib /= n
        dynamic_ratio = self.prune_target_ratio * (1.0 + 0.5 * max(0.0, avg_boredom - 0.5) + 0.3 * (avg_inhib / (avg_inhib + 10.0)))
        dynamic_ratio = min(0.95, max(0.05, dynamic_ratio))
        target_drop = max(1, int(math.ceil(over * dynamic_ratio)))
        # Sample candidates biased to low heat & low confidence by random pre-filter
        ids = list(self._mem.keys())
        self._rng.shuffle(ids)
        candidates = ids[: min(self.prune_sample, len(ids))]
        scored: List[Tuple[float, str]] = []
        for cid in candidates:
            ms = self._mem.get(cid)
            if not ms:
                continue
            scored.append((ms.composite_score(self._tick, self.recency_half_life_ticks, self.boredom_weight), cid))
        # Take lowest target_drop
        worst = heapq.nsmallest(target_drop, scored, key=lambda kv: kv[0])
        for _, cid in worst:
            self._mem.pop(cid, None)

    # ---------------- Public API ----------------
    def register_chunks(self, ids: Iterable[str], raw_texts: Iterable[str], embeddings: Optional[Iterable[List[float]]] = None):
        """
        Register new chunks into memory. If embeddings are provided, use
        embedding-driven emergent territory assignment; otherwise fall back to
        text-hash based deterministic territory (legacy behavior).
        """
        now = time.time()
        if self._lock:
            with self._lock:
                self._register(ids, raw_texts, now, embeddings)
        else:
            self._register(ids, raw_texts, now, embeddings)

    def _register(self, ids: Iterable[str], raw_texts: Iterable[str], now: float, embeddings: Optional[Iterable[List[float]]] = None):
        # If embeddings present, use them for emergent territory assignment
        emb_iter = iter(embeddings) if embeddings is not None else None
        for cid, text in zip(ids, raw_texts):
            if cid in self._mem:
                continue
            novelty = self._estimate_novelty(text)
            # Choose territory id
            terr = None
            emb = None
            if emb_iter is not None:
                try:
                    emb = next(emb_iter)
                except StopIteration:
                    emb = None
            if emb is not None:
                try:
                    terr = self._assign_to_territory(emb, cid)
                except Exception:
                    terr = self._territory_of(text)
            else:
                terr = self._territory_of(text)
            self._mem[cid] = MemoryState(
                id=cid,
                created=now,
                ttl=self.base_ttl,
                novelty=novelty,
                territory=terr,
                last_touch_tick=self._tick
            )
            # If we assigned using embedding, record distance sample for tau estimator
            if emb is not None:
                try:
                    d = self._cosine_distance_to_centroid(emb, terr)
                    if d is not None:
                        self._nn_distances.append(d)
                except Exception:
                    pass
        # Post-registration lifecycle maintenance (must remain inside method scope)
        self._tick += 1
        self._decay_pass(now)
        self._prune_if_needed(now)
        self._maybe_diffuse(now)

    # ---------------- Embedding / Territory helpers ----------------
    def _normalize(self, v: List[float]) -> List[float]:
        # L2-normalize embedding for cosine calculations
        s = 0.0
        for x in v:
            s += x * x
        if s <= 0.0:
            return v
        inv = 1.0 / math.sqrt(s)
        return [x * inv for x in v]

    def _update_centroid_incremental(self, territory: int, emb: List[float]):
        n = self._territory_counts.get(territory, 0)
        if n == 0:
            self._territory_centroids[territory] = list(emb)
            self._territory_counts[territory] = 1
        else:
            c = self._territory_centroids[territory]
            # running mean: c = (c*n + emb) / (n+1)
            for i in range(len(c)):
                c[i] = (c[i] * n + emb[i]) / (n + 1)
            self._territory_counts[territory] = n + 1
        # cache norm
        c = self._territory_centroids[territory]
        self._territory_centroid_norms[territory] = math.sqrt(sum(x * x for x in c))

    def _cosine_distance(self, a: List[float], b: List[float]) -> float:
        # assumes non-empty lists with same length
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - (dot / (math.sqrt(na) * math.sqrt(nb)))

    def _cosine_distance_to_centroid(self, emb: List[float], territory: int) -> Optional[float]:
        c = self._territory_centroids.get(territory)
        if not c:
            return None
        return self._cosine_distance(emb, c)

    def _assign_to_territory(self, emb: List[float], cid_hint: str) -> int:
        """Assign an embedding to the nearest territory centroid if within adaptive tau,
        otherwise create a new territory id and initialize centroid.
        Returns territory id (int).
        """
        # normalize embedding
        embn = self._normalize(emb)
        # Warmup: if we have few samples, create new territory for each
        if len(self._nn_distances) < self._territory_warmup and len(self._territory_centroids) < 50:
            new_tid = self._next_territory_id
            self._next_territory_id += 1
            self._territory_centroids[new_tid] = list(embn)
            self._territory_counts[new_tid] = 1
            self._territory_member_dists[new_tid] = deque(maxlen=512)
            return new_tid

        # compute nearest territory
        best_tid = None
        best_d = 1.0
        for tid, centroid in self._territory_centroids.items():
            d = self._cosine_distance(embn, centroid)
            if d < best_d:
                best_d = d
                best_tid = tid

        # Update adaptive tau from NN samples median if available
        if self._nn_distances:
            sorted_sample = sorted(list(self._nn_distances))
            mid = len(sorted_sample) // 2
            tau = sorted_sample[mid]
            # conservative floor
            tau = max(0.05, min(0.6, tau))
            self._territory_tau = tau

        if best_tid is not None and best_d <= self._territory_tau:
            # assign
            self._update_centroid_incremental(best_tid, embn)
            self._territory_member_dists.setdefault(best_tid, deque(maxlen=1024)).append(best_d)
            return best_tid

        # else create new territory
        new_tid = self._next_territory_id
        self._next_territory_id += 1
        self._territory_centroids[new_tid] = list(embn)
        self._territory_counts[new_tid] = 1
        self._territory_member_dists[new_tid] = deque(maxlen=1024)
        return new_tid

    def reinforce(self, results: Dict[str, List]):
        """Reinforce chunks from a Chroma query() result object.

        Expects standard Chroma result dict with keys: 'ids', 'distances'.
        Uses similarity = 1 - distance.
        """
        now = time.time()
        ids_rows = results.get('ids') or []
        dist_rows = results.get('distances') or []
        if not ids_rows:
            return
        for row_ids, row_dists in zip(ids_rows, dist_rows):
            for cid, dist in zip(row_ids, row_dists):
                ms = self._mem.get(cid)
                if not ms:
                    continue
                sim = 1.0 - float(dist)
                ms.reinforce(
                    sim=sim,
                    tick=self._tick,
                    ttl_boost=self.base_ttl,
                    heat_gain=1.0,
                    conf_gain=0.05,
                    habituation_start=self.habituation_start,
                    habituation_scale=self.habituation_scale
                )
        self._tick += 1
        self._decay_pass(now)
        self._prune_if_needed(now)
        self._maybe_diffuse(now)

    def stats(self) -> Dict[str, float]:
        size = len(self._mem)
        if size == 0:
            return {"count": 0, "avg_conf": 0.0, "avg_mass": 0.0, "avg_heat": 0.0, "territories": 0, "avg_boredom": 0.0, "reward_ema": 0.0, "exploration_temp": 0.0, "churn_pairs": 0, "engrams": 0}
        acc_conf = acc_mass = acc_heat = 0.0
        terrs = set()
        acc_bored = 0.0
        acc_inhib = 0.0
        for ms in self._mem.values():
            acc_conf += ms.confidence
            acc_mass += ms.mass
            acc_heat += ms.heat
            terrs.add(ms.territory)
            acc_bored += ms.boredom
            acc_inhib += ms.inhibition
        avg_boredom = acc_bored / size
        return {
            "count": float(size),
            "avg_conf": acc_conf / size,
            "avg_mass": acc_mass / size,
            "avg_heat": acc_heat / size,
            "territories": float(len(terrs)),
            "avg_boredom": avg_boredom,
            "avg_inhibition": acc_inhib / size,
            "reward_ema": self._reward_ema,
            "exploration_temp": self._exploration_temp,
            "churn_pairs": float(len(self._pair_churn)),
            "tick": float(self._tick),
            "engrams": float(len(self._engrams)),
            "split_counter": float(getattr(self, "_split_counter", 0)),
            "merge_counter": float(getattr(self, "_merge_counter", 0)),
            "territory_tau": float(self._territory_tau),
            "nn_samples": float(len(self._nn_distances)),
        }

    def top(self, k: int = 10) -> List[Tuple[str, float]]:
        k = max(1, min(k, 100))
        scored = [
            (ms.composite_score(self._tick, self.recency_half_life_ticks, self.boredom_weight), ms.id)
            for ms in self._mem.values()
        ]
        topk = heapq.nlargest(k, scored, key=lambda kv: kv[0])
        return [(cid, score) for score, cid in topk]

    # ---------------- Phase A-D Enhancements ----------------
    def register_condensation_callback(self, fn):
        """Set an external callback: fn(list[str]) -> (summary_id, text) (optional)."""
        self._condense_callback = fn

    def consume_pending_condensations(self) -> List[str]:
        out = list(self._pending_condense)
        self._pending_condense.clear()
        return out

    def tick_post_retrieval(self, retrieved_ids: List[str]):
        """High-level retrieval lifecycle update composed of smaller focused steps."""
        if not retrieved_ids:
            return
        self._decay_inhibition()
        self._update_inhibition_and_churn(retrieved_ids)
        territories_seen = self._frontier_and_condensation(retrieved_ids)
        self._update_reward_and_temperature(territories_seen, len(retrieved_ids))

    # --- Helper decomposition (complexity reduction) ---
    def _decay_inhibition(self):
        for ms in self._mem.values():
            if ms.inhibition > 0.0:
                ms.inhibition *= 0.98

    def _update_inhibition_and_churn(self, retrieved_ids: List[str]):
        n = len(retrieved_ids)
        for i in range(n):
            ms_i = self._mem.get(retrieved_ids[i])
            if not ms_i:
                continue
            for j in range(i + 1, n):
                ms_j = self._mem.get(retrieved_ids[j])
                if not ms_j:
                    continue
                inc = 0.05
                ms_i.inhibition += inc
                ms_j.inhibition += inc
                key = (min(ms_i.territory, ms_j.territory), max(ms_i.territory, ms_j.territory))
                last = self._pair_last_tick.get(key)
                if last is not None:
                    dt = self._tick - last
                    if dt > 0:
                        self._pair_churn[key] = self._pair_churn.get(key, 0.0) + 1.0 / (1.0 + dt)
                self._pair_last_tick[key] = self._tick

    def _frontier_and_condensation(self, retrieved_ids: List[str]) -> set:
        territories_seen = set()
        for rid in retrieved_ids:
            ms = self._mem.get(rid)
            if not ms:
                continue
            territories_seen.add(ms.territory)
            if ms.novelty >= self.frontier_novelty_threshold and ms.boredom < 0.5:
                c = self._territory_frontier.get(ms.territory, 0) + 1
                self._territory_frontier[ms.territory] = c
                if c >= self.frontier_patience:
                    self._split_territory(ms.territory)
                    self._territory_frontier[ms.territory] = 0
                    self._events.append((self._tick, "territory_split", {"from": ms.territory}))
            if (ms.boredom >= self.condensation_boredom and ms.confidence >= self.condensation_conf and ms.mass >= self.condensation_mass):
                self._pending_condense.append(ms.id)
        return territories_seen

    def _update_reward_and_temperature(self, territories_seen: set, retrieved_len: int):
        uniq = len(territories_seen)
        reward = float(uniq)
        if self._reward_ema == 0.0:
            self._reward_ema = reward
        else:
            self._reward_ema = (1 - self._reward_alpha) * self._reward_ema + self._reward_alpha * reward
        churn_density = 0.0
        if self._pair_churn:
            recent_pairs = [v for k, v in self._pair_churn.items() if (self._tick - self._pair_last_tick.get(k, self._tick)) <= self.exploration_churn_window]
            if recent_pairs:
                churn_density = sum(recent_pairs) / len(recent_pairs)
        diversity_delta = max(0.0, reward - self._reward_ema)
        self._exploration_temp = min(1.0, 0.5 * (diversity_delta / (self._reward_ema + 1e-6)) + 0.5 * (churn_density / (1.0 + churn_density)))
        self._events.append((self._tick, "retrieval_cycle", {"retrieved": retrieved_len, "uniq_territories": uniq, "exploration_temp": self._exploration_temp}))

    def exploration_temperature(self) -> float:
        return self._exploration_temp

    def _split_territory(self, territory_id: int):
        """Frontier split: high-novelty subset gets a new territory id."""
        members = [ms for ms in self._mem.values() if ms.territory == territory_id]
        if len(members) < 6:
            return
        # Partition by novelty above median
        novs = sorted(m.novelty for m in members)
        median = novs[len(novs)//2]
        new_id = self._next_territory_id
        self._next_territory_id += 1
        moved = 0
        for m in members:
            if m.novelty > median and m.boredom < 0.7:
                m.territory = new_id
                moved += 1
        if moved == 0:
            return
        self._territory_frontier[new_id] = 0
        # telemetry
        try:
            self._split_counter += 1
            self._events.append((self._tick, "territory_split_complete", {"from": territory_id, "to": new_id, "moved": moved}))
        except Exception:
            pass

    def _territory_radius(self, territory: int) -> float:
        """Estimate a territory 'radius' (p95 of member distances to centroid) conservatively."""
        d = list(self._territory_member_dists.get(territory, []) or [])
        if not d:
            # fallback conservative guess
            return min(0.6, max(0.05, self._territory_tau))
        d_sorted = sorted(d)
        idx = int(len(d_sorted) * 0.95) - 1
        idx = max(0, min(len(d_sorted) - 1, idx))
        return float(d_sorted[idx])

    def _merge_territories(self, to_tid: int, from_tid: int) -> bool:
        """Merge 'from_tid' into 'to_tid'. Returns True if merged."""
        if to_tid == from_tid:
            return False
        if to_tid not in self._territory_centroids or from_tid not in self._territory_centroids:
            return False
        # Conservative gating: only merge small-ish territories to avoid catastrophic consolidation
        ca = self._territory_counts.get(to_tid, 0)
        cb = self._territory_counts.get(from_tid, 0)
        if ca + cb > 1000:
            return False

        # weighted centroid merge
        ca = max(1, ca)
        cb = max(1, cb)
        ca_float = float(ca)
        cb_float = float(cb)
        a = self._territory_centroids[to_tid]
        b = self._territory_centroids[from_tid]
        new_c = [ (a[i] * ca_float + b[i] * cb_float) / (ca_float + cb_float) for i in range(len(a)) ]
        self._territory_centroids[to_tid] = new_c
        self._territory_counts[to_tid] = ca + cb
        self._territory_centroid_norms[to_tid] = math.sqrt(sum(x * x for x in new_c))

        # merge member dist buffers
        da = list(self._territory_member_dists.get(to_tid, []))
        db = list(self._territory_member_dists.get(from_tid, []))
        merged = deque((da + db)[-1024:], maxlen=1024)
        self._territory_member_dists[to_tid] = merged

        # reassign members
        moved = 0
        for ms in self._mem.values():
            if ms.territory == from_tid:
                ms.territory = to_tid
                moved += 1

        # remove from_tid
        self._territory_centroids.pop(from_tid, None)
        self._territory_counts.pop(from_tid, None)
        self._territory_centroid_norms.pop(from_tid, None)
        self._territory_member_dists.pop(from_tid, None)

        # telemetry
        try:
            self._merge_counter += 1
            self._events.append((self._tick, "territory_merge", {"from": from_tid, "to": to_tid, "moved": moved}))
        except Exception:
            pass
        return True

    def _maybe_diffuse(self, now: float):
        if self.diffusion_interval <= 0:
            return
        if self._tick % self.diffusion_interval != 0:
            return
        # Sample a few territories; for each do pair balancing
        terr_groups: Dict[int, List[MemoryState]] = {}
        for ms in self._mem.values():
            terr_groups.setdefault(ms.territory, []).append(ms)
        terr_ids = list(terr_groups.keys())
        self._rng.shuffle(terr_ids)
        terr_ids = terr_ids[: min(8, len(terr_ids))]
        for tid in terr_ids:
            group = terr_groups[tid]
            if len(group) < 2:
                continue
            # sample a few pairs
            pairs = min(5, len(group) // 2)
            for _ in range(pairs):
                a, b = self._rng.sample(group, 2)
                if a.mass == b.mass:
                    continue
                hi, lo = (a, b) if a.mass > b.mass else (b, a)
                delta = (hi.mass - lo.mass) * self.diffusion_kappa
                if delta <= 0:
                    continue
                hi.mass -= delta
                lo.mass += delta

        # After diffusion step, attempt conservative merges among sampled territories
        try:
            # gating parameters (assumptions): merge when centroid distance <= 0.5 * tau
            merge_beta = 0.5
            # require that at least one territory is 'small' or combined size under limit
            for i in range(len(terr_ids)):
                for j in range(i + 1, len(terr_ids)):
                    ta = terr_ids[i]
                    tb = terr_ids[j]
                    if ta not in self._territory_centroids or tb not in self._territory_centroids:
                        continue
                    # centroid distance
                    d = self._cosine_distance(self._territory_centroids[ta], self._territory_centroids[tb])
                    # radii
                    ra = self._territory_radius(ta)
                    rb = self._territory_radius(tb)
                    # conservative merge predicate
                    if d <= merge_beta * self._territory_tau and max(ra, rb) <= 1.25 * min(self._territory_tau, 0.6) and (self._territory_counts.get(ta,0) + self._territory_counts.get(tb,0) < 500):
                        # merge smaller into larger
                        if self._territory_counts.get(ta,0) >= self._territory_counts.get(tb,0):
                            self._merge_territories(ta, tb)
                        else:
                            self._merge_territories(tb, ta)
        except Exception:
            pass

    # ---------------- Utility Accessors / Mutators ----------------
    def get_state(self, memory_id: str) -> Optional[MemoryState]:
        return self._mem.get(memory_id)

    def degrade(self, ids: Iterable[str], ttl_floor: int):
        """Aggressively lower ttl for originals post-condensation."""
        floor = int(max(1, ttl_floor))
        for mid in ids:
            ms = self._mem.get(mid)
            if not ms:
                continue
            ms.ttl = min(ms.ttl, floor)
            # Increase boredom slightly to encourage eventual pruning
            ms.boredom = min(1.0, ms.boredom + 0.1)

    # ---------------- Engram (Structural Consolidation) ----------------
    def register_engram(self, ids: List[str]):
        """Register a structural engram cluster without injecting new semantic content.

        Effects:
          - Stores the id list for introspection / stats.
          - Light boredom & inhibition nudge to originals to encourage turnover.
        """
        cleaned = [i for i in ids if i in self._mem]
        if len(cleaned) < 2:
            return
        self._engrams.append(cleaned)
        for mid in cleaned:
            ms = self._mem.get(mid)
            if not ms:
                continue
            ms.boredom = min(1.0, ms.boredom + 0.05)
            ms.inhibition += 0.05
        self._events.append((self._tick, "engram", {"size": len(cleaned)}))

    def composite_score_for(self, memory_id: str) -> Optional[float]:
        ms = self._mem.get(memory_id)
        if not ms:
            return None
        return ms.composite_score(self._tick, self.recency_half_life_ticks, self.boredom_weight)

    def exploratory_weight(self, memory_id: str) -> float:
        ms = self._mem.get(memory_id)
        if not ms:
            return 0.0
        # Higher for novelty and low boredom; modest mass contributes.
        novelty_term = ms.novelty * (1.0 - 0.5 * ms.boredom)
        mass_term = ms.mass / (ms.mass + 8.0)
        recency_term = 0.5 ** (max(0, self._tick - ms.last_touch_tick) / max(1.0, self.recency_half_life_ticks))
        return 0.5 * novelty_term + 0.3 * (1.0 - ms.boredom) + 0.2 * recency_term + 0.1 * mass_term

    # ---------------- Events & Persistence ----------------
    def consume_events(self) -> List[tuple]:
        out = list(self._events)
        self._events.clear()
        return out

    def peek_events(self, limit: int = 50) -> List[tuple]:
        """Return up to the most recent 'limit' events without consuming them."""
        if limit <= 0:
            return []
        return list(self._events)[-limit:]

    def to_dict(self) -> dict:
        return {
            "version": self._persistence_version,
            "tick": self._tick,
            "mem": {mid: {
                "created": ms.created,
                "ttl": ms.ttl,
                "confidence": ms.confidence,
                "mass": ms.mass,
                "heat": ms.heat,
                "last_touch_tick": ms.last_touch_tick,
                "novelty": ms.novelty,
                "territory": ms.territory,
                "use_count": ms.use_count,
                "boredom": ms.boredom,
                "inhibition": ms.inhibition,
            } for mid, ms in self._mem.items()},
            "engrams": self._engrams,
            "frontier": self._territory_frontier,
            "next_territory": self._next_territory_id,
            "reward_ema": self._reward_ema,
            "pair_churn": {str(k): v for k, v in self._pair_churn.items()},
            "pair_last": {str(k): v for k, v in self._pair_last_tick.items()},
            "territory_centroids": {str(k): v for k, v in self._territory_centroids.items()},
            "territory_counts": {str(k): v for k, v in self._territory_counts.items()},
            "territory_member_dists": {str(k): list(v) for k, v in self._territory_member_dists.items()},
            "nn_distances": list(self._nn_distances),
            "territory_tau": float(self._territory_tau),
            "split_counter": int(getattr(self, "_split_counter", 0)),
            "merge_counter": int(getattr(self, "_merge_counter", 0)),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoidMemoryManager":
        inst = cls()
        try:
            inst._tick = int(data.get("tick", 0))
            mem = data.get("mem", {})
            now = time.time()
            for mid, d in mem.items():
                inst._mem[mid] = MemoryState(
                    id=mid,
                    created=float(d.get("created", now)),
                    ttl=int(d.get("ttl", 0)),
                    confidence=float(d.get("confidence", 0.0)),
                    mass=float(d.get("mass", 1.0)),
                    heat=float(d.get("heat", 0.0)),
                    last_touch_tick=int(d.get("last_touch_tick", 0)),
                    novelty=float(d.get("novelty", 0.5)),
                    territory=int(d.get("territory", 0)),
                    use_count=int(d.get("use_count", 0)),
                    boredom=float(d.get("boredom", 0.0)),
                    inhibition=float(d.get("inhibition", 0.0)),
                )
            inst._engrams = list(data.get("engrams", []))
            inst._territory_frontier = {int(k): int(v) for k, v in data.get("frontier", {}).items()}
            inst._next_territory_id = int(data.get("next_territory", inst._next_territory_id))
            inst._reward_ema = float(data.get("reward_ema", 0.0))
            # load centroids and counts
            for k, v in data.get("territory_centroids", {}).items():
                try:
                    tid = int(k)
                    inst._territory_centroids[tid] = list(v)
                except Exception:
                    continue
            for k, v in data.get("territory_counts", {}).items():
                try:
                    tid = int(k)
                    inst._territory_counts[tid] = int(v)
                except Exception:
                    continue
            for k, v in data.get("territory_member_dists", {}).items():
                try:
                    tid = int(k)
                    inst._territory_member_dists[tid] = deque(list(v), maxlen=1024)
                except Exception:
                    continue
            inst._nn_distances = deque(list(data.get("nn_distances", [])), maxlen=5000)
            inst._territory_tau = float(data.get("territory_tau", inst._territory_tau))
            try:
                inst._split_counter = int(data.get("split_counter", 0))
            except Exception:
                inst._split_counter = 0
            try:
                inst._merge_counter = int(data.get("merge_counter", 0))
            except Exception:
                inst._merge_counter = 0
            inst._pair_churn = {}
            for k, v in data.get("pair_churn", {}).items():
                try:
                    a, b = k.strip("() ").split(",")
                    inst._pair_churn[(int(a), int(b))] = float(v)
                except Exception:
                    continue
            inst._pair_last_tick = {}
            for k, v in data.get("pair_last", {}).items():
                try:
                    a, b = k.strip("() ").split(",")
                    inst._pair_last_tick[(int(a), int(b))] = int(v)
                except Exception:
                    continue
        except Exception:
            pass
        return inst

    def save_json(self, path: str) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False)
            return True
        except Exception:
            return False

    @classmethod
    def load_json(cls, path: str) -> "VoidMemoryManager | None":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception:
            return None


__all__ = ["VoidMemoryManager", "MemoryState"]
