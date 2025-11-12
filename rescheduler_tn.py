#!/usr/bin/env python3
"""
rescheduler_tn.py

ACO + fast rule-based rescheduler adapted to accept rail_sim_tn_event.py output.

Usage examples:
# ACO only
python rescheduler_tn.py --events ./out/events.csv --config ./out/config.json --out ./out/aco_out --solver aco

# Rule-based only (fast deterministic baseline)
python rescheduler_tn.py --events ./out/events.csv --config ./out/config.json --out ./out/rule_out --solver rule

# Run both and pick best (compare cost = weighted holds + remaining conflicts as penalty)
python rescheduler_tn.py --events ./out/events.csv --config ./out/config.json --out ./out/both_out --solver both --ants 40 --iters 100
"""
import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# plotting (optional)
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# default safety headway margin (seconds) used by algorithms
DEFAULT_SAFETY_MARGIN_S = 300
SAFETY_MARGIN_S = DEFAULT_SAFETY_MARGIN_S  # may be overwritten by CLI


# -------------------------
# Input loader (flexible)
# -------------------------
def load_events(path: Path) -> List[Dict[str, Any]]:
    events = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # time detection (robust)
            if "time_s" in row and row["time_s"] != "":
                t = float(row["time_s"])
            elif "time" in row and row["time"] != "":
                t = float(row["time"])
            elif "t_s" in row and row["t_s"] != "":
                t = float(row["t_s"])
            else:
                # fallback: first numeric
                found = None
                for k, v in row.items():
                    try:
                        if v is not None and v != "":
                            found = float(v)
                            break
                    except Exception:
                        continue
                t = float(found) if found is not None else 0.0

            train = row.get("train") or row.get("train_id") or row.get("tid") or ""
            ev_raw = (row.get("event") or row.get("event_type") or "").strip().upper()
            if ev_raw in ("ENTER", "ENTER_BLOCK", "ENTER_BLOCK_TIME", "DEPART", "DEPART_STATION"):
                event = "ENTER_BLOCK"
            elif ev_raw in ("EXIT", "EXIT_BLOCK", "RELEASE_BLOCK", "LEAVE_BLOCK", "ARRIVE_STATION"):
                event = "RELEASE_BLOCK"
            else:
                event = ev_raw if ev_raw else "OTHER"

            block = row.get("block") or row.get("blk") or row.get("block_id") or ""
            block = str(block).strip()

            # position: accept pos_km or pos_m (keep internal pos_m)
            if "pos_km" in row and row["pos_km"] != "":
                try:
                    pos_km = float(row["pos_km"])
                    pos_m = pos_km * 1000.0
                except Exception:
                    pos_m = float(row.get("pos_m", 0) or 0)
            else:
                pos_m = float(row.get("pos_m", row.get("pos", 0) or 0))

            events.append({
                "t_s": int(round(float(t))),
                "train": str(train),
                "event": event,
                "block": block,
                "pos_m": float(pos_m),
                "raw_event": ev_raw
            })
    return events


# -------------------------
# Utilities: group by train and build block intervals
# -------------------------
def group_events_by_train(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by = defaultdict(list)
    for ev in events:
        by[ev["train"]].append(ev)
    return by


def build_block_intervals(events: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    For each block produce list of (enter_time_s, release_time_s, train_id).
    Relies on event types normalized to 'ENTER_BLOCK' and 'RELEASE_BLOCK'.
    """
    intervals = defaultdict(list)
    last_enter = {}
    evs = sorted(events, key=lambda e: (e["t_s"], e["train"]))
    for ev in evs:
        t = ev["t_s"]
        tid = ev["train"]
        blk = ev["block"]
        if ev["event"] == "ENTER_BLOCK":
            last_enter[(tid, blk)] = t
        elif ev["event"] == "RELEASE_BLOCK":
            key = (tid, blk)
            enter_t = last_enter.pop(key, None)
            if enter_t is None:
                # If release without matched enter, assume a small duration
                enter_t = max(0, t - 5)
            intervals[blk].append((enter_t, t, tid))
    # any unmatched enters -> assume release near end
    max_t = max((e["t_s"] for e in evs), default=0)
    for (tid, blk), enter_t in list(last_enter.items()):
        intervals[blk].append((enter_t, max_t + 10, tid))
    # sort intervals per block by enter time
    for blk in intervals:
        intervals[blk].sort(key=lambda x: x[0])
    return intervals


# -------------------------
# Conflict detection
# -------------------------
def detect_conflicts(intervals: Dict[str, List[Tuple[int, int, str]]], min_headway_s: int) -> List[Dict[str, Any]]:
    """
    Detect both OVERLAP (intervals overlapping) and HEADWAY (gap smaller than min_headway_s).
    Only checks adjacent intervals after sorting per-block which is sufficient for detecting overlaps/headways
    between occupancies when sorted by enter time.
    Returns sorted list of conflict dicts.
    """
    conflicts = []
    for blk, ivs in intervals.items():
        if not ivs:
            continue
        ivs_sorted = sorted(ivs, key=lambda x: x[0])
        for i in range(len(ivs_sorted) - 1):
            a_enter, a_rel, a_tid = ivs_sorted[i]
            b_enter, b_rel, b_tid = ivs_sorted[i + 1]
            if a_tid == b_tid:
                continue
            # overlap
            overlap = max(0, min(a_rel, b_rel) - max(a_enter, b_enter))
            if overlap > 0:
                conflicts.append({
                    "type": "OVERLAP",
                    "block": blk,
                    "a_enter": a_enter, "a_release": a_rel, "a_tid": a_tid,
                    "b_enter": b_enter, "b_release": b_rel, "b_tid": b_tid,
                    "overlap_s": overlap
                })
            # headway (b enters too soon after a releases)
            if a_rel <= b_enter:
                gap = b_enter - a_rel
                if 0 < gap < min_headway_s:
                    conflicts.append({
                        "type": "HEADWAY",
                        "block": blk,
                        "a_enter": a_enter, "a_release": a_rel, "a_tid": a_tid,
                        "b_enter": b_enter, "b_release": b_rel, "b_tid": b_tid,
                        "gap_s": gap, "required_s": min_headway_s
                    })
    # sort conflicts by occurrence time (b_enter if present else a_enter)
    return sorted(conflicts, key=lambda x: x.get("b_enter", x.get("a_enter", 0)))


# -------------------------
# Fast Rule-based Rescheduler (linear-ish)
# -------------------------
def rule_rescheduler(events: List[Dict[str, Any]], min_headway_s: int, safety_margin_s: int = SAFETY_MARGIN_S) -> Tuple[Dict[str, int], int]:
    """
    Fast deterministic baseline:
      - For each block process occupancy intervals in chronological order.
      - If overlap: delay the later train (b) by overlap + safety.
      - If headway too small: delay later train to satisfy headway + safety.
    Returns (hold_map, remaining_conflicts_after).
    """
    intervals = build_block_intervals(events)
    hold_map = defaultdict(int)

    for blk, ivs in intervals.items():
        ivs_sorted = sorted(ivs, key=lambda x: x[0])
        changed = True
        # iterate until stable (usually a few passes)
        while changed:
            changed = False
            for i in range(len(ivs_sorted) - 1):
                a_enter, a_rel, a_tid = ivs_sorted[i]
                b_enter, b_rel, b_tid = ivs_sorted[i + 1]
                if a_tid == b_tid:
                    continue
                # effective times with holds
                a_rel_eff = a_rel + hold_map.get(a_tid, 0)
                b_enter_eff = b_enter + hold_map.get(b_tid, 0)

                # overlap check
                overlap = max(0, min(a_rel_eff, b_rel + hold_map.get(b_tid, 0)) - max(a_enter + hold_map.get(a_tid, 0), b_enter_eff))
                if overlap > 0:
                    need = int(math.ceil(overlap + safety_margin_s))
                    hold_map[b_tid] += need
                    changed = True
                # headway check
                a_rel_eff = a_rel + hold_map.get(a_tid, 0)
                b_enter_eff = b_enter + hold_map.get(b_tid, 0)
                gap_after = b_enter_eff - a_rel_eff
                if 0 < gap_after < min_headway_s:
                    need = int(math.ceil(min_headway_s - gap_after + safety_margin_s))
                    hold_map[b_tid] += need
                    changed = True

    # Build delayed intervals and detect remaining conflicts
    delayed_intervals = defaultdict(list)
    for blk, ivs in intervals.items():
        for enter_t, release_t, tid in ivs:
            delayed_intervals[blk].append((enter_t + hold_map.get(tid, 0), release_t + hold_map.get(tid, 0), tid))

    rem_conf = 0
    for blk, ivs in delayed_intervals.items():
        ivs_sorted = sorted(ivs, key=lambda x: x[0])
        for j in range(len(ivs_sorted) - 1):
            a_enter, a_rel, a_tid = ivs_sorted[j]
            b_enter, b_rel, b_tid = ivs_sorted[j + 1]
            if a_tid == b_tid:
                continue
            overlap = max(0, min(a_rel, b_rel) - max(a_enter, b_enter))
            if overlap > 0:
                rem_conf += 1
            gap = b_enter - a_rel
            if 0 < gap < min_headway_s:
                rem_conf += 1

    return {str(k): int(v) for k, v in hold_map.items()}, rem_conf


# -------------------------
# ACO rescheduler (reworked)
# -------------------------
class ACORescheduler:
    """
    Reworked ACO rescheduler:
      - precomputes original intervals per block (and per-train enters/releases)
      - uses an improved heuristic that considers severity, train degree (how many conflicts a train participates in) and priority
      - constructs solutions per ant via pheromone+heuristic probabilistic rule
      - runs a quick local repair (greedy passes over blocks, fixing adjacent violations) on each ant solution before scoring
      - uses efficient conflict counting by checking only adjacent occupancies (after sorting)
    """

    def __init__(self, conflicts: List[Dict[str, Any]], trains_cfg: Dict[str, Dict[str, Any]],
                 events_by_train: Dict[str, List[Dict[str, Any]]], min_headway_s: int,
                 ants: int = 30, iters: int = 80, alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
                 q0: float = 0.9, seed: Optional[int] = None, penalty_weight: float = 200000.0,
                 max_hold_per_train: int = 4 * 3600, local_repair_passes: int = 3):
        self.conflicts = conflicts
        self.trains_cfg = trains_cfg
        self.events_by_train = events_by_train  # raw events per train
        self.min_headway = min_headway_s
        self.ants = ants
        self.iters = iters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.penalty_weight = penalty_weight
        self.max_hold_per_train = max_hold_per_train
        self.local_repair_passes = local_repair_passes
        if seed is not None:
            random.seed(seed)

        # Precompute: original intervals per block (enter, release, tid)
        # We'll rely on build_block_intervals being done externally, but reconstruct here for safety:
        all_events = []
        for evs in self.events_by_train.values():
            all_events.extend(evs)
        self.original_intervals = build_block_intervals(all_events)

        # Precompute per-train sorted events for quick lookup
        self.events_by_train_sorted = {tid: sorted(evs, key=lambda e: (e["t_s"], e["event"]))
                                       for tid, evs in self.events_by_train.items()}

        # Train-degree (number of conflicts a train participates in)
        self.train_conf_count = Counter()
        for c in self.conflicts:
            self.train_conf_count[str(c["a_tid"])] += 1
            self.train_conf_count[str(c["b_tid"])] += 1

        # Number of binary decisions = number of conflicts
        self.n = len(self.conflicts)
        # initialize pheromone and heuristics
        self.pheromone = [[1.0, 1.0] for _ in range(self.n)]
        self.heuristic = [self._heuristic_for_conf(c) for c in self.conflicts]

    def _heuristic_for_conf(self, conf: Dict[str, Any]) -> Tuple[float, float]:
        """
        Return a (h_a, h_b) desirability pair where higher value means more desirable to delay that train.
        We prefer delaying lower-priority trains and trains with higher conflict-degree less so we prefer to keep them unchanged.
        Incorporates severity (overlap/gap), priority, and train degrees.
        """
        # severity (seconds)
        if conf["type"] == "HEADWAY":
            severity = max(0, conf.get("required_s", self.min_headway) - conf.get("gap_s", 0)) + SAFETY_MARGIN_S
        else:
            severity = conf.get("overlap_s", 1) + SAFETY_MARGIN_S

        a_tid, b_tid = str(conf["a_tid"]), str(conf["b_tid"])
        prio_a = int(self.trains_cfg.get(a_tid, {}).get("priority", 2))
        prio_b = int(self.trains_cfg.get(b_tid, {}).get("priority", 2))

        # weight higher if higher priority (we want to avoid delaying higher priority)
        weight_a = 1.0 + (prio_a - 1) * 0.6
        weight_b = 1.0 + (prio_b - 1) * 0.6

        # degree penalty - trains with many conflicts are "expensive" to delay (or maybe we prefer to delay them slightly differently)
        deg_a = 1.0 + 0.1 * self.train_conf_count.get(a_tid, 0)
        deg_b = 1.0 + 0.1 * self.train_conf_count.get(b_tid, 0)

        # combine severity, priority and degree into desirability: delaying train X more desirable if it has lower weight & lower degree
        s = max(1.0, severity)
        base = math.log(1.0 + s)
        # raw desirability
        ra = 1.0 / (1.0 + base * weight_a * deg_a)
        rb = 1.0 / (1.0 + base * weight_b * deg_b)
        # normalize
        tot = ra + rb
        if tot <= 0:
            return (0.5, 0.5)
        return (ra / tot, rb / tot)

    def _build_delayed_intervals(self, added: Dict[str, int]) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Apply added delays (per-train seconds) to original intervals and return delayed intervals per block.
        This is linear in number of intervals.
        """
        delayed = defaultdict(list)
        for blk, ivs in self.original_intervals.items():
            for enter_t, release_t, tid in ivs:
                shift = added.get(str(tid), 0)
                delayed[blk].append((enter_t + shift, release_t + shift, str(tid)))
        return delayed

    def _count_conflicts_adjacent(self, delayed_intervals: Dict[str, List[Tuple[int, int, str]]]) -> Tuple[int, List[Tuple[str, Tuple[int, int, str], Tuple[int, int, str]]]]:
        """
        Efficient conflict counter: for each block, sort by enter time and check adjacent pairs only.
        Returns total_conflict_count and list of conflict details (block, a_occ, b_occ) for use by local repair.
        """
        rem_conf = 0
        conflict_list = []
        for blk, ivs in delayed_intervals.items():
            if not ivs:
                continue
            ivs_sorted = sorted(ivs, key=lambda x: x[0])
            for i in range(len(ivs_sorted) - 1):
                a_enter, a_rel, a_tid = ivs_sorted[i]
                b_enter, b_rel, b_tid = ivs_sorted[i + 1]
                if a_tid == b_tid:
                    continue
                # overlap
                overlap = max(0, min(a_rel, b_rel) - max(a_enter, b_enter))
                if overlap > 0:
                    rem_conf += 1
                    conflict_list.append((blk, ivs_sorted[i], ivs_sorted[i + 1]))
                    continue
                # headway check
                if a_rel <= b_enter:
                    gap = b_enter - a_rel
                    if 0 < gap < self.min_headway:
                        rem_conf += 1
                        conflict_list.append((blk, ivs_sorted[i], ivs_sorted[i + 1]))
        return rem_conf, conflict_list

    def _local_repair(self, added: Dict[str, int], max_passes: int = 3) -> Dict[str, int]:
        """
        Quick greedy repair: iterate over blocks and fix adjacent violations by adding minimal additional delay
        to the later train. Repeat for a small number of passes.
        This is intentionally simple and fast (not optimal), but reduces remaining conflicts dramatically.
        """
        # copy to modify
        added = dict(added)
        for p in range(max_passes):
            delayed = self._build_delayed_intervals(added)
            changed = False
            # iterate blocks and fix adjacent pairs
            for blk, ivs in delayed.items():
                if not ivs:
                    continue
                ivs_sorted = sorted(ivs, key=lambda x: x[0])
                for i in range(len(ivs_sorted) - 1):
                    a_enter, a_rel, a_tid = ivs_sorted[i]
                    b_enter, b_rel, b_tid = ivs_sorted[i + 1]
                    if a_tid == b_tid:
                        continue
                    # overlap
                    if b_enter < a_rel:
                        overlap = a_rel - b_enter
                        # add minimal to later train (b) to push its enter to at least a_rel + safety
                        need = int(math.ceil(overlap + SAFETY_MARGIN_S))
                        cur = added.get(b_tid, 0)
                        newval = min(self.max_hold_per_train, cur + need)
                        if newval != cur:
                            added[b_tid] = newval
                            changed = True
                            # update local values to reflect change for remaining pairs in this block
                            ivs_sorted[i + 1] = (b_enter + (newval - cur), b_rel + (newval - cur), b_tid)
                        continue
                    # headway violation
                    if a_rel <= b_enter:
                        gap = b_enter - a_rel
                        if 0 < gap < self.min_headway:
                            need = int(math.ceil(self.min_headway - gap + SAFETY_MARGIN_S))
                            cur = added.get(b_tid, 0)
                            newval = min(self.max_hold_per_train, cur + need)
                            if newval != cur:
                                added[b_tid] = newval
                                changed = True
                                ivs_sorted[i + 1] = (b_enter + (newval - cur), b_rel + (newval - cur), b_tid)
            if not changed:
                break
        return added

    def _evaluate(self, decisions: List[int]) -> Tuple[float, Dict[str, int], int]:
        """
        Construct per-train added delays from a binary decision vector, run local repair, then score.
        Returns (score, added_map, remaining_conflicts).
        """
        # build initial added map by splitting each conflict's need
        added = defaultdict(int)
        for i, conf in enumerate(self.conflicts):
            choice = decisions[i]  # 0 => bias to a, 1 => bias to b
            if conf["type"] == "HEADWAY":
                need = max(0, conf.get("required_s", self.min_headway) - conf.get("gap_s", 0)) + SAFETY_MARGIN_S
            else:
                need = conf.get("overlap_s", 1) + SAFETY_MARGIN_S
            half = int(math.ceil(need / 2.0))
            if choice == 0:
                added[str(conf["a_tid"])] += half
                added[str(conf["b_tid"])] += int(math.ceil(need - half))
            else:
                added[str(conf["b_tid"])] += half
                added[str(conf["a_tid"])] += int(math.ceil(need - half))

        # cap holds
        for k in list(added.keys()):
            if added[k] > self.max_hold_per_train:
                added[k] = self.max_hold_per_train

        # local repair (fast greedy)
        added_repaired = self._local_repair(added, max_passes=self.local_repair_passes)

        # build delayed intervals and count conflicts efficiently using adjacent checks
        delayed = self._build_delayed_intervals(added_repaired)
        rem_conf, _ = self._count_conflicts_adjacent(delayed)

        # delay cost weighted by priority
        total_delay_cost = 0.0
        for tid_str, sec in added_repaired.items():
            prio = int(self.trains_cfg.get(str(tid_str), {}).get("priority", 2))
            weight = 1.0 + (prio - 1) * 0.6
            total_delay_cost += sec * weight

        score = rem_conf * self.penalty_weight + total_delay_cost
        return score, {str(k): int(v) for k, v in added_repaired.items()}, rem_conf

    def run(self) -> Tuple[float, Dict[str, int], int, float]:
        """
        Run ACO optimizer. Returns (best_score, best_added_map, remaining_conflicts, accuracy_percent).
        """
        if self.n == 0:
            return 0.0, {}, 0, 100.0

        best_score = float("inf")
        best_added = {}
        best_rem = None
        n_conf_before = len(self.conflicts)

        for it in range(self.iters):
            ant_sols = []
            # construct solutions per ant
            for a in range(self.ants):
                decisions = []
                for i in range(self.n):
                    ph0, ph1 = self.pheromone[i]
                    h0, h1 = self.heuristic[i]
                    val0 = (ph0 ** self.alpha) * (h0 ** self.beta)
                    val1 = (ph1 ** self.alpha) * (h1 ** self.beta)
                    denom = val0 + val1
                    p0 = 0.5 if denom == 0 else val0 / denom
                    if random.random() < self.q0:
                        choice = 0 if p0 >= 0.5 else 1
                    else:
                        choice = 0 if random.random() < p0 else 1
                    decisions.append(choice)
                score, added_map, rem = self._evaluate(decisions)
                ant_sols.append((decisions, score, added_map, rem))
                if score < best_score:
                    best_score = score
                    best_added = added_map.copy()
                    best_rem = rem

            # pheromone evaporation
            for i in range(self.n):
                self.pheromone[i][0] *= (1.0 - self.rho)
                self.pheromone[i][1] *= (1.0 - self.rho)
                self.pheromone[i][0] = max(self.pheromone[i][0], 1e-9)
                self.pheromone[i][1] = max(self.pheromone[i][1], 1e-9)

            # deposit pheromone using top-K ants (elitist)
            ant_sols.sort(key=lambda x: x[1])
            topk = max(1, int(0.15 * len(ant_sols)))
            best_ant_score = ant_sols[0][1]
            worst_ant_score = ant_sols[-1][1]
            for idx in range(topk):
                decs, sc, added_map, rem = ant_sols[idx]
                if worst_ant_score - best_ant_score > 1e-9:
                    contrib = (worst_ant_score - sc) / (worst_ant_score - best_ant_score)
                else:
                    contrib = 1.0
                contrib *= 2.0
                for i, ch in enumerate(decs):
                    self.pheromone[i][ch] += contrib

            if (it + 1) % max(1, self.iters // 10) == 0 or it == 0:
                print(f"ACO iter {it + 1}/{self.iters} best_score={best_score:.1f} best_rem_conf={best_rem}")

        # compute accuracy
        n_conf_after = best_rem if best_rem is not None else 0
        acc = 100.0 if n_conf_before == 0 else 100.0 * (n_conf_before - n_conf_after) / n_conf_before

        return best_score, best_added, n_conf_after, acc


# -------------------------
# Plot helper
# -------------------------
def plot_time_distance(events: List[Dict[str, Any]], out_path: Path, title: str = "Time–Distance"):
    if plt is None:
        print("matplotlib not available; skipping plot:", out_path)
        return
    try:
        if pd is not None:
            df = pd.DataFrame(events)
            if df.empty:
                print("No events to plot:", out_path)
                return
            fig, ax = plt.subplots(figsize=(12, 6))
            for tid, sub in df.groupby("train"):
                sub_s = sub.sort_values("t_s")
                ax.plot(sub_s["t_s"] / 60.0, sub_s["pos_m"] / 1000.0, marker='.', markersize=2, linewidth=0.6, alpha=0.6)
            ax.set_xlabel("Time (min)"); ax.set_ylabel("Position (km)"); ax.set_title(title); ax.grid(True)
            fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
            print("Saved plot to", out_path); return
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(12, 6))
    by = defaultdict(list)
    for ev in events:
        by[ev["train"]].append((ev["t_s"], ev["pos_m"]))
    for tid, seq in by.items():
        seq_sorted = sorted(seq, key=lambda x: x[0])
        ax.plot([t / 60.0 for t, _ in seq_sorted], [p for _, p in seq_sorted], marker='.', linewidth=0.7, markersize=2, alpha=0.6)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Position (km)"); ax.set_title(title); ax.grid(True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print("Saved plot to", out_path)


# -------------------------
# CLI main
# -------------------------
def main():
    global SAFETY_MARGIN_S
    p = argparse.ArgumentParser(description="ACO + rule rescheduler for TN event-driven simulator")
    p.add_argument("--events", required=True, help="Path to events CSV from simulator")
    p.add_argument("--config", required=True, help="Path to config.json exported by simulator")
    p.add_argument("--out", required=True, help="Output folder for rescheduler")
    p.add_argument("--ants", type=int, default=40)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-hold", type=int, default=4 * 3600,
                   help="Maximum hold per train in seconds (default 4h)")
    p.add_argument("--solver", choices=["aco", "rule", "both"], default="aco", help="Which solver to run")
    p.add_argument("--safety-margin", type=int, default=DEFAULT_SAFETY_MARGIN_S,
                   help="Safety margin in seconds added to delays (default 300s)")
    p.add_argument("--local-repair-passes", type=int, default=3,
                   help="Number of greedy local-repair passes per ant (default 3)")
    args = p.parse_args()

    SAFETY_MARGIN_S = args.safety_margin

    events_path = Path(args.events); cfg_path = Path(args.config); outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if not events_path.exists(): print("Events file not found:", events_path); return
    if not cfg_path.exists(): print("Config file not found:", cfg_path); return

    events = load_events(events_path)
    with open(cfg_path, "r") as f: cfg = json.load(f)
    min_headway = cfg.get("section", {}).get("min_headway_s", cfg.get("section", {}).get("min_headway", 30))

    # Save a copy of input ordering for debugging
    before_csv = outdir / "reschedule_before_events.csv"
    with open(before_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_s", "train", "event", "block", "pos_m"])
        writer.writeheader()
        for ev in sorted(events, key=lambda e: (e["t_s"], e["train"])):
            writer.writerow({"t_s": ev["t_s"], "train": ev["train"], "event": ev["event"], "block": ev["block"], "pos_m": ev["pos_m"]})

    print("Loaded", len(events), "events. Building block intervals...")
    intervals = build_block_intervals(events)
    conflicts = detect_conflicts(intervals, min_headway)
    print(f"Detected {len(conflicts)} conflicts (overlap/headway).")
    plot_time_distance(events, outdir / "time_distance_before.png", title="Before: Time–Distance")

    trains_cfg = {}
    trains_config = cfg.get("trains", [])
    
    # Handle different possible structures
    if isinstance(trains_config, list):
        for t in trains_config:
            if isinstance(t, dict):
                tid = t.get("id") or t.get("train") or t.get("train_id")
                if tid: 
                    trains_cfg[str(tid)] = t
            elif isinstance(t, str):
                # If it's just a string, use it as train ID with default config
                trains_cfg[str(t)] = {"priority": 2}
    elif isinstance(trains_config, dict):
        # If trains is already a dictionary keyed by train IDs
        trains_cfg = {str(tid): config for tid, config in trains_config.items()}
    
    # Debug: print what we found
    print(f"Loaded configuration for {len(trains_cfg)} trains")
    if trains_cfg:
        sample = list(trains_cfg.items())[:3]
        print(f"Sample train configs: {sample}")
    events_by_train = group_events_by_train(events)

    results = {}
    # Rule baseline
    if args.solver in ("rule", "both"):
        print("Running rule-based rescheduler (fast baseline)...")
        start = time.time()
        holds_rule, rem_rule = rule_rescheduler(events, min_headway, safety_margin_s=SAFETY_MARGIN_S)
        dt = time.time() - start
        # compute cost metric (weighted holds + big penalty for remaining conflicts)
        cost_rule = 0.0
        for tid, sec in holds_rule.items():
            prio = int(trains_cfg.get(str(tid), {}).get("priority", 2))
            weight = 1.0 + (prio - 1) * 0.6
            cost_rule += sec * weight
        cost_rule += rem_rule * 100000.0
        results["rule"] = (cost_rule, holds_rule, rem_rule)
        acc_rule = 100.0 if len(conflicts) == 0 else 100.0 * (len(conflicts) - rem_rule) / len(conflicts)
        print(f"Rule holds computed for {len(holds_rule)} trains, remaining_conflicts={rem_rule}, accuracy={acc_rule:.2f}% (took {dt:.1f}s)")

    # ACO
    if args.solver in ("aco", "both"):
        print("Running ACO rescheduler...")
        start = time.time()
        aco = ACORescheduler(conflicts, trains_cfg, events_by_train, min_headway,
                             ants=args.ants, iters=args.iters, alpha=1.0, beta=2.0, rho=0.1,
                             q0=0.9, seed=args.seed, penalty_weight=200000.0,
                             max_hold_per_train=args.max_hold, local_repair_passes=args.local_repair_passes)
        score_aco, holds_aco, rem_aco, acc_aco = aco.run()
        dt = time.time() - start
        results["aco"] = (score_aco, holds_aco, rem_aco)
        print(f"ACO finished: score={score_aco:.1f} rem_conflicts={rem_aco} holds={len(holds_aco)} accuracy={acc_aco:.2f}% (took {dt:.1f}s)")

    # pick best solver result if both
    if args.solver == "both":
        best_key = min(results.items(), key=lambda kv: kv[1][0])[0]
        print(f"Best solver = {best_key.upper()} (score={results[best_key][0]:.1f})")
        _, chosen_holds, chosen_rem = results[best_key]
    else:
        key = args.solver
        chosen_holds = results[key][1] if key in results else {}
        chosen_rem = results[key][2] if key in results else 0

    # write actions.json
    actions = {str(tid): {"hold_s": int(sec)} for tid, sec in chosen_holds.items()}
    with open(outdir / "actions.json", "w") as f:
        json.dump(actions, f, indent=2)

    # apply holds to events -> produce rescheduled events
    rescheduled = []
    for ev in events:
        new_ev = ev.copy()
        shift = actions.get(str(ev["train"]), {}).get("hold_s", 0)
        new_ev["t_s"] = int(ev["t_s"] + shift)
        rescheduled.append(new_ev)

    after_csv = outdir / "reschedule_after_events.csv"
    with open(after_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_s", "train", "event", "block", "pos_m"])
        writer.writeheader()
        for ev in sorted(rescheduled, key=lambda e: (e["t_s"], e["train"])):
            writer.writerow({"t_s": ev["t_s"], "train": ev["train"], "event": ev["event"], "block": ev["block"], "pos_m": ev["pos_m"]})

    plot_time_distance(rescheduled, outdir / "time_distance_after.png", title="After: Time–Distance")
    # overlay compare
    if plt is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        for tid, seq in group_events_by_train(events).items():
            seq_sorted = sorted(seq, key=lambda x: x["t_s"])
            ax.plot([x["t_s"] / 60.0 for x in seq_sorted], [x["pos_m"] / 1000.0 for x in seq_sorted], color="red", alpha=0.2)
        for tid, seq in group_events_by_train(rescheduled).items():
            seq_sorted = sorted(seq, key=lambda x: x["t_s"])
            ax.plot([x["t_s"] / 60.0 for x in seq_sorted], [x["pos_m"] / 1000.0 for x in seq_sorted], color="green", alpha=0.4)
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Position (km)"); ax.set_title("Before (red) vs After (green)")
        ax.grid(True); fig.tight_layout(); fig.savefig(outdir / "time_distance_compare.png"); plt.close(fig)

    with open(outdir / "reschedule_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["train", "hold_s"])
        hold_map_for_csv = {tid: actions.get(str(tid), {}).get("hold_s", 0) for tid in set(ev["train"] for ev in events)}
        for tid in sorted(hold_map_for_csv.keys()):
            w.writerow([tid, hold_map_for_csv.get(tid, 0)])

    print("Outputs written to:", outdir.resolve())
    print("Sample actions (first 10):")
    for i, (tid, act) in enumerate(actions.items()):
        if i >= 10: break
        print(f"  {tid}: hold {act['hold_s']} s")


if __name__ == "__main__":
    main()
