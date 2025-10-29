#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lotto_combo_gen_pro_v5.py
-------------------------
A more aggressive (but still heuristic) generator + selector for 6/37 Lotto.
Goals vs v4:
- Larger candidate POOL with softmax sampling then GREEDY COVER selection to
  maximize weighted coverage of hot numbers, pairs, and triplets while keeping
  high diversity between chosen combos.
- Multi-start greedy + local-search refinement to maximize coverage quality.
- Smarter candidate sampling with pair/trip affinity boosts and spread biasing.
- Ensemble jitter & reuse penalties to diversify suggestions between runs.
- Optional light AUTO‑TUNE over the last K draws to pick good knobs for your data.
- Stronger anti‑similarity to recent draws and within‑set overlap constraints.
- Backtest (holdout-at-0) kept and expanded.

⚠️ Nothing here can guarantee wins. It only biases selection using historical
patterns. Use at your own risk.

Author: ChatGPT (Yoel edition, pro v5)
Date: 2025-09-07
"""
from __future__ import annotations

import argparse, sys, os, math, json, time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable, Set

import numpy as np
import pandas as pd
import requests  # <== added for Telegram sending

# ============================== Utilities ==============================

def coerce_numeric(df: pd.DataFrame, min_ratio: float = 0.90) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    numeric_cols = [c for c in df2.columns if df2[c].notna().mean() >= min_ratio]
    return df2[numeric_cols]


def _domain_fit_score(series: pd.Series, num_max: int) -> float:
    s = series.dropna()
    if s.empty: return 0.0
    frac_main = ((s >= 1) & (s <= num_max)).mean()
    uniq = s.nunique()
    return float(frac_main) + 0.10 * min(uniq, num_max) / float(num_max)


def best_six_domain_cols(df_num: pd.DataFrame, num_max: int) -> List[str]:
    scores = [(c, _domain_fit_score(df_num[c].dropna(), num_max)) for c in df_num.columns]
    scores.sort(key=lambda x: -x[1])
    return [c for c, s in scores[:6] if s > 0]


def detect_main_number_columns(df_num: pd.DataFrame, num_max: int = 37, strong_max: int = 7,
                               min_main_unique: Optional[int] = None) -> Tuple[List[str], Optional[str]]:
    if min_main_unique is None:
        min_main_unique = max(12, min(num_max, int(0.5 * num_max)))
    cols = list(df_num.columns)
    if not cols: return [], None
    main, bonus = [], []
    for c in cols:
        s = df_num[c].dropna()
        if s.empty: continue
        frac_main = ((s >= 1) & (s <= num_max)).mean()
        frac_bonus = ((s >= 1) & (s <= strong_max)).mean()
        uniq = s.nunique()
        if frac_bonus > 0.85 and uniq <= strong_max:
            bonus.append(c)
        elif frac_main > 0.85 and uniq >= min_main_unique:
            main.append(c)
    if len(main) > 6:
        spreads = [(c, float(df_num[c].max() - df_num[c].min())) for c in main]
        main = [c for c, _ in sorted(spreads, key=lambda x: -x[1])][:6]
    return main[:6], (bonus[0] if bonus else None)


def zscore_counter(values: Dict, ddof: int = 1) -> Dict:
    if not values: return {}
    arr = np.array(list(values.values()), dtype=float)
    if arr.size <= 1:
        mu, sd = (arr.mean(), 1.0)
    else:
        mu, sd = arr.mean(), arr.std(ddof=ddof) or 1.0
    return {k: float((v - mu) / (sd if sd else 1.0)) for k, v in values.items()}


def merge_zscores(dicts: List[Dict], weights: Optional[List[float]] = None) -> Dict:
    if not dicts: return {}
    if weights is None: weights = [1.0] * len(dicts)
    agg = Counter()
    for d, w in zip(dicts, weights):
        for k, v in d.items():
            agg[k] += float(w) * float(v)
    return zscore_counter(dict(agg))


# ====================== Preference Builders ======================

def prefs_with_decay(df_num: pd.DataFrame, main_cols: List[str], history_cap: int, num_max: int,
                     newest_first: bool = True, decay: float = 0.97, window: int = 500
                     ) -> Tuple[Dict[int, float], Dict[tuple, float], int, float, float]:
    if len(df_num) == 0:
        return {}, {}, 3, 6 * (1 + num_max) / 2.0, max(12.0, num_max / 2.5)
    cap = min(history_cap, len(df_num) - 1)
    start = max(0, cap - window + 1)
    hist = df_num.loc[start:cap, main_cols].dropna(how="any")
    hist = hist.iloc[::-1].reset_index(drop=True)  # newest first
    if hist.empty:
        return {}, {}, 3, 6 * (1 + num_max) / 2.0, max(12.0, num_max / 2.5)

    num_w = Counter(); pair_w = Counter()
    even_sum_w = 0.0; sum_sum_w = 0.0; w_total = 0.0

    for age, (_, r) in enumerate(hist.iterrows()):
        w = decay ** age
        arr = [int(x) for x in r[main_cols].tolist() if pd.notna(x)]
        arr = [v for v in arr if 1 <= v <= num_max]
        if len(arr) < 6: continue
        for v in arr: num_w[v] += w
        for a, b in combinations(sorted(set(arr)), 2): pair_w[(a, b)] += w
        even_sum_w += w * sum(1 for v in arr if v % 2 == 0)
        sum_sum_w  += w * sum(arr)
        w_total += w

    num_pref = zscore_counter(num_w)
    pair_pref = zscore_counter(pair_w)
    even_target = int(round(even_sum_w / w_total)) if w_total > 0 else 3
    sum_mu = float(sum_sum_w / w_total) if w_total > 0 else (6 * (1 + num_max) / 2.0)
    sum_sd = max(12.0, num_max / 2.5)
    return num_pref, pair_pref, even_target, sum_mu, sum_sd


def build_last12_preferences(df_num: pd.DataFrame, main_cols: List[str], history_cap: int,
                             num_max: int, newest_first: bool = True
                             ) -> Tuple[Dict[int, float], Dict[tuple, float], int, float, float]:
    if len(df_num) == 0:
        return {}, {}, 3, 6 * (1 + num_max) / 2.0, max(12.0, num_max / 2.5)

    def history_slice_for_i(i: int) -> pd.DataFrame:
        cap = min(history_cap, len(df_num) - 1)
        lo, hi = (i + 1, cap) if newest_first else (0, min(i - 1, cap))
        if hi < lo: return pd.DataFrame(columns=main_cols)
        return df_num.loc[lo:hi, main_cols]

    end_idx = min(12, len(df_num) - 1)
    num_pref = Counter(); pair_pref = Counter(); sums, evens = [], []

    for i in range(0, end_idx + 1):
        hist = history_slice_for_i(i).dropna(how="any")

        def row_in_range(r) -> bool:
            arr = [int(x) for x in r[main_cols].tolist() if pd.notna(x)]
            return len(arr) >= 6 and all(1 <= v <= num_max for v in arr)

        if not hist.empty:
            hist = hist[hist.apply(row_in_range, axis=1)]

        row_vals = [int(x) for x in df_num.loc[i, main_cols].tolist() if pd.notna(x)]
        if row_vals:
            sums.append(sum(row_vals))
            evens.append(sum(1 for v in row_vals if v % 2 == 0))

        if hist.empty: continue

        freq = Counter()
        for _, r in hist.iterrows():
            arr = [int(x) for x in r[main_cols].tolist() if pd.notna(x)]
            freq.update([v for v in arr if 1 <= v <= num_max])
        if freq: num_pref.update(zscore_counter(freq))

        pc = Counter()
        for _, r in hist.iterrows():
            arr = [int(x) for x in r[main_cols].tolist() if pd.notna(x)]
            arr = [v for v in arr if 1 <= v <= num_max]
            for a, b in combinations(sorted(set(arr)), 2):
                pc[(a, b)] += 1
        if pc: pair_pref.update(zscore_counter(pc))

    num_pref = zscore_counter(num_pref) if num_pref else {}
    pair_pref = zscore_counter(pair_pref) if pair_pref else {}
    even_target = int(round(np.mean(evens))) if evens else 3
    sum_mu = float(np.mean(sums)) if sums else (6 * (1 + num_max) / 2.0)
    sum_sd = float(np.std(sums, ddof=1)) if len(sums) > 1 else max(12.0, num_max / 2.5)
    if sum_sd == 0: sum_sd = max(12.0, num_max / 2.5)
    return num_pref, pair_pref, even_target, sum_mu, sum_sd


def precompute_triplet_loglifts(history_df: pd.DataFrame, main_cols: List[str]) -> Dict[tuple, float]:
    draws = [set(int(x) for x in r[main_cols].tolist() if pd.notna(x)) for _, r in history_df.iterrows()]
    n = len(draws)
    cnt = Counter(x for s in draws for x in s)
    tc = Counter()
    for s in draws:
        arr = sorted(s)
        for a, b, c in combinations(arr, 3):
            tc[(a, b, c)] += 1
    loglift = {}
    denom = (n + 1)
    for (a, b, c), abc in tc.items():
        pabc = (abc + 1) / denom
        pa   = (cnt.get(a, 0) + 1) / denom
        pb   = (cnt.get(b, 0) + 1) / denom
        pc   = (cnt.get(c, 0) + 1) / denom
        loglift[(a, b, c)] = float(np.log(pabc / (pa * pb * pc)))
    return loglift


def waiting_time_pref(df_num: pd.DataFrame, main_cols: List[str], num_max: int, history_cap: int) -> Dict[int, float]:
    cap = min(history_cap, len(df_num) - 1)
    hist = df_num.loc[:cap, main_cols].dropna(how="any")
    last_seen = {}
    for i, (_, r) in enumerate(hist.iterrows()):
        for v in [int(x) for x in r[main_cols].tolist() if pd.notna(x)]:
            if 1 <= v <= num_max: last_seen[v] = i  # smaller index == newer here
    if not last_seen: return {}
    max_i = max(last_seen.values())
    gaps = {n: (max_i - idx) for n, idx in last_seen.items()}
    return zscore_counter(gaps)


# ===================== Candidate Generation =====================

def history_seen_set(df_num: pd.DataFrame, main_cols: List[str], num_max: int,
                     history_cap: int) -> Set[tuple]:
    cap = min(history_cap, len(df_num) - 1)
    seen: Set[tuple] = set()
    for _, r in df_num.loc[:cap, main_cols].dropna(how="any").iterrows():
        arr = [int(x) for x in r[main_cols].tolist() if pd.notna(x)]
        arr = [v for v in arr if 1 <= v <= num_max]
        if len(arr) >= 6:
            seen.add(tuple(sorted(arr[:6])))
    return seen


def too_similar_to_recent(combo: tuple, df_num: pd.DataFrame, main_cols: List[str],
                          num_max: int, recent_m: int = 50, k: int = 5) -> bool:
    if recent_m <= 0: return False
    recent = df_num.loc[:min(recent_m, len(df_num)-1), main_cols].dropna(how="any")
    S = set(combo)
    for _, r in recent.iterrows():
        arr = [int(x) for x in r[main_cols].tolist() if pd.notna(x)]
        arr = [v for v in arr if 1 <= v <= num_max]
        if len(S & set(arr)) >= k: return True
    return False


@dataclass
class Prefs:
    num: Dict[int, float]
    pair: Dict[tuple, float]
    trip: Dict[tuple, float]
    even_target: int
    sum_mu: float
    sum_sd: float
    wait: Dict[int, float]


def build_prefs(df_view: pd.DataFrame, main_cols: List[str], args) -> Prefs:
    # windows ensemble for num/pair + waiting + optional triplets
    if args.windows:
        num_ps, pair_ps, even_ts, sum_mus, sum_sds = [], [], [], [], []
        for w in [int(x) for x in str(args.windows).split(",")]:
            if args.use_decay:
                n, p, even_t, s_mu, s_sd = prefs_with_decay(
                    df_view, main_cols, args.history_cap, args.num_max,
                    newest_first=True, decay=args.decay, window=int(w)
                )
            else:
                n, p, even_t, s_mu, s_sd = build_last12_preferences(
                    df_view, main_cols, args.history_cap, args.num_max, newest_first=True
                )
            num_ps.append(n); pair_ps.append(p); even_ts.append(even_t); sum_mus.append(s_mu); sum_sds.append(s_sd)
        num_pref  = merge_zscores(num_ps)
        pair_pref = merge_zscores(pair_ps)
        even_target = int(round(np.mean(even_ts)))
        sum_mu = float(np.mean(sum_mus)); sum_sd = float(np.mean(sum_sds))
    else:
        if args.use_decay:
            num_pref, pair_pref, even_target, sum_mu, sum_sd = prefs_with_decay(
                df_view, main_cols, args.history_cap, args.num_max, newest_first=True,
                decay=args.decay, window=args.window
            )
        else:
            num_pref, pair_pref, even_target, sum_mu, sum_sd = build_last12_preferences(
                df_view, main_cols, args.history_cap, args.num_max, newest_first=True
            )

    trip_pref = {}
    if args.use_triplets:
        cap = min(args.history_cap, len(df_view) - 1)
        hist = df_view.loc[:cap, main_cols].dropna(how="any")
        raw = precompute_triplet_loglifts(hist, main_cols)
        if raw: trip_pref = zscore_counter(raw)

    wait_pref = waiting_time_pref(df_view, main_cols, args.num_max, args.history_cap) if args.w_wait > 0 else {}

    return Prefs(num=num_pref, pair=pair_pref, trip=trip_pref,
                 even_target=even_target, sum_mu=sum_mu, sum_sd=sum_sd, wait=wait_pref)


def combo_score(vals: tuple, prefs: Prefs, args) -> float:
    vals = tuple(sorted(vals))
    # number and pair components
    s_num = sum(prefs.num.get(v, 0.0) for v in vals) / 6.0
    pairs = list(combinations(vals, 2))
    s_pair = float(np.mean([prefs.pair.get((min(a,b), max(a,b)), 0.0) for a, b in pairs])) if pairs else 0.0
    # triplets (optional)
    s_trip = 0.0
    if args.use_triplets and prefs.trip:
        trips = list(combinations(vals, 3))
        s_trip = float(np.mean([prefs.trip.get(tuple(sorted(t)), 0.0) for t in trips])) if trips else 0.0
    # parity & sum envelope
    even = sum(1 for v in vals if v % 2 == 0)
    s_par = 1.0 - min(abs(even - prefs.even_target), 3) / 3.0
    s_sum = math.exp(-((sum(vals) - prefs.sum_mu) ** 2) / (2 * (prefs.sum_sd ** 2)))
    # waiting time
    s_wait = float(np.mean([prefs.wait.get(v, 0.0) for v in vals])) if prefs.wait else 0.0
    # adjacency penalties (avoid too many consecutive numbers or tight clusters)
    consec = sum(1 for a,b in zip(vals, vals[1:]) if b == a+1)
    spread = max(vals) - min(vals)
    pen_consec = 0.15 * consec  # 0, 0.15, 0.30, ...
    pen_cluster = 0.0 if spread >= (args.num_max * 0.45) else 0.20

    base = (args.w_num  * s_num
            + args.w_pair * s_pair
            + (args.w_trip * s_trip if args.use_triplets else 0.0)
            + args.w_sum  * s_sum
            + args.w_par  * s_par
            + args.w_wait * s_wait)

    return base - pen_consec - pen_cluster


def _standardize_masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = values[mask]
    if vals.size <= 1:
        values[mask] = 0.0 if vals.size == 0 else vals - vals.mean()
        return values
    mu = float(vals.mean())
    sd = float(vals.std())
    if sd < 1e-9:
        values[mask] = vals - mu
    else:
        values[mask] = (vals - mu) / sd
    return values


def sample_combo(all_nums: np.ndarray, base_pref: np.ndarray, prefs: Prefs,
                 args, rng: np.random.Generator) -> tuple:
    chosen: List[int] = []
    available = np.ones_like(all_nums, dtype=bool)

    base = np.array(base_pref, dtype=float)
    if not np.isfinite(base).all():
        base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    if base.std() > 1e-9:
        base = (base - base.mean()) / base.std()
    elif base.size:
        base = base - base.mean()

    pair_boost = float(args.candidate_pair_boost)
    trip_boost = float(args.candidate_trip_boost)
    spread_boost = float(args.candidate_spread_boost)

    for _ in range(6):
        logits = base.copy()

        if chosen:
            pair_scores = np.zeros_like(logits)
            trip_scores = np.zeros_like(logits)
            spread_scores = np.zeros_like(logits)
            chosen_set = set(chosen)

            for idx, n in enumerate(all_nums):
                if not available[idx]:
                    continue
                n_int = int(n)
                # Pair affinity with already chosen numbers
                if pair_boost != 0.0:
                    pair_vals = [prefs.pair.get((min(n_int, c), max(n_int, c)), 0.0) for c in chosen_set]
                    if pair_vals:
                        pair_scores[idx] = float(np.mean(pair_vals))
                # Triplet affinity once at least two numbers chosen
                if trip_boost != 0.0 and len(chosen) >= 2 and prefs.trip:
                    trip_vals = [
                        prefs.trip.get(tuple(sorted((n_int, a, b))), 0.0)
                        for a, b in combinations(chosen_set, 2)
                    ]
                    if trip_vals:
                        trip_scores[idx] = float(np.mean(trip_vals))
                if spread_boost != 0.0:
                    gap = min(abs(n_int - c) for c in chosen_set)
                    spread_scores[idx] = gap / max(1.0, float(args.num_max))

            mask = available.copy()
            if pair_boost != 0.0:
                logits += pair_boost * _standardize_masked(pair_scores, mask.copy())
            if trip_boost != 0.0:
                logits += trip_boost * _standardize_masked(trip_scores, mask.copy())
            if spread_boost != 0.0:
                logits += spread_boost * _standardize_masked(spread_scores, mask.copy())

        logits[~available] = -np.inf
        finite_mask = np.isfinite(logits) & available
        if not finite_mask.any():
            choices = np.where(available)[0]
            idx = int(rng.choice(choices))
        else:
            logits = logits.copy()
            max_logit = float(np.nanmax(logits[finite_mask]))
            logits[finite_mask] = logits[finite_mask] - max_logit
            weights = np.zeros_like(logits)
            weights[finite_mask] = np.exp(np.clip(args.gamma, 0.05, 5.0) * logits[finite_mask])
            total = weights[finite_mask].sum()
            if total <= 0 or not np.isfinite(total):
                choices = np.where(available)[0]
                idx = int(rng.choice(choices))
            else:
                weights[finite_mask] /= total
                idx = int(rng.choice(len(all_nums), p=weights))

        chosen.append(int(all_nums[idx]))
        available[idx] = False

    return tuple(sorted(chosen))


def generate_pool(df_view: pd.DataFrame, main_cols: List[str], prefs: Prefs, args,
                  rng: np.random.Generator, seen_hist: Set[tuple]) -> List[tuple]:
    all_nums = np.arange(1, args.num_max + 1)
    base_w = np.array([prefs.num.get(int(n), 0.0) for n in all_nums], dtype=float)
    if not np.isfinite(base_w).all() or base_w.sum() == 0:
        base_w = np.nan_to_num(base_w, nan=0.0, posinf=0.0, neginf=0.0)

    pool: List[tuple] = []
    tries = 0
    # Initial "wheel" seeds for coverage
    if args.wheel_cover and args.total_needed >= 10:
        hot_sorted = sorted(range(1, args.num_max + 1), key=lambda n: prefs.num.get(n, 0.0), reverse=True)
        top_pool  = hot_sorted[:max(1, args.wheel_top // 2)]
        next_pool = hot_sorted[max(1, args.wheel_top // 2):args.wheel_top]

        def seed_one():
            pick = []
            rng.shuffle(top_pool); rng.shuffle(next_pool)
            pick.extend(top_pool[:min(args.wheel_min_top, 6)])
            pick.extend(next_pool[:max(0, min(args.wheel_min_next, 6 - len(pick)))])
            rest = [n for n in hot_sorted if n not in pick]
            while len(pick) < 6 and rest:
                idx = int(rng.integers(0, len(rest)))
                pick.append(rest.pop(idx))
            return tuple(sorted(pick[:6]))

        seeded = set()
        for _ in range(min(args.total_needed, 8)):
            sc = seed_one()
            if sc not in seeded and sc not in seen_hist:
                seeded.add(sc); pool.append(sc)

    while len(pool) < args.pool and tries < args.max_tries:
        tries += 1
        c = sample_combo(all_nums, base_w, prefs, args, rng)
        if args.no_history_unique is False and c in seen_hist:
            continue
        # keep a generous pool, do not apply recent similarity yet (done during final selection)
        pool.append(c)
    return pool


# ===================== Greedy COVER Selection =====================

@dataclass
class CoverWeights:
    num: float = 1.0
    pair: float = 2.0
    trip: float = 4.0


def marginal_gain(c: tuple, covered_nums: Set[int], covered_pairs: Set[tuple], covered_trips: Set[tuple],
                  weights: CoverWeights, prefs: Prefs) -> float:
    gain = 0.0
    # numbers
    for v in c:
        if v not in covered_nums:
            gain += weights.num * max(0.0, prefs.num.get(v, 0.0))
    # pairs
    for a, b in combinations(c, 2):
        key = (min(a,b), max(a,b))
        if key not in covered_pairs:
            gain += weights.pair * max(0.0, prefs.pair.get(key, 0.0))
    # trips
    if prefs.trip:
        for t in combinations(c, 3):
            key = tuple(sorted(t))
            if key not in covered_trips:
                gain += weights.trip * max(0.0, prefs.trip.get(key, 0.0))
    return float(gain)


def select_greedy_cover(pool: List[tuple], df_view: pd.DataFrame, main_cols: List[str],
                        prefs: Prefs, args, rng: np.random.Generator) -> List[tuple]:
    covered_nums: Set[int] = set()
    covered_pairs: Set[tuple] = set()
    covered_trips: Set[tuple] = set()
    selected: List[tuple] = []
    usage_counts: Counter = Counter()

    cw = CoverWeights(num=args.cw_num, pair=args.cw_pair, trip=args.cw_trip)

    # Precompute base scores to break ties
    base_scores = {c: combo_score(c, prefs, args) for c in pool}

    # Filter pool against recent similarity & adjacency constraints softly
    filt_pool = []
    for c in pool:
        if args.recent_block_k > 0 and too_similar_to_recent(c, df_view, main_cols, args.num_max,
                                                             recent_m=args.recent_block_m, k=args.recent_block_k):
            continue
        # light constraint: avoid 3 or more consecutive numbers in a combo
        consec = sum(1 for a,b in zip(c, c[1:]) if b == a+1)
        if consec >= 3: 
            continue
        filt_pool.append(c)
    pool = filt_pool if filt_pool else pool

    # Greedy add
    tries = 0
    usage_target = max(1, int(getattr(args, "cover_usage_target", 1)))
    usage_penalty = float(getattr(args, "cover_usage_penalty", 0.0))
    while len(selected) < args.total_needed and pool and tries < (len(pool) * 4):
        tries += 1
        # Evaluate marginal gain for all remaining
        best_c, best_val = None, -1e18
        for c in pool:
            # diversity: enforce max overlap within selected
            if selected and any(len(set(c) & set(s)) >= args.dedupe_k for s in selected):
                continue
            mg = marginal_gain(c, covered_nums, covered_pairs, covered_trips, cw, prefs)
            # small addition: include base score so we don't select purely for coverage
            val = mg + args.cover_base_mix * base_scores[c]
            if usage_penalty > 0.0:
                repeat = sum(max(0, usage_counts[v] - usage_target + 1) for v in c)
                if repeat:
                    val -= usage_penalty * repeat
            if val > best_val:
                best_val, best_c = val, c
        if best_c is None:
            break
        # add to selected
        selected.append(best_c)
        # update covered sets
        for v in best_c: covered_nums.add(v)
        for a,b in combinations(best_c, 2): covered_pairs.add((min(a,b), max(a,b)))
        if prefs.trip:
            for t in combinations(best_c, 3): covered_trips.add(tuple(sorted(t)))
        for v in best_c:
            usage_counts[v] += 1
        # remove chosen from pool
        pool.remove(best_c)

    # If still short, fill by top base score under diversity
    if len(selected) < args.total_needed and pool:
        leftovers = sorted(pool, key=lambda c: base_scores[c], reverse=True)
        for c in leftovers:
            if any(len(set(c) & set(s)) >= args.dedupe_k for s in selected): 
                continue
            if usage_penalty > 0.0:
                repeat = sum(max(0, usage_counts[v] - usage_target + 1) for v in c)
                if repeat:
                    continue
            selected.append(c)
            for v in c:
                usage_counts[v] += 1
            if len(selected) >= args.total_needed: break

    return selected[:args.total_needed]


def set_quality(combos: List[tuple], prefs: Prefs, args) -> float:
    if not combos:
        return -1e18
    avg_score = float(np.mean([combo_score(c, prefs, args) for c in combos]))
    covered_nums: Set[int] = set()
    covered_pairs: Set[tuple] = set()
    covered_trips: Set[tuple] = set()
    for c in combos:
        covered_nums.update(c)
        for a, b in combinations(c, 2):
            covered_pairs.add((min(a, b), max(a, b)))
        if prefs.trip:
            for t in combinations(c, 3):
                covered_trips.add(tuple(sorted(t)))

    cov_num = (sum(max(0.0, prefs.num.get(v, 0.0)) for v in covered_nums)
               / max(1, len(covered_nums)))
    cov_pair = (sum(max(0.0, prefs.pair.get(p, 0.0)) for p in covered_pairs)
                / max(1, len(covered_pairs)))
    cov_trip = 0.0
    if prefs.trip:
        cov_trip = (sum(max(0.0, prefs.trip.get(t, 0.0)) for t in covered_trips)
                    / max(1, len(covered_trips)))
    spread_bonus = float(np.mean([(max(c) - min(c)) / max(1.0, args.num_max) for c in combos]))

    usage_penalty = float(getattr(args, "cover_usage_penalty", 0.0))
    usage_target = max(1, int(getattr(args, "cover_usage_target", 1)))
    repeat_penalty = 0.0
    if usage_penalty > 0.0:
        counts = Counter(v for c in combos for v in c)
        repeat_penalty = usage_penalty * sum(max(0, cnt - usage_target) for cnt in counts.values()) * 0.6

    return (avg_score
            + 0.45 * cov_num
            + 0.30 * cov_pair
            + (0.20 * cov_trip if prefs.trip else 0.0)
            + 0.05 * spread_bonus
            - repeat_penalty)


def refine_combos(combos: List[tuple], prefs: Prefs, args, rng: np.random.Generator) -> List[tuple]:
    if args.refine_iters <= 0 or not combos:
        return combos

    refined = [list(c) for c in combos]
    top_sorted = sorted(range(1, args.num_max + 1), key=lambda n: prefs.num.get(n, 0.0), reverse=True)
    top_limit = max(6, min(args.num_max, args.refine_candidates))
    top_candidates = top_sorted[:top_limit]

    for _ in range(args.refine_iters):
        changed = False
        for idx, combo in enumerate(refined):
            best_score = combo_score(tuple(sorted(combo)), prefs, args)
            best_variant = combo[:]
            others = [set(refined[j]) for j in range(len(refined)) if j != idx]

            random_candidates = list(rng.choice(range(1, args.num_max + 1), size=min(6, args.num_max), replace=False))
            candidates = list(dict.fromkeys(top_candidates + random_candidates))

            for drop_pos in range(len(combo)):
                original_val = combo[drop_pos]
                for candidate_val in candidates:
                    if candidate_val == original_val:
                        continue
                    trial = combo[:]
                    trial[drop_pos] = int(candidate_val)
                    trial_sorted = sorted(set(trial))
                    if len(trial_sorted) != 6:
                        continue
                    if not all(1 <= v <= args.num_max for v in trial_sorted):
                        continue
                    if any(len(set(trial_sorted) & other) >= args.dedupe_k for other in others):
                        continue
                    score = combo_score(tuple(trial_sorted), prefs, args)
                    if score > best_score + 1e-6:
                        best_score = score
                        best_variant = trial_sorted
            if best_variant != combo:
                refined[idx] = list(best_variant)
                changed = True
        if not changed:
            break

    return [tuple(sorted(c)) for c in refined]


def multi_start_select(df_view: pd.DataFrame, main_cols: List[str], prefs: Prefs,
                       args, rng: np.random.Generator, seen_hist: Set[tuple]) -> List[tuple]:
    runs = max(1, int(args.ensemble_runs))
    best_set: Optional[List[tuple]] = None
    best_quality = -1e18

    for _ in range(runs):
        run_rng = np.random.default_rng(rng.integers(0, 2 ** 32 - 1))
        run_args = argparse.Namespace(**vars(args))
        jitter = float(getattr(args, "ensemble_jitter", 0.0))
        if jitter > 0.0:
            jitter_norm = lambda scale: float(run_rng.normal(0.0, jitter * scale))
            run_args.gamma = max(0.35, args.gamma * (1.0 + jitter_norm(1.0)))
            run_args.cover_base_mix = float(np.clip(args.cover_base_mix + jitter_norm(0.6), 0.0, 1.0))
            run_args.candidate_pair_boost = max(0.0, args.candidate_pair_boost * (1.0 + jitter_norm(0.8)))
            run_args.candidate_trip_boost = max(0.0, args.candidate_trip_boost * (1.0 + jitter_norm(0.8)))
            run_args.candidate_spread_boost = max(0.0, args.candidate_spread_boost * (1.0 + jitter_norm(0.7)))
            run_args.cover_usage_penalty = max(0.0, args.cover_usage_penalty * (1.0 + jitter_norm(0.5)))
        pool = generate_pool(df_view, main_cols, prefs, run_args, run_rng, seen_hist)
        selected = select_greedy_cover(pool, df_view, main_cols, prefs, run_args, run_rng)
        refined = refine_combos(selected, prefs, run_args, run_rng)
        quality = set_quality(refined, prefs, run_args)
        if quality > best_quality:
            best_quality = quality
            best_set = refined

    return best_set if best_set is not None else []


# ===================== Backtest & Auto‑Tune =====================

def backtest_holdout_at_zero(df_view: pd.DataFrame, main_cols: List[str], args,
                             rng: np.random.Generator) -> Dict[int,int]:
    N = min(int(args.backtest), max(0, len(df_view) - 1))
    if N <= 0: return {}
    hist = Counter()
    for t in range(N):
        hold_vals = [int(x) for x in df_view.loc[t, main_cols].tolist() if pd.notna(x)]
        hold_set = set(v for v in hold_vals if 1 <= v <= args.num_max)
        train = df_view.loc[t+1:, main_cols]
        if train.empty: break
        # Build prefs on the training slice
        prefs = build_prefs(train, main_cols, args)
        seen_hist = history_seen_set(train, main_cols, args.num_max, args.history_cap) if not args.no_history_unique else set()
        pool = generate_pool(train, main_cols, prefs, args, rng, seen_hist)
        combos_bt = select_greedy_cover(pool, train, main_cols, prefs, args, rng)
        best = 0
        for c in combos_bt:
            hits = len(hold_set & set(c))
            if hits > best: best = hits
            if best == 6: break
        hist[best] += 1
    return hist


def evaluate_settings(df_view, main_cols, base_args, rng, steps=120) -> float:
    # returns weighted score to compare settings (heavier weight to 4+ hits)
    args = base_args
    args = argparse.Namespace(**vars(args))  # shallow copy
    args.backtest = steps
    hist = backtest_holdout_at_zero(df_view, main_cols, args, rng)
    # scoring: emphasize 4,5,6
    score = (hist.get(6,0)*30 + hist.get(5,0)*10 + hist.get(4,0)*4 +
             hist.get(3,0)*1.5 + hist.get(2,0)*0.6 + hist.get(1,0)*0.2)
    # normalize by steps
    return float(score) / max(1, steps)


def auto_tune(df_view: pd.DataFrame, main_cols: List[str], args, rng) -> argparse.Namespace:
    # light grid
    gamma_grid = [0.95, 1.10, 1.25]
    windows_grid = ["150,300,600", "300,600,1200"]
    use_trip_grid = [False, True]
    w_wait_grid = [0.0, args.w_wait]
    cover_mix_grid = [0.15, 0.3, 0.5]
    best_score = -1e18; best_cfg = None

    for g in gamma_grid:
        for w in windows_grid:
            for ut in use_trip_grid:
                for ww in w_wait_grid:
                    for cbm in cover_mix_grid:
                        cfg = argparse.Namespace(**vars(args))
                        cfg.gamma = g; cfg.windows = w; cfg.use_triplets = ut; cfg.w_wait = ww
                        cfg.cover_base_mix = cbm
                        score = evaluate_settings(df_view, main_cols, cfg, rng, steps=min(args.auto_tune, 200))
                        if score > best_score:
                            best_score, best_cfg = score, cfg
    return best_cfg if best_cfg is not None else args


# ============================== CLI ==============================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Suggest 6-number combos for 6/37 Lotto with greedy coverage selection.")
    ap.add_argument("--csv", help="Path to input CSV with draws (6 main-number columns)")
    ap.add_argument("--cols", help="Comma-separated list of the 6 main columns to use, e.g. 'A,B,C,D,E,F'")
    ap.add_argument("--bonus_col", help="Optional bonus/strong column name (ignored by generator)")

    ap.add_argument("--history_cap", type=int, default=2300, help="Cap index for historical uniqueness (default 2300)")
    ap.add_argument("--num_max", type=int, default=37, help="Maximum number in game (default 37)")
    ap.add_argument("--strong_max", type=int, default=7, help="Max for bonus/strong if present (default 7)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (omit for random seed)")
    ap.add_argument("--out", default="", help="Optional path to save the combos CSV")
    ap.add_argument("--total_needed", type=int, default=4, help="How many combos to return (4)")
    ap.add_argument("--max_tries", type=int, default=50000, help="Max sampling attempts for pool generation")
    ap.add_argument("--pool", type=int, default=5000, help="Number of candidate combos to sample before selection")

    # Orientation / preference controls
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--newest_first", action="store_true", help="Row 0 is newest (default)")
    g.add_argument("--oldest_first", action="store_true", help="Row 0 is oldest (use if CSV is chronological)")
    ap.add_argument("--use_decay", action="store_true", help="Use decay-weighted preferences (recommended)")
    ap.add_argument("--decay", type=float, default=0.97, help="Decay factor for --use_decay (default 0.97)")
    ap.add_argument("--window", type=int, default=500, help="History window size for --use_decay (default 500)")
    ap.add_argument("--windows", default="150,300,600,1200", help="Comma-separated windows for decay ensemble (empty=off)")

    # Blocking / uniqueness
    ap.add_argument("--dedupe_k", type=int, default=5, help="Block combos sharing >=k with a selected combo (default 5)")
    ap.add_argument("--recent_block_k", type=int, default=5, help="Block if share >=k with any of the last M draws (default 5)")
    ap.add_argument("--recent_block_m", type=int, default=50, help="How many recent draws to check for overlap (default 50)")
    ap.add_argument("--no_history_unique", action="store_true", help="Allow combos that appeared historically (NOT recommended)")

    # Scoring weights
    ap.add_argument("--gamma", type=float, default=1.05, help="Softmax sharpness for number sampling")
    ap.add_argument("--use_triplets", action="store_true", default=True, help="Include triplet log-lift in scoring")
    ap.add_argument("--w_num", type=float, default=0.25, help="Weight for number prefs")
    ap.add_argument("--w_pair", type=float, default=0.50, help="Weight for pair prefs")
    ap.add_argument("--w_trip", type=float, default=0.20, help="Weight for triplet prefs (if --use_triplets)")
    ap.add_argument("--w_sum", type=float, default=0.03, help="Weight for sum gaussian proximity")
    ap.add_argument("--w_par", type=float, default=0.02, help="Weight for parity target proximity")
    ap.add_argument("--w_wait", type=float, default=0.15, help="Weight for waiting-time preference")

    # Selection coverage mixing
    ap.add_argument("--cw_num", type=float, default=0.8, help="Coverage weight for numbers")
    ap.add_argument("--cw_pair", type=float, default=2.5, help="Coverage weight for pairs")
    ap.add_argument("--cw_trip", type=float, default=5.0, help="Coverage weight for triplets")
    ap.add_argument("--cover_base_mix", type=float, default=0.45, help="Mix-in of base score during coverage selection (0..1)")
    ap.add_argument("--cover_usage_penalty", type=float, default=0.08,
                    help="Penalty applied when a number is reused beyond cover_usage_target across the suggested set")
    ap.add_argument("--cover_usage_target", type=int, default=1,
                    help="How many times a number can appear across the set before the reuse penalty activates")


    # Wheel coverage for hot pools
    ap.add_argument("--wheel_cover", action="store_true", help="Seed candidates to cover hot pools across the suggestions")
    ap.add_argument("--wheel_top", type=int, default=24, help="How many top numbers define the hot pool (default 24)")
    ap.add_argument("--wheel_min_top", type=int, default=3, help="Min from top pool per combo (default 3)")
    ap.add_argument("--wheel_min_next", type=int, default=2, help="Min from next pool per combo (default 2)")

    # Candidate sampling tweaks
    ap.add_argument("--candidate_pair_boost", type=float, default=0.45,
                    help="Strength of pair-affinity boost during candidate sampling")
    ap.add_argument("--candidate_trip_boost", type=float, default=0.25,
                    help="Strength of triplet-affinity boost during candidate sampling")
    ap.add_argument("--candidate_spread_boost", type=float, default=0.12,
                    help="Encourage wider number spread during candidate sampling")

    # Multi-start & refinement controls
    ap.add_argument("--ensemble_runs", type=int, default=3,
                    help="How many multi-start attempts to try before picking the best set (default 3)")
    ap.add_argument("--refine_iters", type=int, default=2,
                    help="Local-search refinement passes per selected set (default 2; 0 disables)")
    ap.add_argument("--refine_candidates", type=int, default=28,
                    help="Top-N numbers (plus a few randoms) considered during refinement swaps (default 28)")
    ap.add_argument("--ensemble_jitter", type=float, default=0.15,
                    help="Standard deviation for random jitters applied to sampling/selection weights across ensemble runs (0 disables)")

    # Diagnostics / Output
    ap.add_argument("--debug", action="store_true", help="Print extra info (dev use)")
    ap.add_argument("--plain", action="store_true", help="Print ONLY the combos as lines like '- (1, 2, 3, 4, 5, 6)'")

    # Backtest & auto-tune
    ap.add_argument("--backtest", type=int, default=0, help="If >0, evaluate first N rows (holdout-at-0)")
    ap.add_argument("--bt_verbose", action="store_true", help="Print per-step details for the backtest")
    ap.add_argument("--auto_tune", type=int, default=0, help="Light grid-search on last K draws to choose knobs (0=off)")

    args, _unknown = ap.parse_known_args(argv)
    if not args.newest_first and not args.oldest_first:
        args.newest_first = True
    return args


def resolve_csv_path(arg_csv: Optional[str], in_ipy: bool) -> str:
    if arg_csv and os.path.exists(arg_csv):
        return arg_csv

    # חיפוש ברירת־מחדל אך ורק ב-/proj
    for p in ("/home/yoel/Downloads/yoel1.csv",):  # שים לב לפסיק -> tuple עם פריט אחד
        if os.path.exists(p):
            return p

    raise SystemExit("--csv is required (expected at /home/yoel/Downloads/yoel1.csv).")

# ============================== Telegram + CSV (minimal additions) ==============================
#TELEGRAM_BOT_TOKEN = "6929450262:AAEZnwHFihCs6CxK0PC45IPEzWVqZ_5buMI"   # לדוגמה: "123456789:AA...xyz"
#TELEGRAM_CHAT_ID   = "2131421492"   


def send_telegram(text: str, timeout: int = 15) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "ghgfhghhhgfhfgh").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "6765767").strip()
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, json=payload, timeout=timeout).raise_for_status()
    except Exception:
        pass  # stay silent

def write_runs_csv(combos: List[tuple], runs_path: Path, mode: str = "w") -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, mode, encoding="utf-8") as f:
        for c in combos:
            f.write(", ".join(map(str, c)) + "\n")  # no header, no parentheses


# ============================== Main ==============================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Seed
    if args.seed is None:
        seed = int.from_bytes(os.urandom(8), "little") & 0xFFFFFFFF
    else:
        seed = int(args.seed)
    rng = np.random.default_rng(seed)

    # CSV & orientation
    in_ipy = any(a.startswith("-f") for a in sys.argv)
    csv_path = resolve_csv_path(args.csv, in_ipy)
    df_raw = pd.read_csv(csv_path)
    df_num = coerce_numeric(df_raw, min_ratio=0.90)

    # Column selection
    if args.cols:
        main_cols = [c.strip() for c in args.cols.split(",") if c.strip()]
        missing = [c for c in main_cols if c not in df_num.columns]
        if missing:
            raise SystemExit(f"--cols not found in CSV (after numeric filter): {missing}")
        if len(main_cols) != 6:
            raise SystemExit("--cols must specify exactly 6 columns.")
    else:
        mains, bonus = detect_main_number_columns(df_num, num_max=args.num_max, strong_max=args.strong_max)
        if len(mains) != 6:
            auto = best_six_domain_cols(df_num, num_max=args.num_max)
            if len(auto) == 6:
                mains = auto
            elif len(mains) == 5 and bonus:
                mains = mains + [bonus]
            else:
                df_relaxed = coerce_numeric(df_raw, min_ratio=0.70)
                auto_relaxed = best_six_domain_cols(df_relaxed, num_max=args.num_max)
                if len(auto_relaxed) == 6:
                    df_num = df_relaxed; mains = auto_relaxed
                else:
                    numeric_cols = list(df_num.columns)
                    raise SystemExit(
                        "Could not select 6 columns automatically.\n"
                        f"Detected mains: {mains}\nDetected bonus: {bonus}\n"
                        f"Numeric columns available: {numeric_cols}\n"
                        'Try: --cols "A,B,C,D,E,F"'
                    )
        main_cols = mains

    df_view = df_num.copy()
    if args.oldest_first and not args.newest_first:
        df_view = df_view.iloc[::-1].reset_index(drop=True)

    # Optional auto-tune on the current data
    if args.auto_tune and args.auto_tune > 0 and len(df_view) > 50:
        tuned = auto_tune(df_view, main_cols, args, rng)
        if not args.plain:
            print(f"[AUTO‑TUNE] Selected: gamma={tuned.gamma}, windows={tuned.windows}, "
                  f"use_triplets={tuned.use_triplets}, w_wait={tuned.w_wait}, cover_base_mix={tuned.cover_base_mix}")
        args = tuned  # use tuned settings

    # Build prefs
    prefs = build_prefs(df_view, main_cols, args)

    # Prepare seen history (for uniqueness) on view (all except row 0 if newest first)
    seen_hist = history_seen_set(df_view, main_cols, args.num_max, args.history_cap) if not args.no_history_unique else set()

    # Generate suggestions using multi-start with optional refinement
    combos = multi_start_select(df_view, main_cols, prefs, args, rng, seen_hist)
    if not combos:
        pool = generate_pool(df_view, main_cols, prefs, args, rng, seen_hist)
        combos = select_greedy_cover(pool, df_view, main_cols, prefs, args, rng)
        combos = refine_combos(combos, prefs, args, rng)

    if args.debug:
        quality = set_quality(combos, prefs, args)
        print(f"[DEBUG] Quality={quality:.4f} avg_score={np.mean([combo_score(c, prefs, args) for c in combos]):.4f}")

    # Output (unchanged print behavior)
    if args.plain:
        for c in combos: print(c)
    else:
        for c in combos: print(c)

    # Save results to yoel1_runs.csv (same folder as input CSV)
    runs_path = Path(csv_path).with_name("yoel1_runs4.csv")
    write_runs_csv(combos, runs_path, mode="w")

    # Optional --out remains as in original
    if args.out:
        pd.DataFrame({"combo": [",".join(map(str, c)) for c in combos]}).to_csv(args.out, index=False)
        if not args.plain: print(f"Saved to: {args.out}")

    # Send to Telegram (env: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)
    tg_text = "\n".join(f"({', '.join(map(str, c))})" for c in combos)[:4096]
    if tg_text:
        send_telegram(tg_text)

    # Backtest if requested
    if args.backtest and args.backtest > 0:
        hist = backtest_holdout_at_zero(df_view, main_cols, args, rng)
        if not args.plain:
            total = sum(hist.values())
            print(f"Evaluated steps: {total}")
            for k in range(6, -1, -1):
                print(f"hits={k}: {hist.get(k, 0)}")


if __name__ == "__main__":
    main()
