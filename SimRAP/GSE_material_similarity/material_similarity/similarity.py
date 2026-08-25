#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""L1-L4 similarity blocks moved verbatim from the original script."""

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import NULL
from .chemistry import substitution_weight


def exact_element_site_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    ca = a["element_site_counts"]
    cb = b["element_site_counts"]
    if len(ca) > len(cb):
        ca, cb = cb, ca
    return int(sum(min(int(v), int(cb.get(k, 0))) for k, v in ca.items()))


def l2_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, int]:
    matched = exact_element_site_overlap(a, b)
    denom = max(int(a["num_atoms"]), int(b["num_atoms"]), 1)
    return float(matched / denom), matched


def _expanded_unmatched_elements(
    counts_a: Dict[str, int], counts_b: Dict[str, int]
) -> Tuple[List[str], List[str], int]:
    """Remove exact same-element matches and return expanded unmatched atom lists."""
    keys = set(counts_a) | set(counts_b)
    exact = 0
    rem_a: List[str] = []
    rem_b: List[str] = []
    for sym in sorted(keys):
        na = int(counts_a.get(sym, 0))
        nb = int(counts_b.get(sym, 0))
        m = min(na, nb)
        exact += m
        if na > m:
            rem_a.extend([sym] * (na - m))
        if nb > m:
            rem_b.extend([sym] * (nb - m))
    return rem_a, rem_b, exact


def l3_l4_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float, int, float]:
    """
    Requires exact Wyckoff occupancy. Returns (L3, L4, exact_match_atoms, bonus_equivalent_atoms).
    """
    if a["wy_signature"] != b["wy_signature"]:
        raise ValueError("L3/L4 requested for non-identical Wyckoff occupancy")

    total = max(int(a["num_atoms"]), 1)
    exact_total = 0
    bonus_total = 0.0

    all_wy = sorted(set(a["site_elements"]) | set(b["site_elements"]))
    for wy in all_wy:
        ca = a["site_elements"].get(wy, {})
        cb = b["site_elements"].get(wy, {})
        rem_a, rem_b, exact = _expanded_unmatched_elements(ca, cb)
        exact_total += exact

        if not rem_a or not rem_b:
            continue

        # Exact Wyckoff occupancy should make lengths equal. Rectangular matching is still
        # supported defensively; unmatched atoms receive zero bonus.
        weights = np.zeros((len(rem_a), len(rem_b)), dtype=float)
        any_positive = False
        for i, sa in enumerate(rem_a):
            for j, sb in enumerate(rem_b):
                w = substitution_weight(sa, sb)
                weights[i, j] = w
                if w > 0:
                    any_positive = True

        if any_positive:
            rows, cols = linear_sum_assignment(-weights)
            bonus_total += float(weights[rows, cols].sum())

    l3 = float(exact_total / total)
    l4 = float(min(1.0, l3 + bonus_total / total))
    return l3, l4, int(exact_total), float(bonus_total)


def calculate_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Standalone requested function shape: (L1, L2, L3, L4)."""
    if int(a["spg_number"]) != int(b["spg_number"]):
        return 0.0, NULL, NULL, NULL
    if a["wy_signature"] != b["wy_signature"]:
        l2, _ = l2_similarity(a, b)
        return 1.0, l2, NULL, NULL
    l3, l4, _, _ = l3_l4_similarity(a, b)
    return 1.0, 1.0, l3, l4
