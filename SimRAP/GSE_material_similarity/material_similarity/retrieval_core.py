#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Core nearest-neighbor retrieval blocks moved verbatim from the original script."""

import heapq
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constants import NULL
from .similarity import l2_similarity, l3_l4_similarity
from .tree import trie_lookup


def _filtered_ids(ids: Sequence[int], self_id: Optional[int]) -> List[int]:
    if self_id is None:
        return list(ids)
    return [rid for rid in ids if int(rid) != int(self_id)]


def _heap_push_topk(
    heap: List[Tuple[Tuple[float, ...], int, Dict[str, Any]]],
    key: Tuple[float, ...],
    rid: int,
    result: Dict[str, Any],
    k: int,
) -> None:
    item = (key, -int(rid), result)
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def retrieve_neighbors(
    query: Dict[str, Any],
    index_obj: Dict[str, Any],
    k: int,
    self_id: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tree = index_obj["tree"]
    records = index_obj["records"]
    spg = int(query["spg_number"])
    spg_node = tree.get(spg)

    profile = {
        "spg_candidate_count": 0,
        "wy_candidate_count": 0,
        "scored_candidate_count": 0,
        "retrieval_stage": "",
    }

    if spg_node is None:
        profile["retrieval_stage"] = "NO_EXACT_SPG"
        return [], profile

    spg_ids = _filtered_ids(spg_node["ids"], self_id)
    profile["spg_candidate_count"] = len(spg_ids)
    if not spg_ids:
        profile["retrieval_stage"] = "NO_EXACT_SPG_AFTER_SELF_EXCLUSION"
        return [], profile

    wy_leaf = trie_lookup(spg_node["wy_root"], query["wy_tokens"])
    wy_ids: List[int] = []
    if wy_leaf is not None:
        wy_ids = _filtered_ids(wy_leaf.get("_ids", []), self_id)
    profile["wy_candidate_count"] = len(wy_ids)

    # -------------------------------------------------------------------------
    # Exact Wyckoff occupancy -> L2=1, calculate L3/L4.
    # -------------------------------------------------------------------------
    if wy_ids:
        profile["retrieval_stage"] = "L4_EXACT_WYCKOFF"

        # Fast path through the deeper tree. Exact element@site signature means L3=L4=1.
        perfect_ids: List[int] = []
        group_leaf = trie_lookup(wy_leaf.get("_group_root"), query["group_tokens"])
        if group_leaf is not None:
            element_leaf = trie_lookup(
                group_leaf.get("_element_root"), query["element_site_tokens"]
            )
            if element_leaf is not None:
                perfect_ids = _filtered_ids(element_leaf.get("_ids", []), self_id)

        if len(perfect_ids) >= k:
            selected = sorted(perfect_ids)[:k]
            profile["scored_candidate_count"] = 0
            return [
                {
                    "record_id": rid,
                    "sim_l1": 1.0,
                    "sim_l2": 1.0,
                    "sim_l3": 1.0,
                    "sim_l4": 1.0,
                    "final_similarity": 1.0,
                    "exact_element_site_match_atoms": int(query["num_atoms"]),
                    "l4_bonus_equivalent_atoms": 0.0,
                    "retrieval_stage": "L4_PERFECT_TREE_FAST_PATH",
                }
                for rid in selected
            ], profile

        heap: List[Tuple[Tuple[float, ...], int, Dict[str, Any]]] = []
        for rid in wy_ids:
            ref = records[rid]
            l3, l4, exact_atoms, bonus = l3_l4_similarity(query, ref)
            result = {
                "record_id": rid,
                "sim_l1": 1.0,
                "sim_l2": 1.0,
                "sim_l3": l3,
                "sim_l4": l4,
                "final_similarity": l4,
                "exact_element_site_match_atoms": exact_atoms,
                "l4_bonus_equivalent_atoms": bonus,
                "retrieval_stage": "L4_EXACT_WYCKOFF",
            }
            # L4 is the terminal score; L3 is the tie-breaker. Smaller record_id wins final ties.
            _heap_push_topk(heap, (l4, l3), rid, result, k)

        profile["scored_candidate_count"] = len(wy_ids)
        out = [item[2] for item in heap]
        out.sort(key=lambda r: (-r["sim_l4"], -r["sim_l3"], r["record_id"]))
        return out, profile

    # -------------------------------------------------------------------------
    # No exact Wyckoff occupancy -> score only L2.
    # Use element@Wyckoff postings to avoid full same-SPG pairwise scoring.
    # -------------------------------------------------------------------------
    profile["retrieval_stage"] = "L2_WYCKOFF_MISMATCH"
    candidate_set: set[int] = set()
    postings = spg_node["element_site_postings"]
    for pair_token in query["element_site_counts"].keys():
        candidate_set.update(postings.get(pair_token, []))
    if self_id is not None:
        candidate_set.discard(int(self_id))

    heap2: List[Tuple[Tuple[float, ...], int, Dict[str, Any]]] = []
    scored_ids: set[int] = set()
    for rid in candidate_set:
        ref = records[rid]
        l2, exact_atoms = l2_similarity(query, ref)
        scored_ids.add(rid)
        result = {
            "record_id": rid,
            "sim_l1": 1.0,
            "sim_l2": l2,
            "sim_l3": NULL,
            "sim_l4": NULL,
            "final_similarity": l2,
            "exact_element_site_match_atoms": exact_atoms,
            "l4_bonus_equivalent_atoms": NULL,
            "retrieval_stage": "L2_WYCKOFF_MISMATCH",
        }
        _heap_push_topk(heap2, (l2,), rid, result, k)

    profile["scored_candidate_count"] = len(candidate_set)
    out = [item[2] for item in heap2]
    out.sort(key=lambda r: (-r["sim_l2"], r["record_id"]))

    # If fewer than k candidates have positive element@site overlap, every remaining same-SPG
    # candidate has L2=0. Fill deterministically without scoring all of them.
    if len(out) < k:
        already = {r["record_id"] for r in out}
        for rid in spg_ids:
            if rid in already or rid in scored_ids:
                continue
            out.append(
                {
                    "record_id": rid,
                    "sim_l1": 1.0,
                    "sim_l2": 0.0,
                    "sim_l3": NULL,
                    "sim_l4": NULL,
                    "final_similarity": 0.0,
                    "exact_element_site_match_atoms": 0,
                    "l4_bonus_equivalent_atoms": NULL,
                    "retrieval_stage": "L2_WYCKOFF_MISMATCH_ZERO_FILL",
                }
            )
            if len(out) >= k:
                break

    return out[:k], profile
