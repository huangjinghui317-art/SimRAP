#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Trie and multi-way tree blocks moved verbatim from the original script."""

from typing import Any, Dict, Optional, Sequence


def new_trie_root() -> Dict[str, Any]:
    return {"_children": {}}


def trie_insert(root: Dict[str, Any], tokens: Sequence[str], record_id: int) -> Dict[str, Any]:
    node = root
    for token in tokens:
        children = node.setdefault("_children", {})
        node = children.setdefault(token, {"_children": {}})
    node.setdefault("_ids", []).append(int(record_id))
    return node


def trie_lookup(root: Optional[Dict[str, Any]], tokens: Sequence[str]) -> Optional[Dict[str, Any]]:
    if root is None:
        return None
    node = root
    for token in tokens:
        node = node.get("_children", {}).get(token)
        if node is None:
            return None
    return node


def new_spg_node() -> Dict[str, Any]:
    return {
        "ids": [],
        "wy_root": new_trie_root(),
        # Auxiliary inverted posting list for exact element@Wyckoff overlap.
        "element_site_postings": {},
    }


def insert_record_into_tree(tree: Dict[int, Dict[str, Any]], record: Dict[str, Any]) -> None:
    rid = int(record["record_id"])
    spg = int(record["spg_number"])
    node = tree.setdefault(spg, new_spg_node())
    node["ids"].append(rid)

    # Inverted postings used for exact top-k L2 fallback without scoring the whole SPG bucket.
    postings = node["element_site_postings"]
    for pair_token in record["element_site_counts"].keys():
        postings.setdefault(pair_token, []).append(rid)

    # Main hierarchy:
    # SPG -> Wyckoff trie -> family@Wyckoff trie -> element@Wyckoff trie.
    wy_leaf = trie_insert(node["wy_root"], record["wy_tokens"], rid)
    group_root = wy_leaf.setdefault("_group_root", new_trie_root())
    group_leaf = trie_insert(group_root, record["group_tokens"], rid)
    element_root = group_leaf.setdefault("_element_root", new_trie_root())
    trie_insert(element_root, record["element_site_tokens"], rid)
