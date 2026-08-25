#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Periodic-table helper blocks moved verbatim from the original script."""

from functools import lru_cache
from typing import Any, Dict

from pymatgen.core import Element


@lru_cache(maxsize=None)
def element_info(symbol: str) -> Dict[str, Any]:
    el = Element(symbol)
    if el.is_lanthanoid:
        family = "LN"
    elif el.is_actinoid:
        family = "AC"
    elif symbol == "H":
        family = "H"
    else:
        family = f"G{el.group}" if el.group is not None else "GNA"
    return {
        "symbol": symbol,
        "Z": int(el.Z),
        "group": int(el.group) if el.group is not None else None,
        "row": int(el.row),
        "is_lanthanoid": bool(el.is_lanthanoid),
        "is_actinoid": bool(el.is_actinoid),
        "family": family,
    }


def family_token(symbol: str) -> str:
    return str(element_info(symbol)["family"])


def substitution_weight(symbol_a: str, symbol_b: str) -> float:
    """Weight used only for non-identical elements in L4."""
    if symbol_a == symbol_b:
        return 1.0

    a = element_info(symbol_a)
    b = element_info(symbol_b)

    # Same lanthanoid series.
    if a["is_lanthanoid"] and b["is_lanthanoid"]:
        return max(0.0, 1.0 - abs(a["Z"] - b["Z"]) / 14.0)

    # Same actinoid series.
    if a["is_actinoid"] and b["is_actinoid"]:
        return max(0.0, 1.0 - abs(a["Z"] - b["Z"]) / 14.0)

    # H is explicitly excluded from normal same-group substitution.
    if symbol_a == "H" or symbol_b == "H":
        return 0.0

    # Lanthanoids/actinoids must NOT accidentally match ordinary group-3 elements.
    if (
        a["is_lanthanoid"]
        or a["is_actinoid"]
        or b["is_lanthanoid"]
        or b["is_actinoid"]
    ):
        return 0.0

    if a["group"] is not None and a["group"] == b["group"]:
        return max(0.0, 1.0 - abs(a["row"] - b["row"]) / 4.0)

    return 0.0
