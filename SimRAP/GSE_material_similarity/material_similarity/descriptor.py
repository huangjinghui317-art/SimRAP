#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Crystal structure descriptor construction blocks."""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .chemistry import element_info, family_token


def _get_dataset_attr(dataset: Any, name: str) -> Any:
    val = getattr(dataset, name, None)
    if val is not None:
        return val
    try:
        return dataset[name]
    except Exception as exc:
        raise KeyError(f"symmetry dataset missing field: {name}") from exc


def _tokens_from_counter(counter: Dict[str, int], sep: str = "#") -> List[str]:
    return [f"{k}{sep}{int(counter[k])}" for k in sorted(counter)]


def build_descriptor(
    cif_text: str,
    symprec: float,
    angle_tolerance: float,
    standardize: str,
) -> Dict[str, Any]:
    if cif_text is None or (isinstance(cif_text, float) and math.isnan(cif_text)):
        raise ValueError("empty CIF")
    cif_text = str(cif_text).strip()
    if not cif_text:
        raise ValueError("empty CIF")

    structure = Structure.from_str(cif_text, fmt="cif")
    input_num_atoms = int(len(structure))

    if not structure.is_ordered:
        raise ValueError(
            "disordered structure detected; this script requires ordered structures "
            "because the requested similarity is defined by discrete atom counts"
        )

    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    if standardize == "refined":
        structure = analyzer.get_refined_structure()
        if not structure.is_ordered:
            raise ValueError("refined structure became disordered")
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
    elif standardize != "none":
        raise ValueError(f"unsupported standardize mode: {standardize}")

    dataset = analyzer.get_symmetry_dataset()
    spg_number = int(_get_dataset_attr(dataset, "number"))
    try:
        spg_symbol = str(_get_dataset_attr(dataset, "international"))
    except Exception:
        spg_symbol = str(analyzer.get_space_group_symbol())

    wyckoffs = [str(x) for x in list(_get_dataset_attr(dataset, "wyckoffs"))]
    if len(wyckoffs) != len(structure):
        raise RuntimeError(
            f"wyckoff count mismatch: {len(wyckoffs)} != {len(structure)}"
        )

    wy_counts: Counter[str] = Counter()
    element_counts: Counter[str] = Counter()
    element_site_counts: Counter[str] = Counter()
    group_site_counts: Counter[str] = Counter()
    site_elements: Dict[str, Counter[str]] = defaultdict(Counter)

    for site, wy in zip(structure.sites, wyckoffs):
        if not site.is_ordered:
            raise ValueError("disordered site detected")
        symbol = str(site.specie.symbol)
        # Validate element and warm cache.
        _ = element_info(symbol)

        wy_counts[wy] += 1
        element_counts[symbol] += 1
        element_site_counts[f"{symbol}@{wy}"] += 1
        group_site_counts[f"{family_token(symbol)}@{wy}"] += 1
        site_elements[wy][symbol] += 1

    wy_tokens = _tokens_from_counter(dict(wy_counts))
    group_tokens = _tokens_from_counter(dict(group_site_counts))
    element_site_tokens = _tokens_from_counter(dict(element_site_counts))

    wy_signature = "|".join(wy_tokens)
    group_signature = "|".join(group_tokens)
    element_site_signature = "|".join(element_site_tokens)

    return {
        "spg_number": spg_number,
        "spg_symbol": spg_symbol,
        "input_num_atoms": input_num_atoms,
        "num_atoms": int(len(structure)),
        "formula": str(structure.composition.reduced_formula),
        "wy_counts": dict(sorted(wy_counts.items())),
        "element_counts": dict(sorted(element_counts.items())),
        "element_site_counts": dict(sorted(element_site_counts.items())),
        "group_site_counts": dict(sorted(group_site_counts.items())),
        "site_elements": {
            wy: dict(sorted(cnt.items())) for wy, cnt in sorted(site_elements.items())
        },
        "wy_tokens": wy_tokens,
        "group_tokens": group_tokens,
        "element_site_tokens": element_site_tokens,
        "wy_signature": wy_signature,
        "group_signature": group_signature,
        "element_site_signature": element_site_signature,
    }
