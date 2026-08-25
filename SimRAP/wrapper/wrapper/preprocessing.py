#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""CSV rank discovery and wrapper-input preprocessing."""

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


RANK_CIF_RE = re.compile(
    r"^rank(\d+)_neighbor_cif$"
)


def discover_ranks(columns: Sequence[str]) -> List[int]:
    ranks = []

    for col in columns:
        m = RANK_CIF_RE.match(str(col))
        if m:
            ranks.append(int(m.group(1)))

    return sorted(set(ranks))


def finite_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        x = float(value)
    except Exception:
        return default

    if not math.isfinite(x):
        return default

    return x


def nonempty_text(value: Any) -> Optional[str]:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    s = str(value).strip()

    return s if s else None


@dataclass
class RowRecord:
    source_row_idx: int
    query_cif: str
    target: float
    backbone_output: float

    # Each item:
    # {
    #   "rank": int,
    #   "cif": str,
    #   "target": float,
    #   "similarity": float,
    # }
    neighbors: List[Dict[str, Any]]


def dataframe_to_records(
    df: pd.DataFrame,
    target_col: str,
    query_cif_col: str,
    backbone_output_col: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[RowRecord], Dict[str, Any]]:

    required = [
        target_col,
        query_cif_col,
        backbone_output_col,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"CSV missing required columns: {missing}"
        )

    ranks = discover_ranks(df.columns)

    if not ranks:
        raise KeyError(
            "No rankN_neighbor_cif columns found."
        )

    for rank in ranks:
        required_rank_cols = [
            f"rank{rank}_neighbor_cif",
            f"rank{rank}_neighbor_target_value",
            f"rank{rank}_final_similarity",
        ]

        absent = [
            c
            for c in required_rank_cols
            if c not in df.columns
        ]

        if absent:
            raise KeyError(
                f"Rank {rank} is incomplete. Missing: {absent}"
            )

    records: List[RowRecord] = []

    skipped_target = 0
    skipped_query_cif = 0
    skipped_backbone_output = 0

    total_neighbors = 0
    neighbor_target_imputed_zero = 0
    neighbor_similarity_imputed_zero = 0

    for row_idx, row in df.iterrows():

        target = finite_float(
            row[target_col],
            default=None,
        )

        if target is None:
            skipped_target += 1
            continue

        query_cif = nonempty_text(
            row[query_cif_col]
        )

        if query_cif is None:
            skipped_query_cif += 1
            continue

        backbone_output = finite_float(
            row[backbone_output_col],
            default=None,
        )

        if backbone_output is None:
            skipped_backbone_output += 1
            continue

        neighbors = []

        for rank in ranks:
            cif_col = (
                f"rank{rank}_neighbor_cif"
            )

            n_cif = nonempty_text(
                row[cif_col]
            )

            # A neighbor without a CIF cannot be converted to a graph, so this rank is skipped.
            if n_cif is None:
                continue

            target_raw = row[
                f"rank{rank}_neighbor_target_value"
            ]

            n_target = finite_float(
                target_raw,
                default=None,
            )

            if n_target is None:
                # Missing neighbor targets are imputed as 0.0.
                n_target = 0.0
                neighbor_target_imputed_zero += 1

            sim_raw = row[
                f"rank{rank}_final_similarity"
            ]

            n_similarity = finite_float(
                sim_raw,
                default=None,
            )

            if n_similarity is None:
                n_similarity = 0.0
                neighbor_similarity_imputed_zero += 1

            neighbors.append(
                {
                    "rank": int(rank),
                    "cif": n_cif,
                    "target": float(n_target),
                    "similarity": float(n_similarity),
                }
            )

        neighbors.sort(
            key=lambda x: x["rank"]
        )

        total_neighbors += len(neighbors)

        records.append(
            RowRecord(
                source_row_idx=int(row_idx),
                query_cif=query_cif,
                target=float(target),
                backbone_output=float(backbone_output),
                neighbors=neighbors,
            )
        )

    stats = {
        "input_rows": int(len(df)),
        "usable_rows": int(len(records)),
        "skipped_target": int(skipped_target),
        "skipped_query_cif": int(skipped_query_cif),
        "skipped_backbone_output": int(skipped_backbone_output),
        "ranks": ranks,
        "max_rank": int(max(ranks)),
        "total_neighbors": int(total_neighbors),
        "neighbor_target_imputed_zero":
            int(neighbor_target_imputed_zero),
        "neighbor_similarity_imputed_zero":
            int(neighbor_similarity_imputed_zero),
    }

    if logger:
        logger.info(
            "CSV preprocessing: %s",
            json.dumps(
                stats,
                ensure_ascii=False,
            ),
        )

    if not records:
        raise ValueError(
            "No usable rows remain after preprocessing."
        )

    return records, stats
