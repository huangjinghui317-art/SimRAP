#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""CSV range, output formatting, merge, and profile aggregation blocks."""

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from .constants import NULL
from .utils import RunningStats, file_fingerprint, resolve_path, safe_scalar


def count_csv_records(path: Path, encoding: str) -> int:
    """Exact logical CSV row count; correctly handles quoted multiline CIF fields."""
    with open(path, "r", newline="", encoding=encoding) as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def build_query_layout(query_paths: Sequence[Path], encoding: str) -> Tuple[List[Dict[str, Any]], int]:
    layout: List[Dict[str, Any]] = []
    offset = 0
    for path in query_paths:
        n = count_csv_records(path, encoding)
        fp = file_fingerprint(path)
        layout.append(
            {
                "path": resolve_path(path),
                "rows": n,
                "global_start": offset,
                "global_end": offset + n,
                "size_bytes": fp["size_bytes"],
                "mtime_ns": fp["mtime_ns"],
            }
        )
        offset += n
    return layout, offset


def iter_query_rows(
    query_paths: Sequence[Path],
    layout: Sequence[Dict[str, Any]],
    start: int,
    end: int,
    cif_column: str,
    target_column: str,
    id_column: Optional[str],
    chunk_size: int,
    encoding: str,
) -> Iterator[Dict[str, Any]]:
    for path, info in zip(query_paths, layout):
        fstart = int(info["global_start"])
        fend = int(info["global_end"])
        overlap_start = max(start, fstart)
        overlap_end = min(end, fend)
        if overlap_start >= overlap_end:
            continue

        local_start = overlap_start - fstart
        local_end = overlap_end - fstart

        header = pd.read_csv(path, nrows=0, encoding=encoding)
        missing = [c for c in [cif_column, target_column] if c not in header.columns]
        if missing:
            raise KeyError(f"{path} missing columns: {missing}")
        actual_id_col = id_column if id_column and id_column in header.columns else None
        if actual_id_col is None and "material_id" in header.columns:
            actual_id_col = "material_id"

        usecols = [cif_column, target_column]
        if actual_id_col and actual_id_col not in usecols:
            usecols.append(actual_id_col)

        local_base = 0
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            chunksize=chunk_size,
            encoding=encoding,
        ):
            chunk_start = local_base
            chunk_end = local_base + len(chunk)
            local_base = chunk_end
            if chunk_end <= local_start:
                continue
            if chunk_start >= local_end:
                break

            s = max(0, local_start - chunk_start)
            e = min(len(chunk), local_end - chunk_start)
            sub = chunk.iloc[s:e]
            for offset_in_sub, (_, row) in enumerate(sub.iterrows()):
                local_row = chunk_start + s + offset_in_sub
                global_index = fstart + local_row
                yield {
                    "global_index": global_index,
                    "query_file": resolve_path(path),
                    "query_row": int(local_row),
                    "query_cif": safe_scalar(row[cif_column]),
                    "query_target_value": safe_scalar(row[target_column]),
                    "query_material_id": safe_scalar(row[actual_id_col]) if actual_id_col else None,
                }


OUTPUT_FIELDS = [
    "query_global_index",
    "query_file",
    "query_row",
    "query_material_id",
    "query_cif",
    "query_target_value",
    "query_formula",
    "query_spg_number",
    "query_spg_symbol",
    "query_wy_signature",
    "neighbor_rank",
    "neighbor_record_id",
    "neighbor_train_row",
    "neighbor_material_id",
    "neighbor_cif",
    "neighbor_target_value",
    "neighbor_formula",
    "neighbor_spg_number",
    "neighbor_spg_symbol",
    "neighbor_wy_signature",
    "sim_level1",
    "sim_level2",
    "sim_level3",
    "sim_level4",
    "final_similarity",
    "exact_element_site_match_atoms",
    "l4_bonus_equivalent_atoms",
    "retrieval_stage",
    "status",
    "error",
]


PROFILE_FIELDS = [
    "query_global_index",
    "query_file",
    "query_row",
    "status",
    "elapsed_sec",
    "descriptor_reused_from_train_index",
    "spg_candidate_count",
    "wy_candidate_count",
    "scored_candidate_count",
    "neighbor_count",
    "retrieval_stage",
    "error",
]


def placeholder_output_row(query_row: Dict[str, Any], desc: Optional[Dict[str, Any]], status: str, error: str = "", sim_l1: Any = NULL) -> Dict[str, Any]:
    return {
        "query_global_index": query_row["global_index"],
        "query_file": query_row["query_file"],
        "query_row": query_row["query_row"],
        "query_material_id": query_row["query_material_id"],
        "query_cif": query_row["query_cif"],
        "query_target_value": query_row["query_target_value"],
        "query_formula": desc.get("formula") if desc else None,
        "query_spg_number": desc.get("spg_number") if desc else None,
        "query_spg_symbol": desc.get("spg_symbol") if desc else None,
        "query_wy_signature": desc.get("wy_signature") if desc else None,
        "neighbor_rank": None,
        "neighbor_record_id": None,
        "neighbor_train_row": None,
        "neighbor_material_id": None,
        "neighbor_cif": None,
        "neighbor_target_value": None,
        "neighbor_formula": None,
        "neighbor_spg_number": None,
        "neighbor_spg_symbol": None,
        "neighbor_wy_signature": None,
        "sim_level1": sim_l1,
        "sim_level2": None,
        "sim_level3": None,
        "sim_level4": None,
        "final_similarity": sim_l1,
        "exact_element_site_match_atoms": None,
        "l4_bonus_equivalent_atoms": None,
        "retrieval_stage": status,
        "status": status,
        "error": error,
    }


def neighbor_output_rows(
    query_row: Dict[str, Any],
    query_desc: Dict[str, Any],
    neighbors: Sequence[Dict[str, Any]],
    index_obj: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rank, result in enumerate(neighbors, start=1):
        ref = index_obj["records"][result["record_id"]]
        out.append(
            {
                "query_global_index": query_row["global_index"],
                "query_file": query_row["query_file"],
                "query_row": query_row["query_row"],
                "query_material_id": query_row["query_material_id"],
                "query_cif": query_row["query_cif"],
                "query_target_value": query_row["query_target_value"],
                "query_formula": query_desc["formula"],
                "query_spg_number": query_desc["spg_number"],
                "query_spg_symbol": query_desc["spg_symbol"],
                "query_wy_signature": query_desc["wy_signature"],
                "neighbor_rank": rank,
                "neighbor_record_id": result["record_id"],
                "neighbor_train_row": ref["train_row"],
                "neighbor_material_id": ref.get("material_id"),
                "neighbor_cif": ref["cif"],
                "neighbor_target_value": ref["target_value"],
                "neighbor_formula": ref["formula"],
                "neighbor_spg_number": ref["spg_number"],
                "neighbor_spg_symbol": ref["spg_symbol"],
                "neighbor_wy_signature": ref["wy_signature"],
                "sim_level1": result["sim_l1"],
                "sim_level2": result["sim_l2"],
                "sim_level3": result["sim_l3"],
                "sim_level4": result["sim_l4"],
                "final_similarity": result["final_similarity"],
                "exact_element_site_match_atoms": result["exact_element_site_match_atoms"],
                "l4_bonus_equivalent_atoms": result["l4_bonus_equivalent_atoms"],
                "retrieval_stage": result["retrieval_stage"],
                "status": "ok",
                "error": "",
            }
        )
    return out


def write_csv_atomic(rows: Sequence[Dict[str, Any]], fields: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def merge_csv_parts(part_paths: Sequence[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as out_f:
        wrote_header = False
        writer = None
        fieldnames = None
        for part in part_paths:
            with open(part, "r", newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                if not wrote_header:
                    fieldnames = reader.fieldnames or []
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                    wrote_header = True
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
        if not wrote_header:
            # Should not happen because each query produces at least one row, but keep valid CSV.
            csv.writer(out_f).writerow([])
    os.replace(tmp, output_path)


def parse_part_range(path: Path) -> Tuple[int, int]:
    # part_000000000000_000000000100.csv
    stem = path.stem
    _, s, e = stem.split("_")
    return int(s), int(e)


def aggregate_profile_stats(profile_path: Path) -> Dict[str, Any]:
    time_stats = RunningStats()
    statuses: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    neighbor_stats = RunningStats()
    spg_stats = RunningStats()
    wy_stats = RunningStats()
    scored_stats = RunningStats()
    total = 0
    if not profile_path.exists():
        return {"total_queries": 0}
    with open(profile_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            statuses[row.get("status", "")] += 1
            stages[row.get("retrieval_stage", "")] += 1
            for key, stats_obj in [
                ("elapsed_sec", time_stats),
                ("neighbor_count", neighbor_stats),
                ("spg_candidate_count", spg_stats),
                ("wy_candidate_count", wy_stats),
                ("scored_candidate_count", scored_stats),
            ]:
                val = row.get(key, "")
                if val not in ("", None):
                    try:
                        stats_obj.update(float(val))
                    except Exception:
                        pass
    return {
        "total_queries": total,
        "status_counts": dict(statuses),
        "retrieval_stage_counts": dict(stages),
        "per_query_time_sec": time_stats.to_dict(),
        "neighbors_per_query": neighbor_stats.to_dict(),
        "spg_candidate_count": spg_stats.to_dict(),
        "wy_candidate_count": wy_stats.to_dict(),
        "scored_candidate_count": scored_stats.to_dict(),
    }


def aggregate_resource_samples(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"sample_count": 0}
    cols = [
        "process_cpu_percent_raw",
        "process_cpu_percent_machine_normalized",
        "system_cpu_percent",
        "rss_mb",
    ]
    stats = {c: RunningStats() for c in cols}
    sessions = set()
    sample_count = 0
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_count += 1
            sessions.add(row.get("session_id", ""))
            for c in cols:
                try:
                    stats[c].update(float(row[c]))
                except Exception:
                    pass
    return {
        "sample_count": sample_count,
        "session_count": len([s for s in sessions if s]),
        **{c: stats[c].to_dict() for c in cols},
    }
