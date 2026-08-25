#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Step-1 index construction block moved verbatim from the original script."""

import argparse
import csv
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import psutil
from tqdm import tqdm

from .constants import SCHEMA_VERSION
from .descriptor import build_descriptor
from .query_io import count_csv_records
from .tree import insert_record_into_tree
from .utils import (
    ResourceMonitor,
    RunningStats,
    atomic_json_dump,
    atomic_pickle_dump,
    file_fingerprint,
    now_iso,
    resolve_path,
    safe_scalar,
    setup_logger,
)


BUILD_PROFILE_FIELDS = [
    "train_row",
    "status",
    "elapsed_sec",
    "spg_number",
    "spg_symbol",
    "input_num_atoms",
    "num_atoms",
    "formula",
    "wy_signature",
    "group_signature",
    "element_site_signature",
    "rss_mb_after",
    "error",
]


def build_index(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train_csv).expanduser().resolve()
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    index_path = Path(args.index_file).expanduser().resolve() if args.index_file else output_dir / "similarity_index.pkl"
    log_path = output_dir / "build_index.log"
    logger = setup_logger("build_index", log_path, append=False)

    logger.info("Step 1 started")
    logger.info("train_csv=%s", train_path)
    logger.info("index_file=%s", index_path)
    logger.info("cif_column=%s target_column=%s", args.cif_column, args.target_column)
    logger.info(
        "symprec=%s angle_tolerance=%s standardize=%s",
        args.symprec,
        args.angle_tolerance,
        args.standardize,
    )

    header = pd.read_csv(train_path, nrows=0, encoding=args.encoding)
    required = [args.cif_column, args.target_column]
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise KeyError(f"train CSV missing columns: {missing}")

    usecols = [args.cif_column, args.target_column]
    id_col = args.id_column if args.id_column and args.id_column in header.columns else None
    if id_col and id_col not in usecols:
        usecols.append(id_col)

    total_rows = count_csv_records(train_path, args.encoding)
    logger.info("train_rows=%d", total_rows)

    monitor = ResourceMonitor(args.resource_sample_interval, session_id="build")
    monitor.start()
    wall_start = time.perf_counter()

    tree: Dict[int, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    train_row_to_id: Dict[int, int] = {}
    failed_rows: Dict[int, str] = {}

    time_stats = RunningStats()
    atom_stats = RunningStats()
    spg_counter: Counter[int] = Counter()
    wy_counter: Counter[str] = Counter()
    group_counter: Counter[str] = Counter()
    element_site_counter: Counter[str] = Counter()
    full_signature_counter: Counter[str] = Counter()

    profile_path = output_dir / "build_profile.csv"
    error_path = output_dir / "build_errors.csv"
    signature_counts_path = output_dir / "index_signature_counts.csv"

    with open(profile_path, "w", newline="", encoding="utf-8") as pf, open(
        error_path, "w", newline="", encoding="utf-8"
    ) as ef:
        p_writer = csv.DictWriter(pf, fieldnames=BUILD_PROFILE_FIELDS)
        p_writer.writeheader()
        e_writer = csv.DictWriter(ef, fieldnames=["train_row", "error"])
        e_writer.writeheader()

        row_base = 0
        pbar = tqdm(total=total_rows, desc="Step 1 | build index", unit="material")
        for chunk in pd.read_csv(
            train_path,
            usecols=usecols,
            chunksize=args.csv_chunk_size,
            encoding=args.encoding,
        ):
            for local_i, (_, row) in enumerate(chunk.iterrows()):
                train_row = row_base + local_i
                t0 = time.perf_counter()
                profile: Dict[str, Any] = {
                    "train_row": train_row,
                    "status": "",
                    "elapsed_sec": "",
                    "spg_number": "",
                    "spg_symbol": "",
                    "input_num_atoms": "",
                    "num_atoms": "",
                    "formula": "",
                    "wy_signature": "",
                    "group_signature": "",
                    "element_site_signature": "",
                    "rss_mb_after": "",
                    "error": "",
                }
                try:
                    cif_text = safe_scalar(row[args.cif_column])
                    desc = build_descriptor(
                        cif_text,
                        symprec=args.symprec,
                        angle_tolerance=args.angle_tolerance,
                        standardize=args.standardize,
                    )
                    rid = len(records)
                    record = {
                        "record_id": rid,
                        "train_row": train_row,
                        "source_file": resolve_path(train_path),
                        "cif": cif_text,
                        "target_value": safe_scalar(row[args.target_column]),
                        "material_id": safe_scalar(row[id_col]) if id_col else None,
                        **desc,
                    }
                    records.append(record)
                    train_row_to_id[train_row] = rid
                    insert_record_into_tree(tree, record)

                    spg_counter[desc["spg_number"]] += 1
                    wy_counter[desc["wy_signature"]] += 1
                    group_counter[desc["group_signature"]] += 1
                    element_site_counter[desc["element_site_signature"]] += 1
                    full_sig = (
                        f"SPG={desc['spg_number']}||WY={desc['wy_signature']}||"
                        f"GRP={desc['group_signature']}||EL={desc['element_site_signature']}"
                    )
                    full_signature_counter[full_sig] += 1
                    atom_stats.update(desc["num_atoms"])

                    profile.update(
                        {
                            "status": "ok",
                            "spg_number": desc["spg_number"],
                            "spg_symbol": desc["spg_symbol"],
                            "input_num_atoms": desc["input_num_atoms"],
                            "num_atoms": desc["num_atoms"],
                            "formula": desc["formula"],
                            "wy_signature": desc["wy_signature"],
                            "group_signature": desc["group_signature"],
                            "element_site_signature": desc["element_site_signature"],
                        }
                    )
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    failed_rows[train_row] = msg
                    profile["status"] = "failed"
                    profile["error"] = msg
                    e_writer.writerow({"train_row": train_row, "error": msg})
                    logger.warning("index parse failed | row=%d | %s", train_row, msg)

                elapsed = time.perf_counter() - t0
                time_stats.update(elapsed)
                profile["elapsed_sec"] = f"{elapsed:.9f}"
                try:
                    profile["rss_mb_after"] = f"{psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2):.3f}"
                except Exception:
                    pass
                p_writer.writerow(profile)
                pbar.update(1)

                if (train_row + 1) % args.log_every == 0:
                    pf.flush()
                    ef.flush()
                    logger.info(
                        "build progress row=%d/%d success=%d failed=%d avg_sec=%.6f",
                        train_row + 1,
                        total_rows,
                        len(records),
                        len(failed_rows),
                        time_stats.mean,
                    )

            row_base += len(chunk)
        pbar.close()

    wall_elapsed = time.perf_counter() - wall_start
    monitor.stop()
    monitor.save_csv(output_dir / "build_cpu_samples.csv", append=False)

    index_meta = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "train_fingerprint": file_fingerprint(train_path),
        "cif_column": args.cif_column,
        "target_column": args.target_column,
        "id_column": id_col,
        "symprec": args.symprec,
        "angle_tolerance": args.angle_tolerance,
        "standardize": args.standardize,
        "total_train_rows": total_rows,
        "indexed_records": len(records),
        "failed_rows": len(failed_rows),
        "tree_layout": "SPG -> Wyckoff occupancy trie -> family@Wyckoff trie -> element@Wyckoff trie",
        "wyckoff_signature_definition": "multiset of per-atom Wyckoff letters in the analyzed cell",
        "family_definition": "LN and AC are special families; H is separate; otherwise IUPAC group G1..G18",
    }
    index_obj = {
        "meta": index_meta,
        "tree": tree,
        "records": records,
        "train_row_to_id": train_row_to_id,
        "failed_rows": failed_rows,
    }

    logger.info("serializing index ...")
    atomic_pickle_dump(index_obj, index_path)

    # Signature-count table gives a concrete view of sample diversity / duplicates.
    with open(signature_counts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["full_signature", "sample_count"])
        for sig, cnt in full_signature_counter.most_common():
            writer.writerow([sig, cnt])

    leaf_sizes = list(full_signature_counter.values())
    duplicate_samples = sum(cnt for cnt in leaf_sizes if cnt > 1)
    duplicate_excess = sum(cnt - 1 for cnt in leaf_sizes if cnt > 1)
    singleton_signatures = sum(1 for cnt in leaf_sizes if cnt == 1)

    stats = {
        "step": 1,
        "created_at": now_iso(),
        "train_csv": resolve_path(train_path),
        "index_file": str(index_path),
        "wall_time_sec": wall_elapsed,
        "per_sample_build_time_sec": time_stats.to_dict(),
        "resource_usage": monitor.summary(),
        "counts": {
            "total_rows": total_rows,
            "success": len(records),
            "failed": len(failed_rows),
        },
        "sample_diversity": {
            "unique_space_groups": len(spg_counter),
            "unique_wyckoff_signatures_global": len(wy_counter),
            "unique_family_signatures_global": len(group_counter),
            "unique_element_site_signatures_global": len(element_site_counter),
            "unique_full_index_signatures": len(full_signature_counter),
            "singleton_full_signatures": singleton_signatures,
            "samples_in_non_singleton_full_signatures": duplicate_samples,
            "duplicate_excess_samples": duplicate_excess,
            "full_signature_leaf_size_mean": statistics.fmean(leaf_sizes) if leaf_sizes else None,
            "full_signature_leaf_size_max": max(leaf_sizes) if leaf_sizes else None,
            "standardized_atom_count": atom_stats.to_dict(),
            "top_20_space_groups": [
                {"spg": int(spg), "count": int(cnt)} for spg, cnt in spg_counter.most_common(20)
            ],
        },
        "index_size_bytes": index_path.stat().st_size,
    }
    atomic_json_dump(stats, output_dir / "build_stats.json")

    logger.info("Step 1 finished")
    logger.info("wall_time_sec=%.3f", wall_elapsed)
    logger.info("avg_build_sec_per_sample=%.6f", time_stats.mean if time_stats.count else float("nan"))
    logger.info("success=%d failed=%d", len(records), len(failed_rows))
    logger.info("index_size_mb=%.2f", index_path.stat().st_size / (1024 ** 2))
    logger.info("resource_summary=%s", json.dumps(monitor.summary(), ensure_ascii=False))
    logger.info("index=%s", index_path)
    return index_path
