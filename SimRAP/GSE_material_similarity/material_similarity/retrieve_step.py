#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Step-2 retrieval pipeline blocks moved verbatim from the original script."""

import argparse
import json
import pickle
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .constants import SCHEMA_VERSION

from .descriptor import build_descriptor
from .query_io import (
    OUTPUT_FIELDS,
    PROFILE_FIELDS,
    aggregate_profile_stats,
    aggregate_resource_samples,
    build_query_layout,
    iter_query_rows,
    merge_csv_parts,
    neighbor_output_rows,
    parse_part_range,
    placeholder_output_row,
    write_csv_atomic,
)
from .retrieval_core import retrieve_neighbors
from .utils import (
    ResourceMonitor,
    atomic_json_dump,
    file_fingerprint,
    now_iso,
    setup_logger,
    short_hash,
)


def load_index(index_path: Path) -> Dict[str, Any]:
    with open(index_path, "rb") as f:
        obj = pickle.load(f)
    if obj.get("meta", {}).get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"index schema mismatch: {obj.get('meta', {}).get('schema_version')} != {SCHEMA_VERSION}"
        )
    return obj


def retrieve_step(args: argparse.Namespace, index_path: Optional[Path] = None) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if index_path is None:
        index_path = Path(args.index_file).expanduser().resolve() if args.index_file else output_dir / "similarity_index.pkl"
    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")
    if not args.query_csvs:
        raise ValueError("Step 2 requires --query-csvs FILE1 FILE2 ...")

    query_paths = [Path(p).expanduser().resolve() for p in args.query_csvs]
    for p in query_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    index_obj = load_index(index_path)
    meta = index_obj["meta"]
    index_train_path = str(meta["train_fingerprint"]["path"])

    # Descriptor settings are taken from the index so query descriptors are guaranteed comparable.
    symprec = float(meta["symprec"])
    angle_tolerance = float(meta["angle_tolerance"])
    standardize = str(meta["standardize"])

    layout, total_queries = build_query_layout(query_paths, args.encoding)
    start = max(0, int(args.start_index))
    end = total_queries if args.end_index is None else min(int(args.end_index), total_queries)
    if start >= end:
        raise ValueError(f"empty range [{start}, {end}) for total_queries={total_queries}")

    tag = f"{start:012d}_{end:012d}"
    final_output = Path(args.output_csv).expanduser().resolve() if args.output_csv else output_dir / f"neighbors_{tag}.csv"
    final_profile = output_dir / f"retrieval_profile_{tag}.csv"
    checkpoint_path = output_dir / f"retrieval_checkpoint_{tag}.json"
    parts_dir = output_dir / f"retrieval_parts_{tag}"
    profile_parts_dir = output_dir / f"retrieval_profile_parts_{tag}"
    cpu_samples_path = output_dir / f"retrieval_cpu_samples_{tag}.csv"
    stats_path = output_dir / f"retrieval_stats_{tag}.json"
    log_path = output_dir / f"retrieve_{tag}.log"

    config_for_hash = {
        "schema_version": SCHEMA_VERSION,
        "index_fingerprint": file_fingerprint(index_path),
        "query_layout": layout,
        "start": start,
        "end": end,
        "k": int(args.k),
        "cif_column": args.cif_column,
        "target_column": args.target_column,
        "id_column": args.id_column,
    }
    config_hash = short_hash(config_for_hash)

    if args.overwrite:
        for p in [final_output, final_profile, checkpoint_path, cpu_samples_path, stats_path]:
            if p.exists():
                p.unlink()
        for d in [parts_dir, profile_parts_dir]:
            if d.exists():
                shutil.rmtree(d)

    existing_artifacts = (
        checkpoint_path.exists() or parts_dir.exists() or profile_parts_dir.exists()
    )
    if existing_artifacts and not args.resume and not args.overwrite:
        raise RuntimeError(
            "retrieval checkpoint/parts already exist. Use --resume to continue or --overwrite to restart."
        )

    next_index = start
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        if ckpt.get("config_hash") != config_hash:
            raise RuntimeError(
                "checkpoint configuration does not match current command/index/query files; refusing unsafe resume"
            )
        next_index = max(start, int(ckpt.get("next_global_index", start)))

    parts_dir.mkdir(parents=True, exist_ok=True)
    profile_parts_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("retrieve", log_path, append=bool(args.resume))
    logger.info("Step 2 started/resumed")
    logger.info("index=%s", index_path)
    logger.info("query_files=%s", [str(p) for p in query_paths])
    logger.info("query_layout=%s", json.dumps(layout, ensure_ascii=False))
    logger.info("global_range=[%d,%d) next_index=%d total_queries=%d", start, end, next_index, total_queries)
    logger.info("k=%d", args.k)
    logger.info("descriptor settings from index: symprec=%s angle_tolerance=%s standardize=%s", symprec, angle_tolerance, standardize)

    if next_index >= end:
        logger.info("checkpoint already reached range end; merging existing parts")
    else:
        session_id = uuid.uuid4().hex[:12]
        monitor = ResourceMonitor(args.resource_sample_interval, session_id=session_id)
        monitor.start()
        session_wall_start = time.perf_counter()

        out_batch: List[Dict[str, Any]] = []
        prof_batch: List[Dict[str, Any]] = []
        batch_start = next_index
        processed_since_flush = 0

        pbar = tqdm(
            total=end - next_index,
            desc=f"Step 2 | retrieve [{next_index},{end})",
            unit="material",
        )

        def flush_batch(batch_end_exclusive: int) -> None:
            nonlocal out_batch, prof_batch, batch_start, processed_since_flush
            if not prof_batch:
                return
            out_part = parts_dir / f"part_{batch_start:012d}_{batch_end_exclusive:012d}.csv"
            prof_part = profile_parts_dir / f"part_{batch_start:012d}_{batch_end_exclusive:012d}.csv"
            write_csv_atomic(out_batch, OUTPUT_FIELDS, out_part)
            write_csv_atomic(prof_batch, PROFILE_FIELDS, prof_part)
            atomic_json_dump(
                {
                    "config_hash": config_hash,
                    "next_global_index": batch_end_exclusive,
                    "range_start": start,
                    "range_end": end,
                    "updated_at": now_iso(),
                    "last_output_part": str(out_part),
                    "last_profile_part": str(prof_part),
                },
                checkpoint_path,
            )
            logger.info(
                "checkpoint saved next_global_index=%d batch_queries=%d output_rows=%d",
                batch_end_exclusive,
                len(prof_batch),
                len(out_batch),
            )
            out_batch = []
            prof_batch = []
            batch_start = batch_end_exclusive
            processed_since_flush = 0

        try:
            for qrow in iter_query_rows(
                query_paths=query_paths,
                layout=layout,
                start=next_index,
                end=end,
                cif_column=args.cif_column,
                target_column=args.target_column,
                id_column=args.id_column,
                chunk_size=args.csv_chunk_size,
                encoding=args.encoding,
            ):
                q0 = time.perf_counter()
                gidx = int(qrow["global_index"])
                reused = False
                status = "ok"
                error = ""
                profile_info = {
                    "spg_candidate_count": 0,
                    "wy_candidate_count": 0,
                    "scored_candidate_count": 0,
                    "retrieval_stage": "",
                }
                neighbor_count = 0

                try:
                    self_id: Optional[int] = None
                    if qrow["query_file"] == index_train_path:
                        self_id = index_obj["train_row_to_id"].get(int(qrow["query_row"]))

                    # Reuse descriptor for train.csv itself; no need to parse CIF again.
                    if self_id is not None:
                        query_desc = index_obj["records"][self_id]
                        reused = True
                    else:
                        query_desc = build_descriptor(
                            qrow["query_cif"],
                            symprec=symprec,
                            angle_tolerance=angle_tolerance,
                            standardize=standardize,
                        )

                    neighbors, profile_info = retrieve_neighbors(
                        query_desc,
                        index_obj,
                        k=int(args.k),
                        self_id=self_id,
                    )
                    neighbor_count = len(neighbors)
                    if neighbors:
                        out_batch.extend(
                            neighbor_output_rows(qrow, query_desc, neighbors, index_obj)
                        )
                    else:
                        # Exact SPG does not exist after self-exclusion -> L1=0, later NULL.
                        out_batch.append(
                            placeholder_output_row(
                                qrow,
                                query_desc,
                                status=profile_info["retrieval_stage"],
                                sim_l1=0.0,
                            )
                        )
                        status = profile_info["retrieval_stage"]
                except Exception as exc:
                    status = "QUERY_PARSE_OR_RETRIEVAL_FAILED"
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "query failed | global=%d file=%s row=%d | %s",
                        gidx,
                        qrow["query_file"],
                        qrow["query_row"],
                        error,
                    )
                    out_batch.append(
                        placeholder_output_row(qrow, None, status=status, error=error, sim_l1=None)
                    )

                q_elapsed = time.perf_counter() - q0
                prof_batch.append(
                    {
                        "query_global_index": gidx,
                        "query_file": qrow["query_file"],
                        "query_row": qrow["query_row"],
                        "status": status,
                        "elapsed_sec": f"{q_elapsed:.9f}",
                        "descriptor_reused_from_train_index": int(reused),
                        "spg_candidate_count": profile_info.get("spg_candidate_count", 0),
                        "wy_candidate_count": profile_info.get("wy_candidate_count", 0),
                        "scored_candidate_count": profile_info.get("scored_candidate_count", 0),
                        "neighbor_count": neighbor_count,
                        "retrieval_stage": profile_info.get("retrieval_stage", status),
                        "error": error,
                    }
                )
                processed_since_flush += 1
                pbar.update(1)

                if processed_since_flush >= args.checkpoint_every:
                    flush_batch(gidx + 1)

            if prof_batch:
                last_gidx = int(prof_batch[-1]["query_global_index"])
                flush_batch(last_gidx + 1)
        finally:
            pbar.close()
            session_wall_elapsed = time.perf_counter() - session_wall_start
            monitor.stop()
            monitor.save_csv(cpu_samples_path, append=cpu_samples_path.exists())
            logger.info("retrieval session wall_time_sec=%.3f", session_wall_elapsed)
            logger.info("retrieval session resource_summary=%s", json.dumps(monitor.summary(), ensure_ascii=False))

    # Merge atomic parts into final CSVs.
    out_parts = sorted(parts_dir.glob("part_*.csv"), key=parse_part_range)
    prof_parts = sorted(profile_parts_dir.glob("part_*.csv"), key=parse_part_range)
    if not out_parts or not prof_parts:
        raise RuntimeError("no retrieval parts found; cannot build final output")
    merge_csv_parts(out_parts, final_output)
    merge_csv_parts(prof_parts, final_profile)

    profile_stats = aggregate_profile_stats(final_profile)
    stats = {
        "step": 2,
        "created_at": now_iso(),
        "index_file": str(index_path),
        "query_layout": layout,
        "global_range": [start, end],
        "k": int(args.k),
        "output_csv": str(final_output),
        "profile_csv": str(final_profile),
        "checkpoint": str(checkpoint_path),
        "profile_stats": profile_stats,
        "resource_usage_across_resume_sessions": aggregate_resource_samples(cpu_samples_path),
        "note": "global range is left-closed/right-open over query files in the exact order passed to --query-csvs",
    }
    atomic_json_dump(stats, stats_path)

    logger.info("Step 2 finished")
    logger.info("output=%s", final_output)
    logger.info("profile=%s", final_profile)
    logger.info("stats=%s", stats_path)
    return final_output
