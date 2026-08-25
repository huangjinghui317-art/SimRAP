#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Command-line parser and top-level dispatch blocks."""

import argparse
from pathlib import Path
from typing import Optional

from .build_index import build_index
from .retrieve_step import retrieve_step


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Hierarchical SPG/Wyckoff/family/element-site kNN retrieval for crystal CIF datasets",
    )
    p.add_argument("--step", choices=["1", "2", "all"], default="all", help="Run index build, retrieval, or both")
    p.add_argument("--train-csv", required=True, help="Path to train.csv used to build/search the index")
    p.add_argument("--query-csvs", nargs="*", default=None, help="Query CSV files for Step 2, e.g. train.csv val.csv test.csv")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--index-file", default=None, help="Optional custom index .pkl path")
    p.add_argument("--output-csv", default=None, help="Optional custom final Step-2 CSV path")

    p.add_argument("--cif-column", default="cif", help="CIF text column")
    p.add_argument("--target-column", default="target_value", help="Target property column")
    p.add_argument("--id-column", default=None, help="Optional material ID column; if omitted, Step 2 auto-uses material_id when present")

    p.add_argument("--k", type=int, default=3, help="Maximum number of neighbors per query")
    p.add_argument("--start-index", type=int, default=0, help="Global Step-2 start index, inclusive")
    p.add_argument("--end-index", type=int, default=None, help="Global Step-2 end index, exclusive; default=all")
    p.add_argument("--resume", action="store_true", help="Resume Step 2 from its checkpoint/part files")
    p.add_argument("--overwrite", action="store_true", help="Delete existing Step-2 range artifacts and restart")
    p.add_argument("--checkpoint-every", type=int, default=100, help="Atomic Step-2 shard/checkpoint interval in query materials")

    p.add_argument("--symprec", type=float, default=0.01, help="Pymatgen/spglib symmetry tolerance used when building index")
    p.add_argument("--angle-tolerance", type=float, default=5.0, help="SpacegroupAnalyzer angle tolerance")
    p.add_argument("--standardize", choices=["refined", "none"], default="refined", help="Refine to a conventional symmetry cell before extracting Wyckoff letters")

    p.add_argument("--csv-chunk-size", type=int, default=1000, help="Pandas CSV chunk size")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding")
    p.add_argument("--resource-sample-interval", type=float, default=0.5, help="CPU/RSS sample interval in seconds")
    p.add_argument("--log-every", type=int, default=1000, help="Step-1 progress logging interval")
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")
    if args.csv_chunk_size <= 0:
        raise ValueError("--csv-chunk-size must be > 0")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.end_index is not None and args.end_index <= args.start_index:
        raise ValueError("--end-index must be greater than --start-index")
    if args.step in ("2", "all") and not args.query_csvs:
        raise ValueError("Step 2/all requires --query-csvs")


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path: Optional[Path] = None
    if args.step in ("1", "all"):
        index_path = build_index(args)

    if args.step in ("2", "all"):
        if index_path is None:
            index_path = Path(args.index_file).expanduser().resolve() if args.index_file else output_dir / "similarity_index.pkl"
        retrieve_step(args, index_path=index_path)

    return 0
