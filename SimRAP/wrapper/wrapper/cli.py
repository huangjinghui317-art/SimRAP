#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Command-line arguments for wrapper training."""

import argparse


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train-csv",
        required=True,
    )

    p.add_argument(
        "--val-csv",
        default=None,
        help=(
            "Independent validation CSV. If omitted while --val-every > 0, "
            "a deterministic validation split is created from the training CSV."
        ),
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--target-col",
        default="query_target_value",
    )

    p.add_argument(
        "--query-cif-col",
        default="query_cif",
    )

    p.add_argument(
        "--backbone_output-col",
        default=
            "backbone_output",
    )

    p.add_argument(
        "--device",
        default="auto",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    p.add_argument(
        "--loss",
        choices=[
            "smoothl1",
            "mse",
            "mae",
        ],
        default="smoothl1",
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--amp",
        action="store_true",
        help="CUDA mixed precision",
    )

    p.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run validation every m epochs; 0 disables validation.",
    )

    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save a checkpoint every k epochs.",
    )

    p.add_argument(
        "--max-ckpts",
        type=int,
        default=4,
        help="Maximum number of checkpoints to retain.",
    )

    p.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume training from auto or an explicit checkpoint path."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Graph
    p.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--graph-cache-size",
        type=int,
        default=20000,
    )

    p.add_argument(
        "--atom-embed-dim",
        type=int,
        default=128,
    )

    p.add_argument(
        "--graph-hidden-dim",
        type=int,
        default=128,
    )

    p.add_argument(
        "--graph-layers",
        type=int,
        default=3,
    )

    p.add_argument(
        "--rbf-dim",
        type=int,
        default=32,
    )

    # Neighbor sequence
    p.add_argument(
        "--source-embed-dim",
        type=int,
        default=16,
    )

    p.add_argument(
        "--neighbor-token-dim",
        type=int,
        default=192,
    )

    p.add_argument(
        "--gru-hidden-dim",
        type=int,
        default=128,
    )

    p.add_argument(
        "--gru-layers",
        type=int,
        default=1,
    )

    p.add_argument(
        "--unidirectional-gru",
        action="store_true",
    )

    # Fusion
    p.add_argument(
        "--backbone-output-token-dim",
        type=int,
        default=64,
    )

    p.add_argument(
        "--fusion-hidden-dim",
        type=int,
        default=256,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.15,
    )

    return p.parse_args()
