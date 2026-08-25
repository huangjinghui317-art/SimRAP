#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Checkpoint-only inference on a validation CSV.

Only --checkpoint and --val-csv are required. Column names and graph
configuration are read from the checkpoint whenever available.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoints import safe_torch_load
from .dataset import BulkFusionDataset, collate_bulk_fusion, move_batch
from .graph import CrystalGraphBuilder
from .model import BulkModulusFusionModel, ModelConfig
from .preprocessing import dataframe_to_records
from .runtime import resolve_device


def parse_infer_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a wrapper checkpoint and run inference on a validation CSV."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a wrapper checkpoint (.pt).",
    )

    parser.add_argument(
        "--val-csv",
        required=True,
        help="Validation CSV used for inference.",
    )

    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Prediction CSV path. By default, "
            "<val_stem>_wrapper_predictions.csv is written beside the input CSV."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--graph-cache-size",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--target-col",
        default=None,
        help="Override the target column stored in the checkpoint.",
    )

    parser.add_argument(
        "--query-cif-col",
        default=None,
        help="Override the query CIF column stored in the checkpoint.",
    )

    parser.add_argument(
        "--backbone-output-col",
        default=None,
        help="Override the backbone-output column stored in the checkpoint.",
    )

    return parser.parse_args()


def _resolve_column(
    explicit_value: Optional[str],
    data_config: Dict[str, Any],
    config_key: str,
    fallback: str,
) -> str:
    if explicit_value:
        return str(explicit_value)

    value = data_config.get(config_key)

    if value:
        return str(value)

    return fallback


@torch.no_grad()
def run_inference(
    model: BulkModulusFusionModel,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()

    predictions = []
    row_indices = []

    for batch in tqdm(
        loader,
        desc="Inference",
        unit="batch",
    ):
        source_rows = list(
            batch["source_row_idx"]
        )

        batch = move_batch(
            batch,
            device,
        )

        pred_scaled = model(
            batch
        )

        pred = model.unscale_value(
            pred_scaled
        )

        predictions.extend(
            pred.detach().cpu().numpy().tolist()
        )

        row_indices.extend(
            int(x)
            for x in source_rows
        )

    return row_indices, predictions


def main():
    args = parse_infer_args()

    checkpoint_path = Path(
        args.checkpoint
    ).expanduser().resolve()

    val_csv_path = Path(
        args.val_csv
    ).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    if not val_csv_path.exists():
        raise FileNotFoundError(
            val_csv_path
        )

    output_csv = (
        Path(args.output_csv)
        .expanduser()
        .resolve()
        if args.output_csv
        else val_csv_path.with_name(
            val_csv_path.stem
            + "_wrapper_predictions.csv"
        )
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Loading checkpoint: {checkpoint_path}"
    )

    checkpoint = safe_torch_load(
        checkpoint_path,
        map_location="cpu",
    )

    required_checkpoint_keys = [
        "model_state",
        "model_config",
        "target_mean",
        "target_std",
    ]

    missing_checkpoint_keys = [
        key
        for key in required_checkpoint_keys
        if key not in checkpoint
    ]

    if missing_checkpoint_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            f"{missing_checkpoint_keys}"
        )

    data_config = dict(
        checkpoint.get(
            "data_config",
            {},
        )
    )

    target_col = _resolve_column(
        args.target_col,
        data_config,
        "target_col",
        "query_target_value",
    )

    query_cif_col = _resolve_column(
        args.query_cif_col,
        data_config,
        "query_cif_col",
        "query_cif",
    )

    backbone_output_col = _resolve_column(
        args.backbone_output_col,
        data_config,
        "backbone_output_col",
        "backbone_output",
    )

    graph_cutoff = float(
        data_config.get(
            "cutoff",
            checkpoint["model_config"].get(
                "cutoff",
                5.0,
            ),
        )
    )

    print(
        f"Validation CSV: {val_csv_path}"
    )

    print(
        "Columns: "
        f"target={target_col}, "
        f"query_cif={query_cif_col}, "
        f"backbone_output={backbone_output_col}"
    )

    val_df = pd.read_csv(
        val_csv_path,
        low_memory=False,
    )

    records, preprocessing_stats = (
        dataframe_to_records(
            val_df,
            target_col=target_col,
            query_cif_col=query_cif_col,
            backbone_output_col=
                backbone_output_col,
            logger=None,
        )
    )

    model_config = ModelConfig(
        **checkpoint[
            "model_config"
        ]
    )

    graph_builder = CrystalGraphBuilder(
        cutoff=graph_cutoff,
        cache_size=
            args.graph_cache_size,
    )

    dataset = BulkFusionDataset(
        records,
        graph_builder,
    )

    loader = DataLoader(
        dataset,
        batch_size=
            args.batch_size,
        shuffle=False,
        num_workers=
            args.num_workers,
        collate_fn=
            collate_bulk_fusion,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = BulkModulusFusionModel(
        config=model_config,
        target_mean=float(
            checkpoint[
                "target_mean"
            ]
        ),
        target_std=float(
            checkpoint[
                "target_std"
            ]
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    row_indices, predictions = (
        run_inference(
            model,
            loader,
            device,
        )
    )

    result_df = val_df.copy()

    result_df[
        "wrapper_prediction"
    ] = np.nan

    result_df[
        "wrapper_absolute_error"
    ] = np.nan

    result_df[
        "backbone_absolute_error"
    ] = np.nan

    result_df[
        "wrapper_inference_status"
    ] = "skipped"

    for row_index, prediction in zip(
        row_indices,
        predictions,
    ):
        result_df.at[
            row_index,
            "wrapper_prediction",
        ] = float(prediction)

        target_value = result_df.at[
            row_index,
            target_col,
        ]

        backbone_value = result_df.at[
            row_index,
            backbone_output_col,
        ]

        try:
            target_value = float(
                target_value
            )
        except Exception:
            target_value = math.nan

        try:
            backbone_value = float(
                backbone_value
            )
        except Exception:
            backbone_value = math.nan

        if math.isfinite(
            target_value
        ):
            result_df.at[
                row_index,
                "wrapper_absolute_error",
            ] = abs(
                float(prediction)
                - target_value
            )

            if math.isfinite(
                backbone_value
            ):
                result_df.at[
                    row_index,
                    "backbone_absolute_error",
                ] = abs(
                    backbone_value
                    - target_value
                )

        result_df.at[
            row_index,
            "wrapper_inference_status",
        ] = "ok"

    result_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    valid_wrapper_error = pd.to_numeric(
        result_df[
            "wrapper_absolute_error"
        ],
        errors="coerce",
    ).dropna()

    valid_backbone_error = pd.to_numeric(
        result_df[
            "backbone_absolute_error"
        ],
        errors="coerce",
    ).dropna()

    metrics = {
        "checkpoint":
            str(checkpoint_path),

        "validation_csv":
            str(val_csv_path),

        "output_csv":
            str(output_csv),

        "checkpoint_epoch":
            int(
                checkpoint.get(
                    "epoch",
                    -1,
                )
            ),

        "input_rows":
            int(
                len(val_df)
            ),

        "inferred_rows":
            int(
                len(predictions)
            ),

        "wrapper_mae":
            (
                float(
                    valid_wrapper_error.mean()
                )
                if len(
                    valid_wrapper_error
                ) > 0
                else None
            ),

        "backbone_mae":
            (
                float(
                    valid_backbone_error.mean()
                )
                if len(
                    valid_backbone_error
                ) > 0
                else None
            ),

        "target_col":
            target_col,

        "query_cif_col":
            query_cif_col,

        "backbone_output_col":
            backbone_output_col,

        "preprocessing":
            preprocessing_stats,
    }

    metrics_path = (
        output_csv.with_suffix(
            ".metrics.json"
        )
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Inferred rows: "
        f"{len(predictions):,}"
    )

    if metrics[
        "wrapper_mae"
    ] is not None:
        print(
            "Wrapper MAE: "
            f"{metrics['wrapper_mae']:.6f}"
        )

    if metrics[
        "backbone_mae"
    ] is not None:
        print(
            "Backbone MAE: "
            f"{metrics['backbone_mae']:.6f}"
        )

    print(
        f"Predictions: {output_csv}"
    )

    print(
        f"Metrics: {metrics_path}"
    )


if __name__ == "__main__":
    main()
