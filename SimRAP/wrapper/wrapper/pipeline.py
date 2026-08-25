#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Top-level wrapper training pipeline."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .checkpoints import (
    find_latest_checkpoint,
    rotate_checkpoints,
    safe_torch_load,
    save_checkpoint,
)
from .cli import parse_args
from .dataset import BulkFusionDataset, collate_bulk_fusion
from .graph import CrystalGraphBuilder
from .history import write_history
from .model import BulkModulusFusionModel, ModelConfig
from .preprocessing import dataframe_to_records
from .runtime import resolve_device, set_seed, setup_logger
from .split import deterministic_split, target_stats
from .training import build_loss, evaluate, train_one_epoch


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logger(
        output_dir
    )

    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    logger.info(
        "Device: %s",
        device,
    )

    # --------------------------------------------------------
    # Resume checkpoint early, because architecture/data config
    # should come from the checkpoint.
    # --------------------------------------------------------

    resume_ckpt = None

    if args.resume:
        if args.resume == "auto":
            resume_path = (
                find_latest_checkpoint(
                    checkpoint_dir
                )
            )

            if resume_path is None:
                raise FileNotFoundError(
                    "No checkpoint found for --resume auto."
                )

        else:
            resume_path = Path(
                args.resume
            ).expanduser().resolve()

            if not resume_path.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_path}"
                )

        logger.info(
            "Loading resume checkpoint: %s",
            resume_path,
        )

        resume_ckpt = safe_torch_load(
            resume_path,
            map_location="cpu",
        )

    # --------------------------------------------------------
    # Read train records
    # --------------------------------------------------------

    train_csv_path = Path(
        args.train_csv
    ).expanduser().resolve()

    train_df = pd.read_csv(
        train_csv_path,
        low_memory=False,
    )

    train_records, train_stats = (
        dataframe_to_records(
            train_df,
            target_col=
                args.target_col,
            query_cif_col=
                args.query_cif_col,
            backbone_output_col=
                args.backbone_output_col,
            logger=logger,
        )
    )

    # --------------------------------------------------------
    # Validation source / split
    # --------------------------------------------------------

    external_val_records = None

    if (
        args.val_every > 0
        and args.val_csv
    ):
        val_path = Path(
            args.val_csv
        ).expanduser().resolve()

        val_df = pd.read_csv(
            val_path,
            low_memory=False,
        )

        external_val_records, val_stats = (
            dataframe_to_records(
                val_df,
                target_col=
                    args.target_col,
                query_cif_col=
                    args.query_cif_col,
                backbone_output_col=
                    args.backbone_output_col,
                logger=logger,
            )
        )

        train_indices = list(
            range(
                len(train_records)
            )
        )

        val_indices = None

    elif args.val_every > 0:
        (
            train_indices,
            val_indices,
        ) = deterministic_split(
            n=len(
                train_records
            ),
            val_fraction=
                args.val_fraction,
            seed=
                args.seed,
        )

        logger.info(
            "Internal split: train=%d, val=%d",
            len(train_indices),
            len(val_indices),
        )

    else:
        train_indices = list(
            range(
                len(train_records)
            )
        )

        val_indices = None

    # --------------------------------------------------------
    # Model/data config
    # --------------------------------------------------------

    if resume_ckpt is not None:
        model_config = ModelConfig(
            **resume_ckpt[
                "model_config"
            ]
        )

        target_mean = float(
            resume_ckpt[
                "target_mean"
            ]
        )

        target_std = float(
            resume_ckpt[
                "target_std"
            ]
        )

        data_config = dict(
            resume_ckpt[
                "data_config"
            ]
        )

        # Ensure resumed graph construction matches checkpoint.
        graph_cutoff = float(
            data_config[
                "cutoff"
            ]
        )

        logger.info(
            "Resume uses checkpoint model/data configuration."
        )

    else:
        target_mean, target_std = (
            target_stats(
                train_records,
                train_indices,
            )
        )

        max_train_rank = max(
            train_stats["ranks"]
        )

        model_config = ModelConfig(
            atom_embed_dim=
                args.atom_embed_dim,

            graph_hidden_dim=
                args.graph_hidden_dim,

            graph_layers=
                args.graph_layers,

            rbf_dim=
                args.rbf_dim,

            cutoff=
                args.cutoff,

            source_embed_dim=
                args.source_embed_dim,

            neighbor_token_dim=
                args.neighbor_token_dim,

            gru_hidden_dim=
                args.gru_hidden_dim,

            gru_layers=
                args.gru_layers,

            bidirectional=
                not args.unidirectional_gru,

            backbone_output_token_dim=
                args.backbone_output_token_dim,

            fusion_hidden_dim=
                args.fusion_hidden_dim,

            dropout=
                args.dropout,

            rank_scale=
                float(
                    max_train_rank
                ),
        )

        graph_cutoff = float(
            args.cutoff
        )

        data_config = {
            "cutoff":
                graph_cutoff,

            "target_col":
                args.target_col,

            "query_cif_col":
                args.query_cif_col,

            "backbone_output_col":
                args.backbone_output_col,

            "train_ranks":
                train_stats["ranks"],
        }

    logger.info(
        "Target normalization: mean=%.6f, std=%.6f",
        target_mean,
        target_std,
    )

    logger.info(
        "Model config: %s",
        json.dumps(
            asdict(
                model_config
            ),
            ensure_ascii=False,
        ),
    )

    # --------------------------------------------------------
    # Dataset / loaders
    # --------------------------------------------------------

    graph_builder = (
        CrystalGraphBuilder(
            cutoff=
                graph_cutoff,
            cache_size=
                args.graph_cache_size,
        )
    )

    train_dataset_full = (
        BulkFusionDataset(
            train_records,
            graph_builder,
        )
    )

    train_dataset = Subset(
        train_dataset_full,
        train_indices,
    )

    if external_val_records is not None:
        val_dataset = (
            BulkFusionDataset(
                external_val_records,
                graph_builder,
            )
        )

    elif val_indices is not None:
        val_dataset = Subset(
            train_dataset_full,
            val_indices,
        )

    else:
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            args.batch_size,
        shuffle=True,
        num_workers=
            args.num_workers,
        collate_fn=
            collate_bulk_fusion,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=
                args.batch_size,
            shuffle=False,
            num_workers=
                args.num_workers,
            collate_fn=
                collate_bulk_fusion,
            pin_memory=(
                device.type
                == "cuda"
            ),
        )
        if val_dataset is not None
        else None
    )

    # --------------------------------------------------------
    # Model / optimizer
    # --------------------------------------------------------

    model = BulkModulusFusionModel(
        config=
            model_config,
        target_mean=
            target_mean,
        target_std=
            target_std,
    ).to(
        device
    )

    criterion = build_loss(
        args.loss
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=
            args.weight_decay,
    )

    start_epoch = 1
    history = []
    best_val_mae = math.inf

    if resume_ckpt is not None:
        model.load_state_dict(
            resume_ckpt[
                "model_state"
            ]
        )

        optimizer.load_state_dict(
            resume_ckpt[
                "optimizer_state"
            ]
        )

        # Move optimizer states onto active device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = (
                        value.to(device)
                    )

        start_epoch = (
            int(
                resume_ckpt[
                    "epoch"
                ]
            )
            + 1
        )

        history = list(
            resume_ckpt.get(
                "history",
                [],
            )
        )

        best_val_mae = float(
            resume_ckpt.get(
                "best_val_mae",
                math.inf,
            )
        )

        logger.info(
            "Resumed from epoch %d; next epoch=%d",
            start_epoch - 1,
            start_epoch,
        )

    if start_epoch > args.epochs:
        logger.info(
            "Checkpoint already reached epoch %d >= requested epochs %d.",
            start_epoch - 1,
            args.epochs,
        )
        return

    # Save run config for readability.
    with (
        output_dir
        / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "args":
                    vars(args),

                "model_config":
                    asdict(
                        model_config
                    ),

                "data_config":
                    data_config,

                "target_mean":
                    target_mean,

                "target_std":
                    target_std,

                "train_stats":
                    train_stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    training_config = {
        "loss":
            args.loss,

        "lr":
            args.lr,

        "weight_decay":
            args.weight_decay,

        "seed":
            args.seed,
    }

    # --------------------------------------------------------
    # Train loop
    # --------------------------------------------------------

    logger.info(
        "Training starts: epochs=%d, train_N=%d, val_N=%s",
        args.epochs,
        len(train_dataset),
        (
            len(val_dataset)
            if val_dataset is not None
            else "disabled"
        ),
    )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=
                args.grad_clip,
            use_amp=
                args.amp,
            epoch=
                epoch,
        )

        val_loss = math.nan
        val_mae = math.nan

        do_validation = (
            args.val_every > 0
            and val_loader is not None
            and (
                epoch
                % args.val_every
                == 0
            )
        )

        if do_validation:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                desc=(
                    f"Validation {epoch}"
                ),
            )

            val_loss = (
                val_metrics[
                    "loss"
                ]
            )

            val_mae = (
                val_metrics[
                    "mae_gpa"
                ]
            )

            if (
                math.isfinite(
                    val_mae
                )
                and val_mae
                < best_val_mae
            ):
                best_val_mae = (
                    val_mae
                )

        elapsed = (
            time.time()
            - epoch_start
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_metrics[
                    "loss"
                ],

            "train_mae_gpa":
                train_metrics[
                    "mae_gpa"
                ],

            "val_loss":
                val_loss,

            "val_mae_gpa":
                val_mae,

            "lr":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ],

            "seconds":
                elapsed,
        }

        history.append(
            row
        )

        logger.info(
            "Epoch %d | train_loss=%.6f | train_MAE=%.6f GPa | "
            "val_loss=%s | val_MAE=%s | %.1fs",
            epoch,
            row["train_loss"],
            row["train_mae_gpa"],
            (
                f"{val_loss:.6f}"
                if math.isfinite(
                    val_loss
                )
                else "NA"
            ),
            (
                f"{val_mae:.6f} GPa"
                if math.isfinite(
                    val_mae
                )
                else "NA"
            ),
            elapsed,
        )

        write_history(
            history,
            output_dir,
        )

        scheduled_save = (
            args.save_every > 0
            and (
                epoch
                % args.save_every
                == 0
            )
        )

        final_save = (
            epoch
            == args.epochs
        )

        if (
            scheduled_save
            or final_save
        ):
            ckpt_path = (
                checkpoint_dir
                / (
                    f"ckpt_epoch_"
                    f"{epoch:04d}.pt"
                )
            )

            save_checkpoint(
                path=ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                history=history,
                model_config=
                    model_config,
                data_config=
                    data_config,
                training_config=
                    training_config,
                best_val_mae=
                    best_val_mae,
            )

            rotate_checkpoints(
                checkpoint_dir,
                args.max_ckpts,
            )

            logger.info(
                "Checkpoint saved: %s",
                ckpt_path,
            )

    logger.info(
        "Training finished."
    )

    if math.isfinite(
        best_val_mae
    ):
        logger.info(
            "Best observed validation MAE: %.6f GPa",
            best_val_mae,
        )
