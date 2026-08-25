#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Checkpoint loading, saving, and rotation."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .model import BulkModulusFusionModel, ModelConfig


def safe_torch_load(
    path: Path,
    map_location="cpu",
):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def find_latest_checkpoint(
    checkpoint_dir: Path,
) -> Optional[Path]:
    ckpts = sorted(
        checkpoint_dir.glob(
            "ckpt_epoch_*.pt"
        )
    )

    return (
        ckpts[-1]
        if ckpts
        else None
    )


def rotate_checkpoints(
    checkpoint_dir: Path,
    max_ckpts: int,
):
    if max_ckpts <= 0:
        return

    ckpts = sorted(
        checkpoint_dir.glob(
            "ckpt_epoch_*.pt"
        )
    )

    while len(ckpts) > max_ckpts:
        old = ckpts.pop(0)

        try:
            old.unlink()
        except FileNotFoundError:
            pass


def save_checkpoint(
    path: Path,
    epoch: int,
    model: BulkModulusFusionModel,
    optimizer,
    history: List[Dict[str, Any]],
    model_config: ModelConfig,
    data_config: Dict[str, Any],
    training_config: Dict[str, Any],
    best_val_mae: float,
):
    state = {
        "epoch": int(epoch),

        "model_state":
            model.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

        "history":
            history,

        "model_config":
            asdict(model_config),

        "data_config":
            data_config,

        "training_config":
            training_config,

        "target_mean":
            float(
                model.target_mean.item()
            ),

        "target_std":
            float(
                model.target_std.item()
            ),

        "best_val_mae":
            float(best_val_mae),
    }

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        state,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )
