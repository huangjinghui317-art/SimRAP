#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Loss construction, evaluation, and one-epoch training."""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import move_batch
from .model import BulkModulusFusionModel


def build_loss(
    name: str,
):
    name = name.lower()

    if name == "smoothl1":
        return nn.SmoothL1Loss(
            beta=1.0
        )

    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return nn.L1Loss()

    raise ValueError(
        f"Unsupported loss={name!r}. "
        "Use smoothl1/mse/mae."
    )


@torch.no_grad()
def evaluate(
    model: BulkModulusFusionModel,
    loader: DataLoader,
    criterion,
    device: torch.device,
    desc: str = "Validation",
):
    model.eval()

    loss_sum = 0.0
    abs_error_sum = 0.0
    n_total = 0

    for batch in tqdm(
        loader,
        desc=desc,
        unit="batch",
        leave=False,
    ):
        batch = move_batch(
            batch,
            device,
        )

        target = batch[
            "target"
        ]

        target_scaled = (
            model.scale_value(
                target
            )
        )

        pred_scaled = model(
            batch
        )

        loss = criterion(
            pred_scaled,
            target_scaled,
        )

        pred = (
            model.unscale_value(
                pred_scaled
            )
        )

        batch_n = int(
            target.shape[0]
        )

        loss_sum += (
            float(loss.item())
            * batch_n
        )

        abs_error_sum += float(
            torch.abs(
                pred - target
            ).sum().item()
        )

        n_total += batch_n

    if n_total == 0:
        return {
            "loss": math.nan,
            "mae_gpa": math.nan,
            "n": 0,
        }

    return {
        "loss":
            loss_sum / n_total,

        "mae_gpa":
            abs_error_sum / n_total,

        "n":
            n_total,
    }


def train_one_epoch(
    model: BulkModulusFusionModel,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    epoch: int,
):
    model.train()

    loss_sum = 0.0
    abs_error_sum = 0.0
    n_total = 0

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}",
        unit="batch",
    )

    for batch in pbar:
        batch = move_batch(
            batch,
            device,
        )

        target = batch[
            "target"
        ]

        target_scaled = (
            model.scale_value(
                target
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=
                device.type,
            enabled=
                amp_enabled,
            dtype=
                torch.float16,
        ):
            pred_scaled = model(
                batch
            )

            loss = criterion(
                pred_scaled,
                target_scaled,
            )

        scaler.scale(
            loss
        ).backward()

        if grad_clip > 0:
            scaler.unscale_(
                optimizer
            )

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        scaler.step(
            optimizer
        )

        scaler.update()

        with torch.no_grad():
            pred = (
                model.unscale_value(
                    pred_scaled
                )
            )

            batch_abs_error = (
                torch.abs(
                    pred - target
                )
                .sum()
                .item()
            )

        batch_n = int(
            target.shape[0]
        )

        loss_sum += (
            float(loss.item())
            * batch_n
        )

        abs_error_sum += float(
            batch_abs_error
        )

        n_total += batch_n

        pbar.set_postfix(
            loss=(
                loss_sum
                / max(n_total, 1)
            ),
            mae=(
                abs_error_sum
                / max(n_total, 1)
            ),
        )

    return {
        "loss":
            loss_sum / n_total,

        "mae_gpa":
            abs_error_sum / n_total,

        "n":
            n_total,
    }
