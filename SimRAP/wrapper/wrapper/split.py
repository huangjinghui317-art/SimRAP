#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Target normalization and deterministic train/validation split."""

import math
import random
from typing import Sequence

import numpy as np

from .preprocessing import RowRecord


def target_stats(
    records: Sequence[RowRecord],
    indices: Sequence[int],
):
    values = np.asarray(
        [
            records[i].target
            for i in indices
        ],
        dtype=np.float64,
    )

    mean = float(
        values.mean()
    )

    std = float(
        values.std()
    )

    if (
        not math.isfinite(std)
        or std < 1e-6
    ):
        std = 1.0

    return mean, std


def deterministic_split(
    n: int,
    val_fraction: float,
    seed: int,
):
    if not (
        0.0
        < val_fraction
        < 1.0
    ):
        raise ValueError(
            "--val-fraction must be in (0,1)."
        )

    indices = list(
        range(n)
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        indices
    )

    n_val = max(
        1,
        int(
            round(
                n
                * val_fraction
            )
        ),
    )

    n_val = min(
        n_val,
        n - 1,
    )

    val_indices = sorted(
        indices[:n_val]
    )

    train_indices = sorted(
        indices[n_val:]
    )

    return (
        train_indices,
        val_indices,
    )
