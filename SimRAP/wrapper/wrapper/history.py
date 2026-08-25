#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Training-history CSV export and curve plotting."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def write_history(
    history: List[Dict[str, Any]],
    output_dir: Path,
):
    history_df = pd.DataFrame(
        history
    )

    history_path = (
        output_dir
        / "history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    if history_df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        markersize=3,
        label="Train loss",
    )

    if (
        "val_loss"
        in history_df.columns
        and history_df[
            "val_loss"
        ].notna().any()
    ):
        val_rows = history_df[
            history_df[
                "val_loss"
            ].notna()
        ]

        ax.plot(
            val_rows["epoch"],
            val_rows["val_loss"],
            marker="s",
            markersize=4,
            label="Validation loss",
        )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        "Training Loss Curve"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "training_curves.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)
