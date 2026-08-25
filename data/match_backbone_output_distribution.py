#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
match_backbone_output_distribution.py

Purpose
-------
Treat CSV B as the target distribution and adjust CSV A using:

    backbone_output
    query_target_value

The script derives:

    residual = backbone_output - query_target_value

For every valid row in CSV A, the same additive perturbation delta is
applied to both quantities:

    backbone_output_new = backbone_output_old + delta
    residual_new        = residual_old + delta

Because:

    residual_old = backbone_output_old - query_target_value

the following invariant is preserved exactly:

    backbone_output_new - residual_new == query_target_value

Therefore, query_target_value is never modified, and the input CSV does
not need to contain a precomputed residual column.

The matching objective jointly considers:

1. the marginal residual distribution;
2. the marginal backbone_output distribution;
3. the conditional residual distribution along backbone_output;
4. the correlation between backbone_output and residual;
5. the jointly adjustable direction under the shared-perturbation
   constraint.

Method
------
Define:

    d = backbone_output - residual
    s = (backbone_output + residual) / 2

Under the shared additive perturbation, d is invariant while s is
movable. The script performs conditional quantile mapping of s_A toward
s_B within d bins and combines it with a global residual quantile
mapping. It then searches over blend weight and perturbation strength
using the original composite distribution-distance objective.

Outputs
-------
output_dir/
    adjusted_A.csv
    B_macro_summary.csv
    B_backbone_output_bin_residual_summary.csv
    comparison_metrics.csv
    search_scores.csv
    perturbation_audit.csv
    matching_report.txt
    best_parameters.json
    figure1_backbone_output_distribution.png
    figure2_residual_distribution.png
    figure3_joint_distribution.png
    figure4_residual_along_backbone_output.png
    figure5_delta_distribution.png

Dependencies
------------
    pip install pandas numpy matplotlib scipy

Example
-------
python match_backbone_output_distribution.py \
    --csv_a train.csv \
    --csv_b val.csv \
    --output_dir distribution_match \
    --backbone_output_col backbone_output \
    --target_col query_target_value \
    --d_bins 12 \
    --backbone_output_bins 12 \
    --seed 42

CSV B is the target distribution and CSV A is the dataset to be
adjusted. Swap --csv_a and --csv_b to reverse the matching direction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon


BACKBONE_OUTPUT_COL_DEFAULT = "backbone_output"
TARGET_COL_DEFAULT = "query_target_value"

# Gray-purple palette
PURPLE_DARK = "#5B476B"
PURPLE = "#7C6A8E"
PURPLE_LIGHT = "#B4A7C2"
GRAY_PURPLE = "#77717F"
GRAY_DARK = "#4F4C55"
GRAY = "#8E8A93"
GRAY_LIGHT = "#D6D2DA"
GRAY_VERY_LIGHT = "#F1EFF3"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Match CSV A backbone_output/residual distribution to CSV B using the same per-row additive perturbation."
        )
    )

    p.add_argument("--csv_a", required=True, help="Source CSV A to be adjusted.")
    p.add_argument("--csv_b", required=True, help="Target-distribution CSV B.")
    p.add_argument("--output_dir", required=True, help="Output directory.")

    p.add_argument(
        "--backbone_output_col",
        default=BACKBONE_OUTPUT_COL_DEFAULT,
        help=f"Backbone-output column. Default: {BACKBONE_OUTPUT_COL_DEFAULT}",
    )
    p.add_argument(
        "--target_col",
        default=TARGET_COL_DEFAULT,
        help=(
            f"Ground-truth target column. Default: {TARGET_COL_DEFAULT}. "
            "The residual is derived as backbone_output_col - target_col."
        ),
    )

    p.add_argument(
        "--d_bins",
        type=int,
        default=12,
        help="Number of quantile bins for invariant d=backbone_output-residual. Default: 12",
    )
    p.add_argument(
        "--backbone_output_bins",
        type=int,
        default=12,
        help="Number of backbone_output quantile bins for conditional residual statistics. Default: 12",
    )
    p.add_argument(
        "--hist_bins",
        type=int,
        default=50,
        help="Number of histogram bins. Default: 50",
    )
    p.add_argument(
        "--min_bin_samples",
        type=int,
        default=12,
        help="Minimum samples per d-bin for conditional quantile mapping. Default: 12",
    )

    p.add_argument(
        "--blend_steps",
        type=int,
        default=21,
        help="Number of blend-weight search points in [0, 1]. Default: 21",
    )
    p.add_argument(
        "--strength_min",
        type=float,
        default=0.70,
        help="Lower bound of perturbation strength. Default: 0.70",
    )
    p.add_argument(
        "--strength_max",
        type=float,
        default=1.15,
        help="Upper bound of perturbation strength. Default: 1.15",
    )
    p.add_argument(
        "--strength_steps",
        type=int,
        default=10,
        help="Number of perturbation-strength search points. Default: 10",
    )

    p.add_argument(
        "--max_abs_delta",
        type=float,
        default=None,
        help="Optional maximum absolute perturbation |delta|. Default: unlimited",
    )
    p.add_argument(
        "--clip_to_b_range",
        action="store_true",
        help=(
            "Optionally clip adjusted backbone_output to the [0.1%, 99.9%] range of B. "
            "The residual is then recomputed as backbone_output-query_target_value, "
            "so query_target_value remains unchanged."
        ),
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)

    return p.parse_args()


def numeric_pair(df: pd.DataFrame, backbone_output_col: str, target_col: str):
    """
    Read backbone_output and query target, then derive:
        residual = backbone_output - query_target_value
    """
    backbone_output = pd.to_numeric(df[backbone_output_col], errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    residual = backbone_output - target
    valid = np.isfinite(backbone_output) & np.isfinite(target) & np.isfinite(residual)
    return backbone_output, residual, target, valid


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name} is missing columns: {missing}\nAvailable columns: {list(df.columns)}"
        )


def safe_std(x: np.ndarray) -> float:
    if len(x) <= 1:
        return float("nan")
    return float(np.std(x, ddof=1))


def describe_array(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f"{prefix}_n": 0}

    qs = np.percentile(x, [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9])
    return {
        f"{prefix}_n": int(len(x)),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": safe_std(x),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_p0_1": float(qs[0]),
        f"{prefix}_p1": float(qs[1]),
        f"{prefix}_p5": float(qs[2]),
        f"{prefix}_p10": float(qs[3]),
        f"{prefix}_p25": float(qs[4]),
        f"{prefix}_median": float(qs[5]),
        f"{prefix}_p75": float(qs[6]),
        f"{prefix}_p90": float(qs[7]),
        f"{prefix}_p95": float(qs[8]),
        f"{prefix}_p99": float(qs[9]),
        f"{prefix}_p99_9": float(qs[10]),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_mean_abs": float(np.mean(np.abs(x))),
    }


def macro_summary(backbone_output: np.ndarray, residual: np.ndarray, name: str) -> pd.DataFrame:
    d = backbone_output - residual
    s = (backbone_output + residual) / 2.0

    row: Dict[str, float | str] = {"dataset": name}
    row.update(describe_array(backbone_output, "backbone_output"))
    row.update(describe_array(residual, "residual"))
    row.update(describe_array(d, "invariant_d_backbone_output_minus_residual"))
    row.update(describe_array(s, "shift_coord_s"))

    if len(backbone_output) > 1 and np.std(backbone_output) > 0 and np.std(residual) > 0:
        row["backbone_output_residual_pearson_r"] = float(np.corrcoef(backbone_output, residual)[0, 1])
    else:
        row["backbone_output_residual_pearson_r"] = np.nan

    return pd.DataFrame([row])


def unique_quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("Cannot build quantile bins from empty data.")

    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)

    if len(edges) < 2:
        v = float(x[0])
        eps = max(abs(v) * 1e-6, 1e-6)
        edges = np.array([v - eps, v + eps])

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def backbone_output_bin_summary(
    backbone_output: np.ndarray,
    residual: np.ndarray,
    edges: np.ndarray,
    dataset_name: str,
) -> pd.DataFrame:
    rows = []
    ids = np.digitize(backbone_output, edges[1:-1], right=False)

    for i in range(len(edges) - 1):
        mask = ids == i
        if not np.any(mask):
            rows.append(
                {
                    "dataset": dataset_name,
                    "bin": i,
                    "n": 0,
                    "backbone_output_left": edges[i],
                    "backbone_output_right": edges[i + 1],
                }
            )
            continue

        b = backbone_output[mask]
        k = residual[mask]
        rows.append(
            {
                "dataset": dataset_name,
                "bin": i,
                "n": int(mask.sum()),
                "backbone_output_left": edges[i],
                "backbone_output_right": edges[i + 1],
                "backbone_output_mean": float(np.mean(b)),
                "backbone_output_median": float(np.median(b)),
                "residual_mean": float(np.mean(k)),
                "residual_std": safe_std(k),
                "residual_median": float(np.median(k)),
                "residual_p10": float(np.percentile(k, 10)),
                "residual_p25": float(np.percentile(k, 25)),
                "residual_p75": float(np.percentile(k, 75)),
                "residual_p90": float(np.percentile(k, 90)),
                "residual_mean_abs": float(np.mean(np.abs(k))),
            }
        )

    return pd.DataFrame(rows)


def empirical_percentile_ranks(x: np.ndarray) -> np.ndarray:
    """Return deterministic empirical percentile ranks in (0,1)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = (np.arange(len(x), dtype=float) + 0.5) / len(x)
    return ranks


def quantile_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map source values to target distribution while preserving source ranks."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if len(source) == 0:
        return source.copy()
    if len(target) == 0:
        return source.copy()
    if len(target) == 1:
        return np.full_like(source, float(target[0]))

    p = empirical_percentile_ranks(source)
    return np.quantile(target, p)


def conditional_s_quantile_map(
    d_a: np.ndarray,
    s_a: np.ndarray,
    d_b: np.ndarray,
    s_b: np.ndarray,
    n_bins: int,
    min_bin_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Under same-delta constraint, d=backbone_output-residual is invariant and s=(backbone_output+residual)/2 is movable.
    Match s_A to s_B conditionally on d bins.
    """
    # Pooled d edges are more robust when A/B ranges differ.
    pooled_d = np.concatenate([d_a, d_b])
    edges = unique_quantile_edges(pooled_d, n_bins)

    bin_a = np.digitize(d_a, edges[1:-1], right=False)
    bin_b = np.digitize(d_b, edges[1:-1], right=False)

    mapped = np.full_like(s_a, np.nan, dtype=float)
    global_map = quantile_map(s_a, s_b)

    for i in range(len(edges) - 1):
        ia = np.where(bin_a == i)[0]
        ib = np.where(bin_b == i)[0]

        if len(ia) == 0:
            continue

        if len(ia) >= min_bin_samples and len(ib) >= min_bin_samples:
            mapped[ia] = quantile_map(s_a[ia], s_b[ib])
        else:
            mapped[ia] = global_map[ia]

    missing = ~np.isfinite(mapped)
    mapped[missing] = global_map[missing]
    return mapped, edges


def histogram_js(a: np.ndarray, b: np.ndarray, bins: int = 60) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=edges, density=False)
    pa = ha.astype(float) + 1e-12
    pb = hb.astype(float) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    return float(jensenshannon(pa, pb, base=2.0) ** 2)


def quantile_mae(a: np.ndarray, b: np.ndarray) -> float:
    q = np.linspace(0.01, 0.99, 99)
    qa = np.quantile(a, q)
    qb = np.quantile(b, q)
    return float(np.mean(np.abs(qa - qb)))


def conditional_metrics(
    backbone_output_a: np.ndarray,
    residual_a: np.ndarray,
    backbone_output_b: np.ndarray,
    residual_b: np.ndarray,
    edges: np.ndarray,
) -> Dict[str, float]:
    ida = np.digitize(backbone_output_a, edges[1:-1], right=False)
    idb = np.digitize(backbone_output_b, edges[1:-1], right=False)

    total_weight = 0.0
    mean_diff = 0.0
    median_diff = 0.0
    std_diff = 0.0

    for i in range(len(edges) - 1):
        aa = residual_a[ida == i]
        bb = residual_b[idb == i]
        if len(aa) < 3 or len(bb) < 3:
            continue

        w = float(len(bb))
        total_weight += w
        mean_diff += w * abs(float(np.mean(aa)) - float(np.mean(bb)))
        median_diff += w * abs(float(np.median(aa)) - float(np.median(bb)))
        std_diff += w * abs(float(np.std(aa)) - float(np.std(bb)))

    if total_weight == 0:
        return {
            "conditional_residual_mean_mae": np.nan,
            "conditional_residual_median_mae": np.nan,
            "conditional_residual_std_mae": np.nan,
        }

    return {
        "conditional_residual_mean_mae": mean_diff / total_weight,
        "conditional_residual_median_mae": median_diff / total_weight,
        "conditional_residual_std_mae": std_diff / total_weight,
    }


def distribution_metrics(
    backbone_output_a: np.ndarray,
    residual_a: np.ndarray,
    backbone_output_b: np.ndarray,
    residual_b: np.ndarray,
    backbone_output_edges: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    backbone_output_scale = max(float(np.std(backbone_output_b)), 1e-8)
    residual_scale = max(float(np.std(residual_b)), 1e-8)

    metrics["backbone_output_wasserstein"] = float(wasserstein_distance(backbone_output_a, backbone_output_b))
    metrics["residual_wasserstein"] = float(wasserstein_distance(residual_a, residual_b))
    metrics["backbone_output_wasserstein_norm"] = metrics["backbone_output_wasserstein"] / backbone_output_scale
    metrics["residual_wasserstein_norm"] = metrics["residual_wasserstein"] / residual_scale

    metrics["backbone_output_ks"] = float(ks_2samp(backbone_output_a, backbone_output_b).statistic)
    metrics["residual_ks"] = float(ks_2samp(residual_a, residual_b).statistic)

    metrics["backbone_output_js"] = histogram_js(backbone_output_a, backbone_output_b)
    metrics["residual_js"] = histogram_js(residual_a, residual_b)

    metrics["backbone_output_quantile_mae"] = quantile_mae(backbone_output_a, backbone_output_b)
    metrics["residual_quantile_mae"] = quantile_mae(residual_a, residual_b)

    corr_a = (
        float(np.corrcoef(backbone_output_a, residual_a)[0, 1])
        if len(backbone_output_a) > 1 and np.std(backbone_output_a) > 0 and np.std(residual_a) > 0
        else np.nan
    )
    corr_b = (
        float(np.corrcoef(backbone_output_b, residual_b)[0, 1])
        if len(backbone_output_b) > 1 and np.std(backbone_output_b) > 0 and np.std(residual_b) > 0
        else np.nan
    )
    metrics["backbone_output_residual_corr_a"] = corr_a
    metrics["backbone_output_residual_corr_b"] = corr_b
    metrics["backbone_output_residual_corr_abs_diff"] = (
        abs(corr_a - corr_b) if np.isfinite(corr_a) and np.isfinite(corr_b) else np.nan
    )

    cond = conditional_metrics(backbone_output_a, residual_a, backbone_output_b, residual_b, backbone_output_edges)
    metrics.update(cond)

    # Composite score: scale-dependent terms are normalized using the target residual scale.
    cmean = cond["conditional_residual_mean_mae"]
    cmedian = cond["conditional_residual_median_mae"]
    cstd = cond["conditional_residual_std_mae"]
    corr_diff = metrics["backbone_output_residual_corr_abs_diff"]

    terms = [
        1.00 * metrics["residual_wasserstein_norm"],
        0.40 * metrics["backbone_output_wasserstein_norm"],
        0.50 * metrics["residual_ks"],
        0.25 * metrics["backbone_output_ks"],
        0.35 * metrics["residual_js"],
        0.15 * metrics["backbone_output_js"],
    ]

    if np.isfinite(cmean):
        terms.append(0.80 * cmean / residual_scale)
    if np.isfinite(cmedian):
        terms.append(0.60 * cmedian / residual_scale)
    if np.isfinite(cstd):
        terms.append(0.30 * cstd / residual_scale)
    if np.isfinite(corr_diff):
        terms.append(0.30 * corr_diff)

    metrics["composite_score"] = float(np.sum(terms))
    return metrics


def kde_line(ax, values: np.ndarray, x_grid: np.ndarray, color: str, lw=2.0, ls="-"):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.std(values) <= 0:
        return
    try:
        kde = gaussian_kde(values)
        ax.plot(x_grid, kde(x_grid), color=color, linewidth=lw, linestyle=ls)
    except Exception:
        pass


def plot_distribution_overlay(
    original_a: np.ndarray,
    adjusted_a: np.ndarray,
    b: np.ndarray,
    xlabel: str,
    title: str,
    out_path: Path,
    hist_bins: int,
    dpi: int,
):
    all_x = np.concatenate([original_a, adjusted_a, b])
    lo = float(np.percentile(all_x, 0.25))
    hi = float(np.percentile(all_x, 99.75))
    if lo >= hi:
        lo, hi = float(np.min(all_x)), float(np.max(all_x))

    edges = np.linspace(lo, hi, hist_bins + 1)
    x_grid = np.linspace(lo, hi, 800)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    ax.hist(
        adjusted_a[(adjusted_a >= lo) & (adjusted_a <= hi)],
        bins=edges,
        density=True,
        alpha=0.38,
        color=PURPLE_LIGHT,
        edgecolor="white",
        linewidth=0.45,
        label="Adjusted A",
    )
    ax.hist(
        b[(b >= lo) & (b <= hi)],
        bins=edges,
        density=True,
        alpha=0.35,
        color=GRAY_PURPLE,
        edgecolor="white",
        linewidth=0.45,
        label="Target B",
    )

    kde_line(ax, original_a, x_grid, GRAY, lw=1.5, ls="--")
    kde_line(ax, adjusted_a, x_grid, PURPLE_DARK, lw=2.3)
    kde_line(ax, b, x_grid, GRAY_DARK, lw=2.3)

    ax.set_xlim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_joint(
    backbone_output_a0: np.ndarray,
    residual_a0: np.ndarray,
    backbone_output_a1: np.ndarray,
    residual_a1: np.ndarray,
    backbone_output_b: np.ndarray,
    residual_b: np.ndarray,
    out_path: Path,
    seed: int,
    dpi: int,
):
    rng = np.random.default_rng(seed)

    def sample_xy(x, y, n=5000):
        if len(x) <= n:
            return x, y
        idx = rng.choice(len(x), size=n, replace=False)
        return x[idx], y[idx]

    xa0, ya0 = sample_xy(backbone_output_a0, residual_a0)
    xa1, ya1 = sample_xy(backbone_output_a1, residual_a1)
    xb, yb = sample_xy(backbone_output_b, residual_b)

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    ax.scatter(xa0, ya0, s=8, alpha=0.08, color=GRAY, label="Original A")
    ax.scatter(xb, yb, s=9, alpha=0.16, color=GRAY_DARK, label="Target B")
    ax.scatter(xa1, ya1, s=9, alpha=0.16, color=PURPLE, label="Adjusted A")
    ax.set_xlabel("Backbone Output")
    ax.set_ylabel("residual")
    ax.set_title("Joint Distribution: Backbone Output vs Residual")
    ax.grid(alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def conditional_curve(backbone_output, residual, edges):
    ids = np.digitize(backbone_output, edges[1:-1], right=False)
    x, mean, med, p25, p75 = [], [], [], [], []
    for i in range(len(edges) - 1):
        mask = ids == i
        if np.sum(mask) < 3:
            continue
        b = backbone_output[mask]
        k = residual[mask]
        x.append(float(np.median(b)))
        mean.append(float(np.mean(k)))
        med.append(float(np.median(k)))
        p25.append(float(np.percentile(k, 25)))
        p75.append(float(np.percentile(k, 75)))
    return map(np.asarray, (x, mean, med, p25, p75))


def plot_residual_along_backbone_output(
    backbone_output_a0,
    residual_a0,
    backbone_output_a1,
    residual_a1,
    backbone_output_b,
    residual_b,
    edges,
    out_path,
    dpi,
):
    x0, _, m0, _, _ = conditional_curve(backbone_output_a0, residual_a0, edges)
    x1, _, m1, q11, q31 = conditional_curve(backbone_output_a1, residual_a1, edges)
    xb, _, mb, q1b, q3b = conditional_curve(backbone_output_b, residual_b, edges)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    if len(x0):
        ax.plot(x0, m0, color=GRAY, linestyle="--", linewidth=1.7, marker="o", ms=4, label="Original A median")
    if len(x1):
        ax.plot(x1, m1, color=PURPLE_DARK, linewidth=2.2, marker="o", ms=4.5, label="Adjusted A median")
        ax.fill_between(x1, q11, q31, color=PURPLE_LIGHT, alpha=0.22)
    if len(xb):
        ax.plot(xb, mb, color=GRAY_DARK, linewidth=2.2, marker="o", ms=4.5, label="Target B median")
        ax.fill_between(xb, q1b, q3b, color=GRAY_PURPLE, alpha=0.16)

    ax.axhline(0, color=GRAY_LIGHT, linewidth=1.0)
    ax.set_xlabel("Backbone Output")
    ax.set_ylabel("residual")
    ax.set_title("Conditional Distribution of Residual Along Backbone Output")
    ax.grid(alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_delta(delta: np.ndarray, out_path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.hist(delta, bins=55, density=True, alpha=0.48, color=PURPLE_LIGHT, edgecolor="white", linewidth=0.45)
    lo, hi = np.percentile(delta, [0.25, 99.75])
    if lo < hi:
        x = np.linspace(lo, hi, 600)
        kde_line(ax, delta, x, PURPLE_DARK, lw=2.2)
        ax.set_xlim(lo, hi)
    ax.axvline(float(np.mean(delta)), color=GRAY_DARK, linestyle="--", linewidth=1.5, label=f"Mean={np.mean(delta):.3f}")
    ax.set_xlabel("Applied Shared Perturbation δ")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Applied Shared Perturbation")
    ax.grid(axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    csv_a = Path(args.csv_a).expanduser().resolve()
    csv_b = Path(args.csv_b).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_a.exists():
        raise FileNotFoundError(f"CSV A not found: {csv_a}")
    if not csv_b.exists():
        raise FileNotFoundError(f"CSV B not found: {csv_b}")
    if args.d_bins < 2:
        raise ValueError("--d_bins must be >= 2")
    if args.backbone_output_bins < 2:
        raise ValueError("--backbone_output_bins must be >= 2")
    if args.blend_steps < 2:
        raise ValueError("--blend_steps must be >= 2")
    if args.strength_steps < 1:
        raise ValueError("--strength_steps must be >= 1")
    if args.strength_min > args.strength_max:
        raise ValueError("--strength_min must be <= --strength_max")

    df_a = pd.read_csv(csv_a, low_memory=False)
    df_b = pd.read_csv(csv_b, low_memory=False)
    require_columns(df_a, [args.backbone_output_col, args.target_col], "CSV A")
    require_columns(df_b, [args.backbone_output_col, args.target_col], "CSV B")

    backbone_output_a_all, residual_a_all, target_a_all, valid_a = numeric_pair(
        df_a, args.backbone_output_col, args.target_col
    )
    backbone_output_b_all, residual_b_all, target_b_all, valid_b = numeric_pair(
        df_b, args.backbone_output_col, args.target_col
    )

    backbone_output_a = backbone_output_a_all[valid_a]
    residual_a = residual_a_all[valid_a]
    target_a = target_a_all[valid_a]
    backbone_output_b = backbone_output_b_all[valid_b]
    residual_b = residual_b_all[valid_b]
    target_b = target_b_all[valid_b]

    if len(backbone_output_a) < 10:
        raise ValueError("CSV A has fewer than 10 valid backbone_output/query_target_value pairs.")
    if len(backbone_output_b) < 10:
        raise ValueError("CSV B has fewer than 10 valid backbone_output/query_target_value pairs.")

    print("=" * 88)
    print("Backbone Output / Residual Distribution Matching")
    print("=" * 88)
    print(f"CSV A (source): {csv_a}")
    print(f"CSV B (target): {csv_b}")
    print(f"backbone_output_col   : {args.backbone_output_col}")
    print(f"target_col : {args.target_col}")
    print("residual formula : residual = backbone_output_col - target_col")
    print(f"A rows: {len(df_a):,}, valid pair: {len(backbone_output_a):,}")
    print(f"B rows: {len(df_b):,}, valid pair: {len(backbone_output_b):,}")
    print()

    # ------------------------------------------------------------
    # 1. Target-distribution summary
    # ------------------------------------------------------------
    summary_b = macro_summary(backbone_output_b, residual_b, "B_target")
    summary_a0 = macro_summary(backbone_output_a, residual_a, "A_original")

    summary_b.to_csv(out_dir / "B_macro_summary.csv", index=False, encoding="utf-8-sig")

    backbone_output_edges = unique_quantile_edges(backbone_output_b, args.backbone_output_bins)
    b_backbone_output_summary = backbone_output_bin_summary(backbone_output_b, residual_b, backbone_output_edges, "B_target")
    b_backbone_output_summary.to_csv(out_dir / "B_backbone_output_bin_residual_summary.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 2. Shared-perturbation coordinates
    # ------------------------------------------------------------
    # Because residual = backbone_output - target, d = backbone_output - residual is exactly query_target_value.
    d_a = backbone_output_a - residual_a
    s_a = (backbone_output_a + residual_a) / 2.0
    d_b = backbone_output_b - residual_b
    s_b = (backbone_output_b + residual_b) / 2.0

    # Joint/conditional proposal: match s|d
    s_cond_target, d_edges = conditional_s_quantile_map(
        d_a=d_a,
        s_a=s_a,
        d_b=d_b,
        s_b=s_b,
        n_bins=args.d_bins,
        min_bin_samples=args.min_bin_samples,
    )
    delta_cond = s_cond_target - s_a

    # Marginal residual proposal: directly rank-map residual A -> residual B.
    residual_marg_target = quantile_map(residual_a, residual_b)
    delta_marg = residual_marg_target - residual_a

    # ------------------------------------------------------------
    # 3. Grid search over blend and strength
    # ------------------------------------------------------------
    before_metrics = distribution_metrics(
        backbone_output_a, residual_a, backbone_output_b, residual_b, backbone_output_edges
    )

    blend_values = np.linspace(0.0, 1.0, args.blend_steps)
    strength_values = np.linspace(
        args.strength_min,
        args.strength_max,
        args.strength_steps,
    )

    search_rows = []
    best = None

    print("Searching matching parameters...")

    for blend in blend_values:
        # blend=0 -> purely conditional joint match
        # blend=1 -> purely residual marginal match
        raw_delta = (1.0 - blend) * delta_cond + blend * delta_marg

        for strength in strength_values:
            delta = strength * raw_delta
            if args.max_abs_delta is not None:
                delta = np.clip(delta, -args.max_abs_delta, args.max_abs_delta)

            ba = backbone_output_a + delta
            ka = residual_a + delta

            metrics = distribution_metrics(ba, ka, backbone_output_b, residual_b, backbone_output_edges)
            row = {
                "blend_weight_residual_marginal": float(blend),
                "strength": float(strength),
                "delta_mean": float(np.mean(delta)),
                "delta_std": float(np.std(delta)),
                "delta_mean_abs": float(np.mean(np.abs(delta))),
                "delta_p95_abs": float(np.percentile(np.abs(delta), 95)),
                **metrics,
            }
            search_rows.append(row)

            if best is None or row["composite_score"] < best["composite_score"]:
                best = row.copy()

    search_df = pd.DataFrame(search_rows).sort_values("composite_score")
    search_df.to_csv(out_dir / "search_scores.csv", index=False, encoding="utf-8-sig")

    best_blend = float(best["blend_weight_residual_marginal"])
    best_strength = float(best["strength"])

    raw_delta = (1.0 - best_blend) * delta_cond + best_blend * delta_marg
    delta = best_strength * raw_delta
    if args.max_abs_delta is not None:
        delta = np.clip(delta, -args.max_abs_delta, args.max_abs_delta)

    backbone_output_a_new = backbone_output_a + delta
    residual_a_new = residual_a + delta

    # Optional clipping.
    # Only backbone_output is clipped. residual is then re-derived from the unchanged query target,
    # which guarantees residual = backbone_output - query_target_value exactly.
    if args.clip_to_b_range:
        b_backbone_output_lo, b_backbone_output_hi = np.percentile(backbone_output_b, [0.1, 99.9])
        backbone_output_a_new = np.clip(backbone_output_a_new, b_backbone_output_lo, b_backbone_output_hi)
        residual_a_new = backbone_output_a_new - target_a
        delta = backbone_output_a_new - backbone_output_a

    after_metrics = distribution_metrics(
        backbone_output_a_new, residual_a_new, backbone_output_b, residual_b, backbone_output_edges
    )

    # ------------------------------------------------------------
    # 4. Export adjusted CSV A while keeping all other columns unchanged
    # ------------------------------------------------------------
    df_out = df_a.copy()
    valid_indices = np.flatnonzero(valid_a)
    df_out.loc[valid_indices, args.backbone_output_col] = backbone_output_a_new
    # query_target_value is intentionally unchanged.
    # residual is an internal derived quantity:
    #     residual = adjusted_backbone_output - query_target_value

    adjusted_csv = out_dir / "adjusted_A.csv"
    df_out.to_csv(adjusted_csv, index=False, encoding="utf-8-sig")

    # Separate audit file; do not pollute adjusted_A columns.
    audit = pd.DataFrame(
        {
            "original_row_index": valid_indices,
            "backbone_output_original": backbone_output_a,
            "query_target_value": target_a,
            "residual_original_derived": residual_a,
            "shared_delta": delta,
            "backbone_output_adjusted": backbone_output_a_new,
            "residual_adjusted_derived": residual_a_new,
            "invariant_before_backbone_output_minus_residual": backbone_output_a - residual_a,
            "invariant_after_backbone_output_minus_residual": backbone_output_a_new - residual_a_new,
        }
    )
    audit["invariant_change"] = (
        audit["invariant_after_backbone_output_minus_residual"]
        - audit["invariant_before_backbone_output_minus_residual"]
    )
    audit.to_csv(out_dir / "perturbation_audit.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 5. Before/after macro + metrics
    # ------------------------------------------------------------
    summary_a1 = macro_summary(backbone_output_a_new, residual_a_new, "A_adjusted")
    macro_all = pd.concat([summary_a0, summary_a1, summary_b], ignore_index=True)
    macro_all.to_csv(out_dir / "macro_summary_all.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    for stage, m in [("A_original_vs_B", before_metrics), ("A_adjusted_vs_B", after_metrics)]:
        metric_rows.append({"comparison": stage, **m})
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(out_dir / "comparison_metrics.csv", index=False, encoding="utf-8-sig")

    # B / adjusted A conditional summaries on identical B backbone_output bins.
    cond_all = pd.concat(
        [
            backbone_output_bin_summary(backbone_output_b, residual_b, backbone_output_edges, "B_target"),
            backbone_output_bin_summary(backbone_output_a, residual_a, backbone_output_edges, "A_original"),
            backbone_output_bin_summary(backbone_output_a_new, residual_a_new, backbone_output_edges, "A_adjusted"),
        ],
        ignore_index=True,
    )
    cond_all.to_csv(out_dir / "backbone_output_bin_residual_comparison.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------
    plot_distribution_overlay(
        backbone_output_a,
        backbone_output_a_new,
        backbone_output_b,
        xlabel="Backbone Output",
        title="Backbone Output Distribution",
        out_path=out_dir / "figure1_backbone_output_distribution.png",
        hist_bins=args.hist_bins,
        dpi=args.dpi,
    )

    plot_distribution_overlay(
        residual_a,
        residual_a_new,
        residual_b,
        xlabel="residual",
        title="Residual Distribution",
        out_path=out_dir / "figure2_residual_distribution.png",
        hist_bins=args.hist_bins,
        dpi=args.dpi,
    )

    plot_joint(
        backbone_output_a,
        residual_a,
        backbone_output_a_new,
        residual_a_new,
        backbone_output_b,
        residual_b,
        out_path=out_dir / "figure3_joint_distribution.png",
        seed=args.seed,
        dpi=args.dpi,
    )

    plot_residual_along_backbone_output(
        backbone_output_a,
        residual_a,
        backbone_output_a_new,
        residual_a_new,
        backbone_output_b,
        residual_b,
        backbone_output_edges,
        out_path=out_dir / "figure4_residual_along_backbone_output.png",
        dpi=args.dpi,
    )

    plot_delta(delta, out_dir / "figure5_delta_distribution.png", args.dpi)

    # ------------------------------------------------------------
    # 7. Text report
    # ------------------------------------------------------------
    invariant_max_change = float(np.max(np.abs((backbone_output_a_new - residual_a_new) - (backbone_output_a - residual_a))))
    report = []
    report.append("Backbone Output / Residual Distribution Matching report")
    report.append("=" * 72)
    report.append(f"CSV A source : {csv_a}")
    report.append(f"CSV B target : {csv_b}")
    report.append(f"A valid pairs: {len(backbone_output_a):,}")
    report.append(f"B valid pairs: {len(backbone_output_b):,}")
    report.append("")
    report.append("Best matching parameters")
    report.append("-" * 72)
    report.append(f"blend_weight_residual_marginal = {best_blend:.6f}")
    report.append(f"strength                 = {best_strength:.6f}")
    report.append(f"delta mean               = {np.mean(delta):.6f}")
    report.append(f"delta std                = {np.std(delta):.6f}")
    report.append(f"delta mean abs           = {np.mean(np.abs(delta)):.6f}")
    report.append(f"delta p95 abs            = {np.percentile(np.abs(delta),95):.6f}")
    report.append(f"max invariant change     = {invariant_max_change:.12e}")
    report.append("")
    report.append("Distribution metrics: before -> after (smaller is better)")
    report.append("-" * 72)
    for key in [
        "composite_score",
        "residual_wasserstein",
        "backbone_output_wasserstein",
        "residual_ks",
        "backbone_output_ks",
        "residual_js",
        "backbone_output_js",
        "residual_quantile_mae",
        "backbone_output_quantile_mae",
        "conditional_residual_mean_mae",
        "conditional_residual_median_mae",
        "conditional_residual_std_mae",
        "backbone_output_residual_corr_abs_diff",
    ]:
        b0 = before_metrics.get(key, np.nan)
        b1 = after_metrics.get(key, np.nan)
        report.append(f"{key:32s}: {b0:.6f} -> {b1:.6f}")

    report.append("")
    report.append("Important constraint")
    report.append("-" * 72)
    report.append(
        "residual is not read from CSV. It is derived as "
        "backbone_output_col - query_target_value. query_target_value is never modified. "
        "Therefore backbone_output-residual == query_target_value is an exact invariant. "
        "If A and B have very different query_target_value distributions, no method "
        "under this constraint can make their full 2D backbone_output/residual distributions identical; "
        "the script finds a best-effort match."
    )

    (out_dir / "matching_report.txt").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "best_parameters.json").write_text(
        json.dumps(
            {
                "backbone_output_col": args.backbone_output_col,
                "target_col": args.target_col,
                "residual_formula": "residual = backbone_output_col - target_col",
                "blend_weight_residual_marginal": best_blend,
                "strength": best_strength,
                "max_abs_delta": args.max_abs_delta,
                "d_bins": args.d_bins,
                "backbone_output_bins": args.backbone_output_bins,
                "before_composite_score": before_metrics["composite_score"],
                "after_composite_score": after_metrics["composite_score"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Best parameters")
    print(f"  blend weight : {best_blend:.4f}")
    print(f"  strength     : {best_strength:.4f}")
    print(f"  score before : {before_metrics['composite_score']:.6f}")
    print(f"  score after  : {after_metrics['composite_score']:.6f}")
    print(f"  mean |delta| : {np.mean(np.abs(delta)):.6f}")
    print(f"  max invariant change: {invariant_max_change:.3e}")
    print()
    print(f"Adjusted CSV: {adjusted_csv}")
    print(f"Output dir  : {out_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
