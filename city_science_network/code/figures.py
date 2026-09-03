"""Matplotlib JPG figure helpers: regression scatter, ANOVA bar, % area bar.

Each function saves one self-contained JPG and returns the saved path. No
figure is left open (`plt.close(fig)` always called) so repeated calls
across many cities/variables don't leak matplotlib state.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .stats import RegressionResult


def regression_scatter(
    x: np.ndarray,
    y: np.ndarray,
    result: RegressionResult,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> str:
    """Scatter plot of `x` vs `y` with the fitted regression line.

    Args:
        x: Independent variable values.
        y: Dependent variable values.
        result: Fit from `code.stats.linreg`.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title (includes R²/p-value automatically).
        out_path: Destination `.jpg` path (parent directory created).

    Returns:
        `out_path`.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.scatter(x, y, s=6, alpha=0.35, color="#3366cc")
    if np.isfinite(result.slope):
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xs, result.slope * xs + result.intercept, color="#cc3333", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nR²={result.r2:.3f}, p={result.p_value:.3g}, n={result.n}")
    fig.tight_layout()
    fig.savefig(out_path, format="jpg")
    plt.close(fig)
    return out_path


def anova_grid(anovas: dict[str, dict], score_label: str, title: str, out_path: str) -> str:
    """One figure with a low/high group-mean bar panel per median-split variable.

    Args:
        anovas: Mapping of split-variable label (e.g. `"pop density"`,
            `"renter_share"`) to its `code.stats.median_split_anova` result
            dict. All variables for a city are drawn as side-by-side panels
            in a single saved figure, rather than one file per variable.
        score_label: Y-axis label shared by every panel (the compared score).
        title: Figure-level title.
        out_path: Destination `.jpg` path (parent directory created).

    Returns:
        `out_path`.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = len(anovas)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2), dpi=150, squeeze=False)
    for ax, (split_label, anova) in zip(axes[0], anovas.items()):
        means = [anova["low_mean"], anova["high_mean"]]
        labels = [f"low\n(n={anova['low_n']})", f"high\n(n={anova['high_n']})"]
        ax.bar(labels, means, color=["#66aacc", "#cc8844"])
        ax.set_ylabel(score_label)
        ax.set_title(f"{split_label}\nWelch t={anova['t_statistic']:.2f}, p={anova['p_value']:.3g}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, format="jpg")
    plt.close(fig)
    return out_path


def access_distribution_bar(
    dist: dict, overlay_label: Optional[str], title: str, out_path: str
) -> str:
    """Horizontal population-by-access-bin bar chart (from `code.stats.access_distribution`).

    One bar per bin showing summed population (and, if `dist["overlay"]` is
    set, a second bar for the overlay count column) with each bar's total
    and % of its own column's total, plus the population- (and overlay-)
    weighted mean level_of_service in the title.

    Args:
        dist: Result of `code.stats.access_distribution`.
        overlay_label: Display name of the overlay column, or `None`.
        title: Plot title prefix (weighted-access values are appended).
        out_path: Destination `.jpg` path (parent directory created).

    Returns:
        `out_path`.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = dist["bin_labels"]
    y = np.arange(len(labels))
    has_overlay = dist["overlay"] is not None
    height = 0.38 if has_overlay else 0.6

    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=150)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e3e6ea", linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # Faint alternating row bands make it easy to trace a bin across to its
    # label even with the grid off, without adding visual weight.
    for i in range(len(labels)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f5f6f8", zorder=0)

    bars_pop = ax.barh(
        y + (height / 2 if has_overlay else 0), dist["population"], height=height,
        color="#3a6ea8", label="population", zorder=3,
    )
    for bar, total, pct in zip(bars_pop, dist["population"], dist["population_pct"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {total:,.0f} ({pct:.1f}%)",
                va="center", fontsize=7.5, color="#1f3a5a")

    if has_overlay:
        bars_ov = ax.barh(
            y - height / 2, dist["overlay"], height=height, color="#d68a3d", label=overlay_label, zorder=3,
        )
        for bar, total, pct in zip(bars_ov, dist["overlay"], dist["overlay_pct"]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {total:,.0f} ({pct:.1f}%)",
                    va="center", fontsize=7.5, color="#8a5219")
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("population")
    ax.set_ylabel("level_of_service bin")
    ax.margins(x=0.12)
    subtitle = f"weighted access={dist['weighted_access']:.3f}"
    if dist.get("weighted_access_overlay") is not None:
        subtitle += f"  |  weighted access ({overlay_label})={dist['weighted_access_overlay']:.3f}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, format="jpg")
    plt.close(fig)
    return out_path


def anova_diff_bar(anovas: dict[str, dict], score_label: str, title: str, out_path: str) -> str:
    """Tornado-style bar chart: population-weighted access difference (high - low group) per variable.

    Args:
        anovas: Mapping of variable label -> `code.stats.weighted_median_split_anova`
            result dict (must include `diff`). All variables are drawn on
            one horizontal bar plot, ordered top-to-bottom by `diff`
            descending, negative values extending left of zero and positive
            right -- so the most access-favorable split is at the top.
        score_label: X-axis label (the compared, weighted score's name).
        title: Plot title.
        out_path: Destination `.jpg` path (parent directory created).

    Returns:
        `out_path`.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    items = sorted(
        anovas.items(),
        key=lambda kv: abs(kv[1]["diff"]) if np.isfinite(kv[1]["diff"]) else -np.inf,
        reverse=True,
    )
    labels = [k for k, _ in items]
    diffs = [v["diff"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.45 * len(labels))), dpi=150)
    y = np.arange(len(labels))[::-1]  # biggest diff at top
    colors = ["#cc3333" if d < 0 else "#33aa55" for d in diffs]
    ax.barh(y, diffs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{score_label} difference (high group - low group, population-weighted)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, format="jpg")
    plt.close(fig)
    return out_path


def percent_area_bar(shares: dict[str, float], title: str, out_path: str) -> str:
    """Bar chart of the share of h3 cells falling into each category.

    Args:
        shares: Mapping of category label (e.g. `"more_transit"`,
            `"more_housing"`) to its fraction of h3 cells (0-1).
        title: Plot title.
        out_path: Destination `.jpg` path (parent directory created).

    Returns:
        `out_path`.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    labels = list(shares.keys())
    values = [shares[k] * 100 for k in labels]
    ax.bar(labels, values, color="#4477aa")
    ax.set_ylabel("% of h3 cells")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, format="jpg")
    plt.close(fig)
    return out_path
