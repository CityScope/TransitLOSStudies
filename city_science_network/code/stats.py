"""Regression and median-split ANOVA helpers for equity analysis.

New code -- no existing package implements these. Pure `numpy`/`scipy.stats`,
no I/O, operating on plain arrays or a `polars`/`pandas`-like DataFrame with
named columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class RegressionResult:
    """Result of a simple (optionally weighted) linear regression.

    Attributes:
        slope: Fitted slope.
        intercept: Fitted intercept.
        r2: Coefficient of determination.
        p_value: Two-sided p-value for the slope being non-zero.
        n: Number of finite observations used.
    """

    slope: float
    intercept: float
    r2: float
    p_value: float
    n: int


def discretize_score(scores: np.ndarray, bin_width: float = 5.0) -> np.ndarray:
    """Round scores onto a fixed grid of `bin_width`-wide bins (e.g. 20 bins of 5 over [0, 100]).

    Args:
        scores: `level_of_service` values, expected in `[0, 100]` (2026-09-23:
            rescaled from the earlier `[0, 1]` at the source in
            `transitlos.level_of_service.compute_level_of_service`; NaNs
            pass through).
        bin_width: Width of each bin. Default 5.0 -> 20 bins covering [0, 100].

    Returns:
        Array of the same shape, each value snapped to the nearest multiple
        of `bin_width` and clipped to `[0, 100]` -- exactly `0.0, 5.0, 10.0,
        ...` (not float-noise; the extra `np.round(..., 10)` clears binary
        floating-point drift introduced by the division/multiplication
        without changing any actual bin).
    """
    snapped = np.round(scores / bin_width) * bin_width
    snapped = np.round(snapped, 10)
    return np.clip(snapped, 0.0, 100.0)


def linreg(x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> RegressionResult:
    """Fit `y ~ x`, optionally population-weighted.

    Args:
        x: Independent variable (e.g. population density, % transit use).
        y: Dependent variable (`level_of_service`).
        weights: Optional non-negative weights (e.g. population); if given,
            a weighted least-squares fit is used instead of
            `scipy.stats.linregress`.

    Returns:
        A `RegressionResult`. All fields are `nan`/`0` if fewer than 2
        finite paired observations remain.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        mask &= np.isfinite(w) & (w >= 0)
    x, y = x[mask], y[mask]

    if x.size < 2:
        return RegressionResult(float("nan"), float("nan"), float("nan"), float("nan"), int(x.size))

    if weights is None:
        fit = scipy_stats.linregress(x, y)
        return RegressionResult(float(fit.slope), float(fit.intercept), float(fit.rvalue**2), float(fit.pvalue), int(x.size))

    w = np.asarray(weights, dtype=float)[mask]
    W = np.sum(w)
    x_mean = np.sum(w * x) / W
    y_mean = np.sum(w * y) / W
    sxx = np.sum(w * (x - x_mean) ** 2)
    sxy = np.sum(w * (x - x_mean) * (y - y_mean))
    slope = sxy / sxx if sxx > 0 else float("nan")
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = np.sum(w * (y - y_pred) ** 2)
    ss_tot = np.sum(w * (y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n = x.size
    dof = max(n - 2, 1)
    mse = ss_res / dof
    se_slope = np.sqrt(mse / sxx) if sxx > 0 else float("nan")
    t_stat = slope / se_slope if se_slope not in (0.0, float("nan")) and se_slope > 0 else float("nan")
    p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), dof))) if np.isfinite(t_stat) else float("nan")

    return RegressionResult(float(slope), float(intercept), float(r2), p_value, int(n))


def median_split_anova(values: np.ndarray, split_col: np.ndarray, score_col: np.ndarray) -> dict:
    """Split rows into 2 groups by the median of `split_col`, compare `score_col` means.

    Args:
        values: Unused placeholder for API symmetry; ignored (kept so
            callers can pass the same positional style as `linreg`). Pass
            the same array as `split_col` if unsure.
        split_col: Column to split rows into "low"/"high" groups by its
            median (e.g. car ownership, renter share, income, population
            density).
        score_col: Column compared between the two groups (`level_of_service`).

    Returns:
        A dict with `median`, `low_mean`, `high_mean`, `low_n`, `high_n`,
        `t_statistic`, `p_value` (Welch's t-test, `scipy.stats.ttest_ind`
        with `equal_var=False`).
    """
    split_col = np.asarray(split_col, dtype=float)
    score_col = np.asarray(score_col, dtype=float)
    mask = np.isfinite(split_col) & np.isfinite(score_col)
    split_col, score_col = split_col[mask], score_col[mask]

    if split_col.size < 4:
        return {
            "median": float("nan"), "low_mean": float("nan"), "high_mean": float("nan"),
            "low_n": 0, "high_n": 0, "t_statistic": float("nan"), "p_value": float("nan"),
        }

    median = float(np.median(split_col))
    low_mask = split_col <= median
    low, high = score_col[low_mask], score_col[~low_mask]

    if low.size < 2 or high.size < 2:
        t_stat, p_value = float("nan"), float("nan")
    else:
        result = scipy_stats.ttest_ind(low, high, equal_var=False)
        t_stat, p_value = float(result.statistic), float(result.pvalue)

    return {
        "median": median,
        "low_mean": float(np.mean(low)) if low.size else float("nan"),
        "high_mean": float(np.mean(high)) if high.size else float("nan"),
        "low_n": int(low.size),
        "high_n": int(high.size),
        "t_statistic": t_stat,
        "p_value": p_value,
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted mean, ignoring non-finite/negative-weight rows."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not mask.any() or np.sum(weights[mask]) <= 0:
        return float("nan")
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted median: the value at which cumulative weight first reaches 50%."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[mask], weights[mask]
    if values.size == 0 or np.sum(weights) <= 0:
        return float("nan")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    cutoff = 0.5 * cum[-1]
    idx = int(np.searchsorted(cum, cutoff))
    idx = min(idx, values.size - 1)
    return float(values[idx])


def weighted_median_split_anova(split_col: np.ndarray, score_col: np.ndarray, weights: np.ndarray) -> dict:
    """Population-weighted median split: groups by `split_col`'s *weighted* median, compares *weighted* mean `score_col`.

    Per the study's spec, both the split point and the group comparison use
    population weighting (not the plain-median/plain-mean of
    `median_split_anova`) -- a handful of near-empty h3 cells with an
    extreme `split_col` value shouldn't move the split point or dominate a
    group's average the way an unweighted median/mean would let them.

    Args:
        split_col: Column to split rows into "low"/"high" groups by its
            population-weighted median.
        score_col: Column compared between the two groups (`level_of_service`).
        weights: Population (or other weight) per row.

    Returns:
        Dict with `median` (weighted), `low_mean`/`high_mean` (weighted),
        `low_n`/`high_n`, `diff` (`high_mean - low_mean`), `t_statistic`,
        `p_value` (unweighted Welch's t-test on the same two groups, as a
        significance indicator only -- the plotted quantity is the weighted
        `diff`).
    """
    split_col = np.asarray(split_col, dtype=float)
    score_col = np.asarray(score_col, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(split_col) & np.isfinite(score_col) & np.isfinite(weights) & (weights >= 0)
    split_col, score_col, weights = split_col[mask], score_col[mask], weights[mask]

    if split_col.size < 4:
        return {
            "median": float("nan"), "low_mean": float("nan"), "high_mean": float("nan"),
            "diff": float("nan"), "low_n": 0, "high_n": 0, "t_statistic": float("nan"), "p_value": float("nan"),
        }

    median = weighted_median(split_col, weights)
    low_mask = split_col <= median
    high_mask = ~low_mask

    low_mean = weighted_mean(score_col[low_mask], weights[low_mask])
    high_mean = weighted_mean(score_col[high_mask], weights[high_mask])

    low, high = score_col[low_mask], score_col[high_mask]
    if low.size < 2 or high.size < 2:
        t_stat, p_value = float("nan"), float("nan")
    else:
        result = scipy_stats.ttest_ind(low, high, equal_var=False)
        t_stat, p_value = float(result.statistic), float(result.pvalue)

    return {
        "median": median,
        "low_mean": low_mean,
        "high_mean": high_mean,
        "diff": high_mean - low_mean if np.isfinite(low_mean) and np.isfinite(high_mean) else float("nan"),
        "low_n": int(low.size),
        "high_n": int(high.size),
        "t_statistic": t_stat,
        "p_value": p_value,
    }


def development_priority_flag(
    density_var: np.ndarray,
    level_of_service: np.ndarray,
    population: np.ndarray,
) -> np.ndarray:
    """Boolean "dense and under-served" flag -- backs `equity_flag`'s `"more_transit"` category.

    This *is* the `"more_transit"` condition (`equity_flag` calls this
    function directly), not a separate concept: cells that are both unusually
    dense *within* the already-dense half of the study, and unusually
    under-served *relative to* the already-sparse half. All three splits use
    population-weighted medians (`weighted_median`), so a few near-empty
    extreme cells can't move a cutoff the way a plain median would let them.

    Steps (population-weighted throughout):
      1. `median_all` = weighted median of `density_var` over every row.
      2. Group A = rows with `density_var > median_all`; group B = the rest.
      3. `median_A` = weighted median of `density_var` within group A.
         `density_flag` = `density_var > median_A` (group A rows only).
      4. `median_B` = weighted median of `level_of_service` within group B.
         `access_flag` = `level_of_service < median_B`, applied to *every* row
         (not just group B) -- a group-A cell that is also worse-served than
         the typical group-B cell is exactly the pocket this flag exists to
         surface.
      5. the flag = `density_flag AND access_flag`.

    Args:
        density_var: Per-row density (population, or population+jobs where a
            jobs count exists -- see `development_density_column`).
        level_of_service: Per-row `level_of_service`.
        population: Per-row population weight.

    Returns:
        Boolean array, same shape as the inputs. Rows with a non-finite
        input, or a study too small to support the three-way split (<4
        finite rows, or an empty group A/B), are `False`.
    """
    density_var = np.asarray(density_var, dtype=float)
    level_of_service = np.asarray(level_of_service, dtype=float)
    population = np.asarray(population, dtype=float)
    result = np.zeros(density_var.shape, dtype=bool)

    finite = np.isfinite(density_var) & np.isfinite(level_of_service) & np.isfinite(population) & (population >= 0)
    if finite.sum() < 4:
        return result

    median_all = weighted_median(density_var[finite], population[finite])
    if not np.isfinite(median_all):
        return result

    group_a = finite & (density_var > median_all)
    group_b = finite & (density_var <= median_all)

    density_flag = np.zeros(density_var.shape, dtype=bool)
    if group_a.any():
        median_a = weighted_median(density_var[group_a], population[group_a])
        if np.isfinite(median_a):
            density_flag[group_a] = density_var[group_a] > median_a

    access_flag = np.zeros(density_var.shape, dtype=bool)
    if group_b.any():
        median_b = weighted_median(level_of_service[group_b], population[group_b])
        if np.isfinite(median_b):
            access_flag[finite] = level_of_service[finite] < median_b

    result[finite] = density_flag[finite] & access_flag[finite]
    return result


def housing_priority_flag(
    density_var: np.ndarray,
    level_of_service: np.ndarray,
    population: np.ndarray,
) -> np.ndarray:
    """Boolean "sparse but already well-served" flag -- backs `equity_flag`'s `"more_housing"` category.

    Mirror image of `development_priority_flag`, applied to the low-density
    side of the study instead of the high-density side. All three splits use
    population-weighted medians (`weighted_median`).

    Steps (population-weighted throughout):
      1. `median_all` = weighted median of `density_var` over every row.
      2. Group A = rows with `density_var > median_all`; group B = the rest
         (`density_var <= median_all`).
      3. `median_B` = weighted median of `density_var` within group B.
         `density_flag_low` = `density_var < median_B` (group B rows only,
         i.e. the bottom quarter by density).
      4. `median_A` = weighted median of `level_of_service` within group A.
         `access_flag_high` = `level_of_service > median_A`, applied to *every*
         row (not just group B) -- a group-B cell that is better-served than
         the typical group-A (dense) cell is exactly the pocket this flag
         exists to surface.
      5. the flag = `density_flag_low AND access_flag_high`.

    Args:
        density_var: Per-row density (population, or population+jobs where a
            jobs count exists -- see `development_density_column`).
        level_of_service: Per-row `level_of_service`.
        population: Per-row population weight.

    Returns:
        Boolean array, same shape as the inputs. Rows with a non-finite
        input, or a study too small to support the three-way split (<4
        finite rows, or an empty group A/B), are `False`.
    """
    density_var = np.asarray(density_var, dtype=float)
    level_of_service = np.asarray(level_of_service, dtype=float)
    population = np.asarray(population, dtype=float)
    result = np.zeros(density_var.shape, dtype=bool)

    finite = np.isfinite(density_var) & np.isfinite(level_of_service) & np.isfinite(population) & (population >= 0)
    if finite.sum() < 4:
        return result

    median_all = weighted_median(density_var[finite], population[finite])
    if not np.isfinite(median_all):
        return result

    group_a = finite & (density_var > median_all)
    group_b = finite & (density_var <= median_all)

    density_flag_low = np.zeros(density_var.shape, dtype=bool)
    if group_b.any():
        median_b = weighted_median(density_var[group_b], population[group_b])
        if np.isfinite(median_b):
            density_flag_low[group_b] = density_var[group_b] < median_b

    access_flag_high = np.zeros(density_var.shape, dtype=bool)
    if group_a.any():
        median_a = weighted_median(level_of_service[group_a], population[group_a])
        if np.isfinite(median_a):
            access_flag_high[finite] = level_of_service[finite] > median_a

    result[finite] = density_flag_low[finite] & access_flag_high[finite]
    return result


def access_distribution(
    scores: np.ndarray, weights: np.ndarray, overlay_weights: Optional[np.ndarray] = None
) -> dict:
    """Bin `scores` into `[0], (0,10], (10,20], ..., (90,100]` and sum `weights` per bin.

    2026-09-23: bin edges rescaled from the earlier `[0, 1]`-by-0.1 grid to
    `[0, 100]`-by-10, matching `level_of_service`'s new 0-100 scale
    (`transitlos.level_of_service.compute_level_of_service`).

    Args:
        scores: `level_of_service` values.
        weights: Population per row (the primary bar).
        overlay_weights: Optional second count column (e.g. `acs_workers_transit`)
            summed per the same bins, for a second overlaid bar.

    Returns:
        Dict with `bin_labels`, `population` (per-bin sum, and as a
        fraction of the total), `overlay` (same, or `None`), and
        `weighted_access`/`weighted_access_overlay` (population- and
        overlay-weighted mean level_of_service).
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    edges = [0.0, 1e-7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 + 1e-7]
    labels = ["0", "0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]

    mask = np.isfinite(scores) & np.isfinite(weights)
    bin_idx = np.digitize(scores[mask], edges[1:-1], right=True)
    pop_sums = np.zeros(len(labels))
    for i in range(len(labels)):
        pop_sums[i] = weights[mask][bin_idx == i].sum()
    total_pop = pop_sums.sum()

    result = {
        "bin_labels": labels,
        "population": pop_sums.tolist(),
        "population_pct": (pop_sums / total_pop * 100 if total_pop > 0 else pop_sums).tolist(),
        "weighted_access": weighted_mean(scores, weights),
        "overlay": None,
        "overlay_pct": None,
        "weighted_access_overlay": None,
    }
    if overlay_weights is not None:
        overlay_weights = np.asarray(overlay_weights, dtype=float)
        omask = np.isfinite(scores) & np.isfinite(overlay_weights)
        obin_idx = np.digitize(scores[omask], edges[1:-1], right=True)
        overlay_sums = np.zeros(len(labels))
        for i in range(len(labels)):
            overlay_sums[i] = overlay_weights[omask][obin_idx == i].sum()
        total_overlay = overlay_sums.sum()
        result["overlay"] = overlay_sums.tolist()
        result["overlay_pct"] = (overlay_sums / total_overlay * 100 if total_overlay > 0 else overlay_sums).tolist()
        result["weighted_access_overlay"] = weighted_mean(scores, overlay_weights)
    return result


def equity_flag(
    pop_density: np.ndarray,
    level_of_service: np.ndarray,
    population: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the map's orange/blue equity flag.

    2026-09-22, explicit user request ("more housing or more transit cells
    should be computed as follows: use the population density column per
    h3 res 7 cell... If the h3 cell transit level of service and population
    density value is above one std above the regression of all h3 res 7
    cells... of log population density (x axis) and transit level of
    service (y axis) then this cell is more housing and if it is more than
    one std below it is more transit") -- replaces the previous
    population-weighted two-step median-split method with a single OLS
    regression of `level_of_service ~ log(pop_density)` over every finite
    cell, then a residual-vs-1-std test:

    - `"more_housing"`: residual (real LOS minus the regression line's
      predicted LOS at that cell's own density) is MORE than one residual
      standard deviation ABOVE the line -- better served than its density
      alone predicts, a candidate for adding housing since the access is
      already there.
    - `"more_transit"`: residual is more than one standard deviation BELOW
      the line -- worse served than its density predicts, a candidate for
      adding transit.

    `population` is accepted (and still required by every call site) for
    API compatibility but no longer used -- the regression itself is
    unweighted, per the literal request (not "population-weighted
    regression").

    Returns:
        Object array of `"more_housing"`, `"more_transit"`, or `None`.
    """
    thresholds = equity_flag_thresholds(pop_density, level_of_service, population)
    return equity_flag_from_thresholds(pop_density, level_of_service, population, thresholds)


def equity_flag_thresholds(
    pop_density: np.ndarray, level_of_service: np.ndarray, population: Optional[np.ndarray] = None
) -> Optional[dict]:
    """`(slope, intercept, std_residual)` of the OLS fit `equity_flag` derives its split from.

    Split out so a chunked pipeline (see
    `code.pipeline.build_h3_by_resolution_chunked`) can fit this ONCE from a
    lazy, geometry-free, whole-city scan of just `(pop_density,
    level_of_service)` -- two float columns, cheap even at Shanghai's
    ~25M-row native resolution -- and then apply it independently to each
    tile with `equity_flag_from_thresholds`, rather than needing the whole
    city's rows physically together in one array. `population` is accepted
    for signature compatibility with every existing call site but unused
    (see `equity_flag`'s docstring). Verified to reproduce `equity_flag`
    exactly by `code.tests.test_equity_flag_thresholds`
    (`city_science_network/tests`) -- that test only checks the
    global-fit/apply-per-tile DECOMPOSITION property, not any particular
    method, so it stays valid across this regression-based rewrite.

    Returns `None` when there are fewer than 4 finite, positive-density
    rows, or the fit degenerates (zero residual spread) -- both "too small/
    too uniform to split" cases. Callers should treat `None` as "every
    row's flag is `None`" (`equity_flag_from_thresholds` does this itself).
    """
    density = np.asarray(pop_density, dtype=float)
    los = np.asarray(level_of_service, dtype=float)
    # 2026-09-25, explicit user request: fit the regression only on cells
    # with level_of_service ABOVE 0 -- a true zero (no access at all) is a
    # different kind of cell than "just poorly served," and pulling it into
    # the fit skews the slope/intercept (and therefore the std-residual
    # equity-flag thresholds) toward that floor. `equity_flag_from_thresholds`
    # still APPLIES the resulting fit to every cell including zero-LOS ones
    # -- only the FIT itself excludes them.
    finite = np.isfinite(density) & np.isfinite(los) & (density > 0) & (los > 0)
    if finite.sum() < 4:
        return None

    x = np.log(density[finite])
    y = los[finite]
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    std_residual = float(np.std(residuals))
    if not np.isfinite(std_residual) or std_residual == 0:
        return None

    return {"slope": float(slope), "intercept": float(intercept), "std_residual": std_residual}


def equity_flag_from_thresholds(
    pop_density: np.ndarray,
    level_of_service: np.ndarray,
    population: Optional[np.ndarray],
    thresholds: Optional[dict],
) -> np.ndarray:
    """Apply `equity_flag_thresholds`' regression fit to one (tile's worth of) rows.

    Pure per-row math against the three already-fit global scalars -- no
    row of this call needs to see any *other* row, which is exactly what
    makes this safe to run independently per H3 tile in
    `code.pipeline.build_h3_by_resolution_chunked` while still reproducing
    the whole-city `equity_flag` split exactly. `population` is accepted
    for signature compatibility but unused -- see `equity_flag`'s docstring.
    """
    density = np.asarray(pop_density, dtype=float)
    los = np.asarray(level_of_service, dtype=float)
    flags = np.full(density.shape, None, dtype=object)
    if thresholds is None:
        return flags

    finite = np.isfinite(density) & np.isfinite(los) & (density > 0)
    if not finite.any():
        return flags

    predicted = np.full(density.shape, np.nan)
    predicted[finite] = thresholds["slope"] * np.log(density[finite]) + thresholds["intercept"]
    residual = los - predicted
    std = thresholds["std_residual"]

    flags[finite & (residual > std)] = "more_housing"
    # 2026-09-29, explicit user request: "for more transit cells require
    # besides the current std things that the population density is above
    # 1000 pers/km2" -- a cell more than 1 std below the regression line is
    # still only flagged "more_transit" if it's also genuinely dense enough
    # to justify new transit investment; a sparse, under-served cell (e.g.
    # a rural edge of the study area) no longer qualifies. "more_housing"
    # is unaffected -- that flag identifies already-well-served cells as
    # housing-development candidates, where density isn't the gating
    # concern.
    flags[finite & (residual < -std) & (density > 1000)] = "more_transit"
    return flags
