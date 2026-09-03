"""`equity_flag_thresholds` + `equity_flag_from_thresholds` must reproduce `equity_flag` exactly.

Split out of `equity_flag` so a chunked pipeline can compute the five
population-weighted-median scalars ONCE from a whole-city (but geometry-free,
cheap) scan, then apply them independently per tile -- see
`code.pipeline.build_h3_by_resolution_chunked`. This is the decomposition
that path relies on for `equity_flag` to come out identical to the unchunked
`build_h3_by_resolution`'s.
"""

import numpy as np

from code.stats import equity_flag, equity_flag_from_thresholds, equity_flag_thresholds


def test_thresholds_decomposition_matches_equity_flag_whole_city():
    rng = np.random.default_rng(1)
    n = 500
    density = rng.exponential(scale=200, size=n)
    density[rng.choice(n, 20, replace=False)] = np.nan
    los = rng.uniform(0, 1, n)
    population = rng.uniform(1, 1000, n)

    expected = equity_flag(density, los, population)

    thresholds = equity_flag_thresholds(density, los, population)
    actual = equity_flag_from_thresholds(density, los, population, thresholds)

    assert expected.tolist() == actual.tolist()


def test_thresholds_decomposition_matches_when_split_into_tiles():
    """The whole point: computing thresholds globally then applying per-slice must equal the whole-city flag."""
    rng = np.random.default_rng(2)
    n = 400
    density = rng.exponential(scale=150, size=n)
    los = rng.uniform(0, 1, n)
    population = rng.uniform(1, 500, n)

    expected = equity_flag(density, los, population)
    thresholds = equity_flag_thresholds(density, los, population)

    # Split into 5 uneven "tiles" and apply the SAME global thresholds to each independently.
    boundaries = [0, 37, 112, 240, 355, n]
    actual = np.empty(n, dtype=object)
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        actual[lo:hi] = equity_flag_from_thresholds(density[lo:hi], los[lo:hi], population[lo:hi], thresholds)

    assert expected.tolist() == actual.tolist()


def test_thresholds_none_for_tiny_input():
    density = np.array([1.0, 2.0, np.nan])
    los = np.array([0.5, 0.6, 0.7])
    population = np.array([10.0, 20.0, 30.0])
    assert equity_flag_thresholds(density, los, population) is None
    flags = equity_flag_from_thresholds(density, los, population, None)
    assert flags.tolist() == [None, None, None]
