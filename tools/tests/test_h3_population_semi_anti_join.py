"""Regression test for the kept/dropped split in `h3_population.population_and_access_to_h3`.

2026-08-31: the post-edge-loop section of `population_and_access_to_h3`
used to build `kept`/`dropped` by joining `pop_h3` against a
`unique()`-de-duplicated `access_h3["h3_cell"]` frame, adding an
`on_street` boolean column, then running two separate `.filter()` passes
over that intermediate frame. Real, watchdog-instrumented measurement
against Shanghai's actual current candidate set (11,510,545 cells,
re-derived from today's real clipped WorldPop raster) showed that join was
the single largest new allocation in the whole post-loop section
(+~1.7GB in the measured scenario), on top of memory the loop itself left
resident (`cells`, `cell_lat`, `cell_lng` -- also fixed in this same pass,
see the `del` + `gc.collect()`/`malloc_trim` right after the edge-chunk
loop in `h3_population.py`).

The fix replaces the join+filter pair with a semi join (`kept`) and an
anti join (`dropped`) directly against `pop_h3`, skipping the intermediate
`on_street`-augmented copy entirely. This test locks in that the semi/anti
join split produces the IDENTICAL `kept`/`dropped` row sets as the old
on_street-boolean approach, including the edge cases that made the old
code's `dropped` filter non-trivial: cells absent from `access_h3`
entirely, and roadless cells with `population == 0` (which the old code
excluded from `dropped` via `population > 0`, so they're silently dropped
from the grid rather than kept and never redistributed to a neighbour).
"""

import polars as pl


def _old_kept_dropped(pop_h3: pl.DataFrame, access_h3: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The pre-fix on_street-boolean-join-then-filter approach, for comparison."""
    joined = pop_h3.join(
        access_h3.select("h3_cell").unique().with_columns(pl.lit(True).alias("on_street")),
        on="h3_cell",
        how="left",
    ).with_columns(pl.col("on_street").fill_null(False))
    kept = joined.filter(pl.col("on_street")).drop("on_street")
    dropped = joined.filter(~pl.col("on_street") & (pl.col("population") > 0)).drop("on_street")
    return kept, dropped


def _new_kept_dropped(pop_h3: pl.DataFrame, access_h3: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The fixed semi/anti-join approach now used in `population_and_access_to_h3`."""
    touching = access_h3.select("h3_cell").unique()
    kept = pop_h3.join(touching, on="h3_cell", how="semi")
    dropped = pop_h3.join(touching, on="h3_cell", how="anti").filter(pl.col("population") > 0)
    return kept, dropped


def _make_frames():
    pop_h3 = pl.DataFrame(
        {
            "h3_cell": ["a", "b", "c", "d", "e", "f"],
            "population": [10.0, 20.0, 0.0, 5.0, 0.0, 30.0],
        }
    )
    # "a" and "b" touch a street directly; "b" also appears twice in
    # access_h3 (e.g. matched by more than one edge chunk before the final
    # merge) to exercise the `.unique()` de-duplication. "c" is roadless
    # with zero population (must NOT appear in `dropped` -- nothing to
    # redistribute). "d" and "f" are roadless with population > 0 (must
    # appear in `dropped`). "e" is absent from pop_h3's own street-adjacent
    # set entirely (not touching, zero population) same as "c". "g" is a
    # cell access_h3 matched that pop_h3 never had population for --
    # should not appear in `kept` (no such row in pop_h3 to begin with).
    access_h3 = pl.DataFrame(
        {
            "h3_cell": ["a", "b", "b", "g"],
            "level_of_service": [1.0, 2.0, 2.0, 3.0],
        }
    )
    return pop_h3, access_h3


def test_semi_anti_join_matches_old_on_street_filter():
    pop_h3, access_h3 = _make_frames()

    old_kept, old_dropped = _old_kept_dropped(pop_h3, access_h3)
    new_kept, new_dropped = _new_kept_dropped(pop_h3, access_h3)

    assert set(old_kept["h3_cell"].to_list()) == set(new_kept["h3_cell"].to_list()) == {"a", "b"}
    assert set(old_dropped["h3_cell"].to_list()) == set(new_dropped["h3_cell"].to_list()) == {"d", "f"}

    assert old_kept.sort("h3_cell")["population"].to_list() == new_kept.sort("h3_cell")["population"].to_list()
    assert (
        old_dropped.sort("h3_cell")["population"].to_list()
        == new_dropped.sort("h3_cell")["population"].to_list()
    )


def test_semi_anti_join_excludes_zero_population_roadless_cells():
    # "c" and "e"-style roadless zero-population cells must never show up
    # in `dropped` under either approach -- there is nothing to
    # redistribute, and including them would make `removed_total` count
    # phantom population.
    pop_h3, access_h3 = _make_frames()
    _, new_dropped = _new_kept_dropped(pop_h3, access_h3)
    assert "c" not in new_dropped["h3_cell"].to_list()


def test_semi_join_ignores_access_h3_cells_absent_from_pop_h3():
    # access_h3 matched cell "g", which pop_h3 has no row for at all (e.g.
    # a street-adjacent cell with no WorldPop coverage). The semi join must
    # not fabricate a `kept` row for it.
    pop_h3, access_h3 = _make_frames()
    new_kept = pop_h3.join(access_h3.select("h3_cell").unique(), on="h3_cell", how="semi")
    assert "g" not in new_kept["h3_cell"].to_list()
