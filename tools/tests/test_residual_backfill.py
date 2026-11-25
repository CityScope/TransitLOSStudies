"""`_residual_backfill`: a coarser census level fills the cells a finer level left empty without dumping the finer level's unplaced mass
into them (Toronto 2026-11-20: 1.6 M residents in Lake Ontario cells)."""
import numpy as np

from studytools.pipeline import _residual_backfill


def test_unplaced_finer_mass_does_not_go_to_empty_water_cells():
    cur = np.array([450.0, 450.0, np.nan, np.nan])   # land cells filled by the blocks; 100 of the polygon's 1000 were never placed
    out = _residual_backfill(cur, np.isnan(cur), np.arange(4), np.zeros(4, int), np.array([1000.0]), np.array([500.0, 500.0, 1.0, 1.0]))
    assert np.isnan(out[:2]).all()
    assert out[2:].sum() < 5.0                        # the water cells get their own weight share, not the 100


def test_column_with_no_finer_level_is_split_by_weight():
    cur = np.full(4, np.nan)
    out = _residual_backfill(cur, np.isnan(cur), np.arange(4), np.zeros(4, int), np.array([1000.0]), np.array([400.0, 400.0, 100.0, 100.0]))
    assert np.allclose(out, [400.0, 400.0, 100.0, 100.0])


def test_never_more_than_the_residual():
    cur = np.array([900.0, np.nan])                   # the finer level already placed 900 of 1000 in a cell holding half the weight
    out = _residual_backfill(cur, np.isnan(cur), np.arange(2), np.zeros(2, int), np.array([1000.0]), np.array([1.0, 1.0]))
    assert np.isclose(out[1], 100.0)
