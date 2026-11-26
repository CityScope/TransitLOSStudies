"""Tests of `flexfit.SmoothModel`, the model of the city science ranking: modes in order, smooth bounded curves, and the unit's score as the
population-weighted mean of its cells' scores."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studytools import flexfit as ff  # noqa: E402


def _tiny():
    """One unit of three cells: a bus stop 100 m from cell 0, a bus and a rail stop near cell 1, no stop near cell 2."""
    it_cell = np.array([0, 1, 1])
    it_mode = np.array([0, 0, 2])
    it_h = np.array([10.0, 6.0, 4.0])
    it_v = np.array([20.0, 18.0, 35.0])
    it_d = np.array([100.0, 300.0, 500.0])
    return ff.FitData(it_cell, it_mode, it_h, it_v, it_d, np.zeros(3, np.int64), np.array([100.0, 300.0, 600.0]), np.array([30.0]), np.array([100.0]),
                      np.array([1000.0]), np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros((1, 0)), ["city"], [], np.array(["u"]))


@pytest.mark.parametrize("a,b", [(-6.0, -6.0), (0.0, 0.0), (3.0, -2.0), (8.0, 8.0), (-2.0, 5.0)])
def test_rail_is_never_below_tram_and_tram_never_below_bus(a, b):
    m = ff.SmoothModel(1)
    with torch.no_grad():
        m.lmode.copy_(torch.tensor([a, b]))
    bus, tram, rail = torch.exp(m.lm()).tolist()
    assert rail == 1.0
    assert 0 < bus <= tram <= rail


def test_no_mode_stage_has_every_factor_equal_to_one():
    assert torch.exp(ff.SmoothModel(1, no_mode=True).lm()).tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("raw", [-40.0, 0.0, 40.0])
def test_curves_are_monotone_and_steepness_stays_in_its_range(raw):
    m = ff.SmoothModel(1)
    with torch.no_grad():
        m.xh.fill_(raw); m.xv.fill_(raw); m.xw.fill_(raw)
    assert ff.KH_RANGE[0] <= float(m.kh()) <= ff.KH_RANGE[1]
    assert ff.KV_RANGE[0] <= float(m.kv()) <= ff.KV_RANGE[1]
    assert ff.KW_RANGE[0] <= float(m.kw()) <= ff.KW_RANGE[1]
    h = torch.tensor([1.0, 3.0, 5.0, 10.0, 20.0, 60.0, 180.0])
    v = torch.tensor([2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    d = torch.tensor([10.0, m.floor_m, 300.0, 800.0, 1500.0, 3000.0])
    H, S, W = torch.exp(m.logH(h)), torch.exp(m.logS(v)), torch.exp(m.logW(d))
    assert bool((H[1:] <= H[:-1] + 1e-6).all()) and bool((S[1:] >= S[:-1] - 1e-6).all()) and bool((W[1:] <= W[:-1] + 1e-6).all())
    assert float(H.max()) <= 1 + 1e-6 and float(S.max()) <= 1 + 1e-6 and float(W.max()) <= 1 + 1e-6
    assert float(H[0]) == pytest.approx(float(H[1]))   # flat below the minimum headway of 3 minutes
    assert float(W[0]) == pytest.approx(float(W[1]))   # flat below the walking floor


def test_unit_service_term_is_the_population_weighted_mean_of_the_cell_scores():
    t = _tiny().tensors("cpu")
    m = ff.SmoothModel(1)
    with torch.no_grad():
        cell_score = m.cell_score(t)
        unit = m.unit_access(t)
    expected = float((cell_score * t.cell_pop).sum() / t.cell_pop.sum())
    assert float(cell_score[2]) == 0.0   # no stop near cell 2
    assert float(unit[0]) == pytest.approx(expected, rel=1e-5)
    # the mean of the scores, not the score of the mean access (the log makes them differ)
    with torch.no_grad():
        mean_access = (m.cell_access(t) * t.cell_pop).sum() / t.cell_pop.sum()
        log_of_mean = float(torch.log1p(mean_access / torch.exp(m.leps)))
    assert abs(log_of_mean - expected) > 1e-3


def test_speed_first_then_mode_stage_two_keeps_the_speed_curve():
    d = _tiny()
    t = d.tensors("cpu")
    w = torch.ones(1, dtype=torch.float64)
    m, losses = ff.fit_sequence_smooth(t, w, None, 0, starts=1, steps=5)
    assert set(losses) == {"stage1", "stage2"}
    for n in ff.SMOOTH_SPEED_PARAMS:
        assert not getattr(m, n).requires_grad   # the speed curve was frozen in the stage that fitted the modes
