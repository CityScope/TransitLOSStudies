"""Regression test for Gipuzkoa's census routing.

History: the original fix here made Gipuzkoa (`country="ESP"`) route to
`pycensus.countries.euskadi` (Basque Country's own Eustat statistics
office) instead of generic `pycensus.countries.spain`/INE, reasoning that
Euskadi's own agency would have richer/finer data for its own territory.

That was revisited 2026-08-22: investigation found `pycensus.countries
.euskadi` has no boundary source finer than `municipality` (Euskadi's own
geo-cartography portals are genuinely unreachable -- 403/404 -- to a
scripted client) and only 3 fields, all ultimately sourced from the same
GISCO/LAU 2021 municipal figures Spain/INE already exposes -- not
independent EUSTAT data. Generic `pycensus.countries.spain`, meanwhile, now
has a real, live section-level (seccion censal) source from INE's own OGC
API Features service covering Gipuzkoa (province code "20") with 544 real
sections and 3 real fields (population/malePopulation/femalePopulation),
live-verified against real published municipal totals for Donostia and
Irun (~99% match). Since Gipuzkoa *is* INE province "20", generic Spain/INE
gives it both finer resolution and the same-or-more real fields than the
Euskadi-specific module -- so `gipuzkoa`'s `CityConfig` no longer overrides
`census_module`, and it dispatches through the generic Spain/INE path like
any other Spanish city. `pycensus.countries.euskadi` remains implemented
(for any future non-Gipuzkoa Basque use, or if EUSTAT's own Data Bank API
-- real and reachable, see `euskadi/api.py` -- is later wired into a
loader with genuinely richer/finer verified data) but is no longer wired
into `city_science_network`.
"""

from __future__ import annotations

from code.city_config import CITY_CONFIGS, CityConfig, _census_supported
from code import pipeline as P


def test_gipuzkoa_routes_to_generic_spain():
    gipuzkoa = CITY_CONFIGS["gipuzkoa"]
    assert gipuzkoa.country == "ESP"
    assert gipuzkoa.census_module is None
    assert gipuzkoa.uses_census is True


def test_census_module_override_beats_country_mapping():
    # Without an override, country="ESP" resolves to pycensus.countries.spain.
    assert _census_supported("ESP", census_module=None) is True  # pycensus.spain exists
    assert _census_supported("ESP", census_module="euskadi") is True  # overridden, pycensus.euskadi exists
    assert _census_supported("ESP", census_module="does_not_exist") is False


def test_census_module_defaults_to_none_for_every_city():
    # No city (including gipuzkoa) should have an override into
    # pycensus.countries.euskadi any more -- see module docstring for why.
    for key, config in CITY_CONFIGS.items():
        assert config.census_module is None, f"{key} unexpectedly has census_module={config.census_module!r}"


def test_join_census_dispatches_generic_spain_for_gipuzkoa(monkeypatch, capsys):
    # Gipuzkoa's config carries no census_module override any more, so
    # _join_census must dispatch it through the same generic Spain/INE
    # branch as any other country="ESP" city -- section-level, "ine_" prefix.
    calls = {}

    def fake_join_polygon_stats(h3_grid, aoi, states, levels, cache_dir, loader, keep_columns, prefix):
        calls["prefix"] = prefix
        calls["keep_columns"] = keep_columns
        calls["levels"] = levels
        return h3_grid

    monkeypatch.setattr(P, "_join_polygon_stats", fake_join_polygon_stats)

    gipuzkoa = CITY_CONFIGS["gipuzkoa"]
    result = P._join_census(
        h3_grid="fake_h3_grid",
        aoi="fake_aoi",
        states=None,
        census_levels=("blockgroup",),
        cache_dir="fake_cache_dir",
        country=gipuzkoa.country,
        census_module=gipuzkoa.census_module,
    )

    assert result == "fake_h3_grid"
    assert calls["prefix"] == "ine_"
    assert calls["keep_columns"] == P.SPAIN_KEEP_COLUMNS
    assert calls["levels"] == P.SPAIN_LEVELS
    captured = capsys.readouterr()
    assert "not yet supported" not in captured.out


def test_join_census_still_dispatches_euskadi_when_explicitly_requested(monkeypatch, capsys):
    # pycensus.countries.euskadi is still implemented and reachable via an
    # explicit override, even though no CityConfig uses it any more.
    calls = {}

    def fake_join_polygon_stats(h3_grid, aoi, states, levels, cache_dir, loader, keep_columns, prefix):
        calls["prefix"] = prefix
        calls["keep_columns"] = keep_columns
        return h3_grid

    monkeypatch.setattr(P, "_join_polygon_stats", fake_join_polygon_stats)

    result = P._join_census(
        h3_grid="fake_h3_grid",
        aoi="fake_aoi",
        states=None,
        census_levels=("blockgroup",),
        cache_dir="fake_cache_dir",
        country="ESP",
        census_module="euskadi",
    )

    assert result == "fake_h3_grid"
    assert calls["prefix"] == "eustat_"
    assert calls["keep_columns"] == P.EUSKADI_KEEP_COLUMNS
    captured = capsys.readouterr()
    assert "not yet supported" not in captured.out


def test_join_census_dispatches_germany(monkeypatch, capsys):
    # Germany was wired 2026-08-21: pycensus.countries.germany.destatis's
    # real GENESIS-Online token was verified live to authenticate and
    # retrieve real data once the host (no "www-" prefix) and auth
    # transport (credentials as HTTP headers) were corrected -- see
    # pyCensus/src/pycensus/countries/germany/destatis/README.md. So
    # country="DEU" must now dispatch to the real Germany join instead of
    # the generic "not yet supported" skip.
    calls = {}

    def fake_join_polygon_stats(h3_grid, aoi, states, levels, cache_dir, loader, keep_columns, prefix):
        calls["prefix"] = prefix
        calls["keep_columns"] = keep_columns
        calls["levels"] = levels
        return h3_grid

    monkeypatch.setattr(P, "_join_polygon_stats", fake_join_polygon_stats)

    result = P._join_census(
        h3_grid="fake_h3_grid",
        aoi="fake_aoi",
        states=None,
        census_levels=("blockgroup",),
        cache_dir="fake_cache_dir",
        country="DEU",
        census_module=None,
    )

    assert result == "fake_h3_grid"
    assert calls["prefix"] == "destatis_"
    assert calls["keep_columns"] == P.GERMANY_KEEP_COLUMNS
    # The h3-grid join uses the admin-polygon levels only -- the real Zensus
    # grid level ("grid", now the finest entry in P.GERMANY_LEVELS, used by
    # the map's census dispatch instead) is joined separately by
    # `_join_zensus_grid_population`, not through this polygon-apportionment
    # path (see `code.pipeline.GERMANY_ADMIN_LEVELS`'s docstring).
    assert calls["levels"] == P.GERMANY_ADMIN_LEVELS
    captured = capsys.readouterr()
    assert "not yet supported" not in captured.out
