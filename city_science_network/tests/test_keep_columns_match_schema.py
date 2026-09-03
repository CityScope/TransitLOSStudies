"""Regression test for the exact bug found and fixed 2026-08-19: `pipeline.py`'s
hardcoded `*_KEEP_COLUMNS` sets (and the `prefix` string handed to
`_join_polygon_stats`) silently going stale whenever pyCensus renames a
source's column prefix or a feature name.

Concretely, an earlier pyCensus schema-restructuring round renamed ACS's
column prefix `acs_` -> `acs5_` and switched DHC/LODES from snake_case to
camelCase feature names, but `pipeline.py`'s keep-column sets kept the old
names. The loaders still succeeded and returned real data, but the
column-name filter in `_join_polygon_stats` (`col.startswith(prefix) and col
in keep_columns`) matched nothing -- so every census join silently returned
zero columns, with no exception and no warning.

This test asserts every name in each `*_KEEP_COLUMNS` set is actually
producible by the corresponding pyCensus schema (`ColumnSchema.prefixed()`
over its `.columns`), so a future pyCensus rename that isn't mirrored here
fails loudly in CI instead of silently degrading a live census join to zero
columns.
"""

from code import pipeline as P

from pycensus.countries.usa.acs5.schema import SCHEMA as ACS5_SCHEMA
from pycensus.countries.usa.dhc.schema import SCHEMA as DHC_SCHEMA
from pycensus.countries.usa.lodes_wac.schema import SCHEMA as LODES_WAC_SCHEMA
from pycensus.countries.germany.ba.schema import SCHEMA as GERMANY_BA_SCHEMA
from pycensus.countries.mexico.inegi.schema import SCHEMA as INEGI_SCHEMA
from pycensus.countries.euskadi.eustat.schema import SCHEMA as EUSKADI_SCHEMA
from pycensus.countries.spain.ine.schema import SCHEMA as SPAIN_SCHEMA
from pycensus.countries.spain.censo2021.schema import SCHEMA as SPAIN_CENSO2021_SCHEMA
from pycensus.countries.israel.cbs.schema import SCHEMA as ISRAEL_SCHEMA
from pycensus.countries.taiwan.moi.schema import SCHEMA as TAIWAN_MOI_SCHEMA
from pycensus.countries.chile.ine.schema import SCHEMA as CHILE_INE_SCHEMA
from pycensus.countries.chile.casen.schema import SCHEMA as CHILE_CASEN_SCHEMA
from pycensus.countries.canada.statcan.schema import SCHEMA as CANADA_SCHEMA
from pycensus.countries.andorra.estadisticaad.schema import SCHEMA as ANDORRA_SCHEMA


def _all_prefixed_columns(schema) -> set[str]:
    return {schema.prefixed(name) for name in schema.columns}


def test_acs_keep_columns_match_current_acs5_schema():
    valid = _all_prefixed_columns(ACS5_SCHEMA)
    stale = P.ACS_KEEP_COLUMNS - valid
    assert not stale, (
        f"ACS_KEEP_COLUMNS contains columns not produced by the current acs5 "
        f"schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_dhc_keep_columns_match_current_dhc_schema():
    valid = _all_prefixed_columns(DHC_SCHEMA)
    stale = P.DHC_KEEP_COLUMNS - valid
    assert not stale, (
        f"DHC_KEEP_COLUMNS contains columns not produced by the current dhc "
        f"schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_lodes_keep_columns_match_current_lodes_wac_schema():
    valid = _all_prefixed_columns(LODES_WAC_SCHEMA)
    stale = P.LODES_KEEP_COLUMNS - valid
    assert not stale, (
        f"LODES_KEEP_COLUMNS contains columns not produced by the current "
        f"lodes_wac schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_germany_ba_keep_columns_match_current_ba_schema():
    valid = _all_prefixed_columns(GERMANY_BA_SCHEMA)
    stale = P.GERMANY_BA_KEEP_COLUMNS - valid
    assert not stale, (
        f"GERMANY_BA_KEEP_COLUMNS contains columns not produced by the current "
        f"germany.ba schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_inegi_keep_columns_match_current_inegi_schema():
    valid = _all_prefixed_columns(INEGI_SCHEMA)
    stale = P.INEGI_KEEP_COLUMNS - valid
    assert not stale, (
        f"INEGI_KEEP_COLUMNS contains columns not produced by the current "
        f"inegi schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_euskadi_keep_columns_match_current_euskadi_schema():
    valid = _all_prefixed_columns(EUSKADI_SCHEMA)
    stale = P.EUSKADI_KEEP_COLUMNS - valid
    assert not stale, (
        f"EUSKADI_KEEP_COLUMNS contains columns not produced by the current "
        f"euskadi eustat schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_spain_keep_columns_match_current_spain_schemas():
    """SPAIN_KEEP_COLUMNS covers columns from TWO real sources merged by
    `_spain_census_loader` -- `spain.ine` (Padron population) and
    `spain.censo2021` (SDC21 education/employment/migration/households).
    Both sources' outputs are canonicalized to bare `global_schema.json`
    names by `pycensus.canonical.apply_canonical_names`, but
    `_join_polygon_stats` reconstructs the `"ine_"`-prefixed form as this
    set's naming convention regardless of which source actually produced a
    column (see `pipeline.py`'s `SPAIN_KEEP_COLUMNS` comment) -- so here
    that reconstruction is mirrored: every bare name from EITHER schema is
    valid once `"ine_"`-prefixed.
    """
    valid = {f"ine_{name}" for name in SPAIN_SCHEMA.columns} | {
        f"ine_{name}" for name in SPAIN_CENSO2021_SCHEMA.columns
    }
    stale = P.SPAIN_KEEP_COLUMNS - valid
    assert not stale, (
        f"SPAIN_KEEP_COLUMNS contains columns not produced by the current "
        f"spain ine/censo2021 schemas (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_israel_keep_columns_match_current_israel_cbs_schema():
    valid = _all_prefixed_columns(ISRAEL_SCHEMA)
    stale = P.ISRAEL_KEEP_COLUMNS - valid
    assert not stale, (
        f"ISRAEL_KEEP_COLUMNS contains columns not produced by the current "
        f"israel cbs schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_taiwan_keep_columns_match_current_moi_schema():
    """Covers the 2026-08-23 age-bracket addition (under18Population/
    adultPopulation/over65Population/medianAge, computed from MOI RIS
    ODRP014's real single-year age columns -- see
    pycensus.countries.taiwan.moi.loader._add_derived_age_columns)."""
    valid = _all_prefixed_columns(TAIWAN_MOI_SCHEMA)
    stale = P.TAIWAN_KEEP_COLUMNS - valid
    assert not stale, (
        f"TAIWAN_KEEP_COLUMNS contains columns not produced by the current "
        f"taiwan moi schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_chile_ine_keep_columns_match_current_ine_schema():
    valid = _all_prefixed_columns(CHILE_INE_SCHEMA)
    stale = P.CHILE_KEEP_COLUMNS - valid
    assert not stale, (
        f"CHILE_KEEP_COLUMNS contains columns not produced by the current "
        f"chile ine schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_chile_casen_keep_columns_match_current_casen_schema():
    """Covers the 2026-08-23 Casen SAE comunal poverty addition
    (povertyRate/povertyRateMultidimensional, real Ministerio de Desarrollo
    Social bulk .xlsx download, distinct from INE's Censo 2017 DPA source
    above -- see pycensus.countries.chile.casen.api module docstring)."""
    valid = _all_prefixed_columns(CHILE_CASEN_SCHEMA)
    stale = P.CHILE_CASEN_KEEP_COLUMNS - valid
    assert not stale, (
        f"CHILE_CASEN_KEEP_COLUMNS contains columns not produced by the current "
        f"chile casen schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_canada_keep_columns_match_current_statcan_schema():
    """Covers the 2026-08-23 household size/income/education/tenure/commute-mode
    addition (catalogue 98-401-X2021006, dissemination-area level, Ontario
    only -- see pycensus.countries.canada.statcan.README.md)."""
    valid = _all_prefixed_columns(CANADA_SCHEMA)
    stale = P.CANADA_KEEP_COLUMNS - valid
    assert not stale, (
        f"CANADA_KEEP_COLUMNS contains columns not produced by the current "
        f"statcan schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_acs_interpolated_keep_columns_match_current_acs5_schema():
    """`ACS_INTERPOLATED_KEEP_COLUMNS` names `<acs5_field>_interpolated` for
    every field in `ACS_COUNT_FIELDS_FOR_INTERPOLATION` -- guard against the
    same staleness this file's other tests guard against, one level removed
    (strip the `_interpolated` suffix, then check the base name is real).
    """
    valid = _all_prefixed_columns(ACS5_SCHEMA)
    bases = {c.removesuffix("_interpolated") for c in P.ACS_INTERPOLATED_KEEP_COLUMNS}
    stale = bases - valid
    assert not stale, (
        f"ACS_INTERPOLATED_KEEP_COLUMNS contains base fields not produced by the "
        f"current acs5 schema (stale after a pyCensus rename?): {sorted(stale)}"
    )
    assert P.ACS_COUNT_FIELDS_FOR_INTERPOLATION == bases


def test_andorra_keep_columns_match_current_estadisticaad_schema():
    """Covers the 2026-08-23 foreignBornPopulation addition (real,
    parish-level, sourced from the Departament d'Estadistica's PDF bulletin
    -- see pycensus.countries.andorra.estadisticaad.README.md)."""
    valid = _all_prefixed_columns(ANDORRA_SCHEMA)
    stale = P.ANDORRA_KEEP_COLUMNS - valid
    assert not stale, (
        f"ANDORRA_KEEP_COLUMNS contains columns not produced by the current "
        f"andorra estadisticaad schema (stale after a pyCensus rename?): {sorted(stale)}"
    )


def test_join_prefix_args_match_keep_columns_prefix():
    """Every name in a *_KEEP_COLUMNS set must actually start with the
    `prefix` string `_join_polygon_stats` is called with for that source --
    otherwise `col.startswith(prefix)` filters everything out before
    `keep_columns` is even consulted (the second half of the original bug:
    ACS_KEEP_COLUMNS was fixed but the call site still passed the stale
    `"acs_"` prefix instead of `"acs5_"`, and LODES similarly passed
    `"lodes_"` instead of `"lodes_wac_"`).
    """
    for keep_columns, prefix, label in [
        (P.ACS_KEEP_COLUMNS, "acs5_", "ACS"),
        (P.DHC_KEEP_COLUMNS, "dhc_", "DHC"),
        (P.LODES_KEEP_COLUMNS, "lodes_wac_", "LODES"),
        (P.INEGI_KEEP_COLUMNS, "inegi_", "INEGI"),
        (P.EUSKADI_KEEP_COLUMNS, "eustat_", "EUSKADI"),
        (P.SPAIN_KEEP_COLUMNS, "ine_", "SPAIN"),
        (P.ISRAEL_KEEP_COLUMNS, "cbs_", "ISRAEL"),
        (P.CHILE_KEEP_COLUMNS, "ine_cl_", "CHILE"),
        (P.CHILE_CASEN_KEEP_COLUMNS, "casen_cl_", "CHILE_CASEN"),
        (P.CANADA_KEEP_COLUMNS, "statcan_", "CANADA"),
        (P.ANDORRA_KEEP_COLUMNS, "estadisticaad_", "ANDORRA"),
    ]:
        mismatched = {c for c in keep_columns if not c.startswith(prefix)}
        assert not mismatched, f"{label}_KEEP_COLUMNS has entries not matching prefix {prefix!r}: {sorted(mismatched)}"
