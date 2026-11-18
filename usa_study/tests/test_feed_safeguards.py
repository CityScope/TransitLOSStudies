"""Regression tests for the feed / mode bugs of 2026-10 (NYC subway replaced by a ferry feed; rail routes scored as bus)."""
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.feed_validate import audit_state, filter_downloaded, replacement_problem  # noqa: E402


def make_feed(folder: Path, route_rows, stops=((1, 40.0, -74.0), (2, 40.01, -74.0))):
    """Tiny valid GTFS: `route_rows` = [(route_id, route_type)], one trip per route along both stops."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "agency.txt").write_text("agency_id,agency_name,agency_url,agency_timezone\n1,Test,http://x,America/New_York\n")
    (folder / "stops.txt").write_text("stop_id,stop_name,stop_lat,stop_lon\n" + "".join(f"{i},S{i},{la},{lo}\n" for i, la, lo in stops))
    (folder / "routes.txt").write_text("route_id,route_short_name,route_type,agency_id\n" + "".join(f"{r},{r.strip()},{t},1\n" for r, t in route_rows))
    (folder / "calendar.txt").write_text("service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\nW,1,1,1,1,1,1,1,20260101,20271231\n")
    trips, times = [], []
    for k, (r, _t) in enumerate(route_rows):
        for j in range(8):
            tid = f"t{k}_{j}"
            trips.append(f"{r},W,{tid}\n")
            h = 8 + j
            times.append(f"{tid},{h}:00:00,{h}:00:00,{stops[0][0]},1\n{tid},{h}:10:00,{h}:10:00,{stops[1][0]},2\n")
    (folder / "trips.txt").write_text("route_id,service_id,trip_id\n" + "".join(trips))
    (folder / "stop_times.txt").write_text("trip_id,arrival_time,departure_time,stop_id,stop_sequence\n" + "".join(times))


def test_ferry_feed_cannot_replace_subway(tmp_path):
    ferry = tmp_path / "ferry"
    make_feed(ferry, [("SIF", 4)])
    rows = [{"agency_name": "MTA NYCT", "mode_name": "Heavy Rail"}]
    assert "ferry" in replacement_problem(ferry, rows)
    assert filter_downloaded({ferry: rows}) == {}
    assert (tmp_path / "_rejected_ferry").is_dir()          # moved aside, never scored


def test_matching_modes_are_accepted(tmp_path):
    subway, bus = tmp_path / "subway", tmp_path / "bus"
    make_feed(subway, [("A", 1)])
    make_feed(bus, [("B1", 3)])
    assert replacement_problem(subway, [{"mode_name": "Heavy Rail"}]) is None
    assert replacement_problem(bus, [{"mode_name": "Bus"}]) is None
    assert replacement_problem(bus, [{"mode_name": "Light Rail"}]) is not None
    assert replacement_problem(subway, []) is None             # no expectation (manual feed)


def test_audit_reports_ntd_rows_without_a_valid_feed(tmp_path):
    ferry = tmp_path / "ferry"
    make_feed(ferry, [("SIF", 4)])
    rows = [{"agency_name": "MTA NYCT", "mode_name": "Heavy Rail", "ntd_id": "20099"}]
    problems = audit_state("NY", rows, [ferry], {ferry: rows})
    assert [p["agency"] for p in problems] == ["MTA NYCT"]


def test_route_modes_survive_id_collisions_and_padding(tmp_path):
    """Two feeds with the SAME route id '1' (bus / rail) and a padded id: every route must keep its own mode."""
    pytest.importorskip("pyGTFSHandler")
    from pyGTFSHandler.feed import Feed

    from code.best_subset_scoring import feed_route_modes

    a, b = tmp_path / "a_bus", tmp_path / "b_rail"
    make_feed(a, [("1", 3), ("9", 3)])
    make_feed(b, [("1", 2), ("C1  ", 2)], stops=((11, 40.0, -74.0), (12, 40.01, -74.0)))
    feed = Feed(gtfs_dirs=[str(a), str(b)])
    given = pl.DataFrame({
        "route_id": ["1", "9", "1", "C1  "], "scoring_mode": ["bus", "bus", "rail", "rail"], "file_id": [0, 0, 1, 1],
    })
    modes = feed_route_modes(feed, given)
    got = dict(zip(modes["route_id"].to_list(), modes["scoring_mode"].to_list()))
    assert got["1_file_0"] == "bus" and got["1_file_1"] == "rail"      # renamed colliding ids
    assert got["C1"] == "rail"                                           # padding trimmed by the Feed, mode kept
    assert got["9"] == "bus"
