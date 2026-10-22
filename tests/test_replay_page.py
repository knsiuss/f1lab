# -*- coding: utf-8 -*-
"""Tests for the Replay page (src/pages/5_Replay.py) and its design-system use.

The page itself is a Streamlit script, so the pure helpers are tested directly
and the page body is exercised through an import with the network stubbed.
"""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "src" / "pages" / "5_Replay.py"
sys.path.insert(0, str(ROOT / "src"))


def _page_module():
    """Import the page module the way Streamlit runs it (the body executes)."""
    spec = importlib.util.spec_from_file_location("replay_page_under_test", PAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def page_module():
    """The page module, imported with the API stubbed out.

    The body runs on import, so the stub has to be in place while it does:
    the page must never reach the network on its own.
    """
    import telemetry  # the module the page imports (top-level, not src.)
    with mock.patch.object(telemetry, "safe_race_sessions", lambda year: []):
        return _page_module()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_format_duration(page_module):
    assert page_module._format_duration(6612) == "1:50:12"
    assert page_module._format_duration(95) == "1:35"
    assert page_module._format_duration(0) == "0:00"
    assert page_module._format_duration(None) == "--:--"
    assert page_module._format_duration("nonsense") == "--:--"


def test_format_lap_time(page_module):
    assert page_module._format_lap_time(91.994) == "1:31.994"
    assert page_module._format_lap_time(59.5) == "0:59.500"
    assert page_module._format_lap_time(None) == "--.---"


def _payload():
    return {
        "meta": {
            "duration": 6612.0, "totalLaps": 58,
            "coverage": {"drivers": 20, "withTelemetry": 9},
            "fastest": [{"driver": 1, "lap": 12, "time": 91.994}],
            "speedTrap": [{"driver": 4, "speed": 337}],
        },
        "drivers": [
            {"number": 1, "acronym": "VER", "team": "Red Bull Racing",
             "positionCount": 3,
             "laps": [{"number": 1, "start": 120.0}, {"number": 2, "start": 212.0}],
             "positionByLap": [1, 2]},
            {"number": 4, "acronym": "NOR", "team": "McLaren",
             "positionCount": 16,
             "laps": [{"number": 1, "start": 121.0}, {"number": 2, "start": 213.0}],
             "positionByLap": [2, 1]},
        ],
    }


def test_position_frame_holds_the_last_measured_position(page_module):
    frame = page_module._position_frame(_payload())
    assert list(frame.columns) == ["Driver", "LapNumber", "Position"]
    assert len(frame) == 4
    assert frame.iloc[0].to_dict() == {"Driver": "VER", "LapNumber": 1, "Position": 1}
    assert frame.iloc[1].to_dict() == {"Driver": "VER", "LapNumber": 2, "Position": 2}


def test_position_frame_is_empty_without_data(page_module):
    assert page_module._position_frame({"drivers": []}).empty
    assert page_module._position_frame({}).empty


def test_kpi_rows_report_coverage_and_the_position_feed(page_module):
    tiles = {tile["label"]: tile for tile in page_module._kpi_rows(_payload())}
    assert tiles["Laps"]["value"] == "58"
    assert tiles["Race time"]["value"] == "1:50:12"
    # Partial telemetry has to be visible in the tile row, not only in a note.
    assert tiles["Drivers"]["value"] == "9/20"
    assert tiles["Drivers"]["tone"] == "warn"
    assert tiles["Position readings"]["value"] == "19"
    assert tiles["Position readings"]["tone"] == "warn"
    assert tiles["Fastest lap"]["value"] == "1:31.994"
    assert tiles["Fastest lap"]["sub"] == "VER"
    assert tiles["Top speed"]["value"] == "337 km/h"


def test_kpi_rows_are_neutral_when_coverage_is_complete(page_module):
    payload = _payload()
    payload["meta"]["coverage"] = {"drivers": 2, "withTelemetry": 2}
    tiles = {tile["label"]: tile for tile in page_module._kpi_rows(payload)}
    assert tiles["Drivers"]["value"] == "2/2"
    assert tiles["Drivers"]["tone"] == "good"


def test_kpi_rows_survive_a_payload_without_facts(page_module):
    tiles = page_module._kpi_rows({"meta": {}, "drivers": []})
    assert tiles
    values = {tile["label"]: tile["value"] for tile in tiles}
    assert values["Fastest lap"] == "--.---"
    assert values["Top speed"] == "--"


# --------------------------------------------------------------------------
# the design system
# --------------------------------------------------------------------------

def test_page_uses_the_design_system():
    """No page-local markup: the components own the styling."""
    source = PAGE.read_text(encoding="utf-8")
    assert "from ui import" in source
    assert "unsafe_allow_html" not in source
    assert 'style="' not in source
    assert "st.divider()" not in source
    assert 'st.markdown("---")' not in source


def test_page_keeps_the_disclosures():
    """The redesign must not drop what the data rules require."""
    source = PAGE.read_text(encoding="utf-8")
    assert "Telemetry coverage is partial" in source
    assert "position feed" in source
    assert "estimate_note(" in source
    assert "CC BY-NC-SA 4.0" in source
    assert "MultiViewer" in source


def test_page_keeps_the_ingest_user_triggered():
    """A race is tens of megabytes: the download must be a button."""
    source = PAGE.read_text(encoding="utf-8")
    assert 'st.button("Download telemetry"' in source
    assert "ingest_session(store, session_key)" in source


# --------------------------------------------------------------------------
# the page body
# --------------------------------------------------------------------------

def test_page_renders_without_reaching_the_network():
    """The page body runs on import; it must not hang on the API."""
    import streamlit as st
    import telemetry  # the module the page imports (top-level, not src.)

    st.cache_data.clear()
    with mock.patch.object(telemetry, "safe_race_sessions", lambda year: []):
        module = _page_module()
    assert callable(module.page)
