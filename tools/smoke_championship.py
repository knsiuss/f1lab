# -*- coding: utf-8 -*-
"""Smoke-render the Head-to-Head page (Championship Race tab) with
synthetic season data via Streamlit AppTest. Not part of the test suite.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

rows = []
drivers = [
    ("Max Verstappen", "Red Bull Racing", 1),
    ("Lando Norris", "McLaren", 4),
    ("Charles Leclerc", "Ferrari", 16),
    ("Lewis Hamilton", "Ferrari", 44),
    ("George Russell", "Mercedes", 63),
]
tracks = [
    ("Australian Grand Prix", 1), ("Chinese Grand Prix", 2),
    ("Japanese Grand Prix", 3), ("Bahrain Grand Prix", 4),
    ("Saudi Arabian Grand Prix", 5), ("Miami Grand Prix", 6),
]
points_table = [[25, 18, 15, 12, 10], [18, 25, 12, 15, 10],
                [25, 10, 18, 15, 12], [12, 18, 25, 10, 15],
                [15, 12, 10, 25, 18], [10, 15, 12, 18, 25]]
for r, (track, rnd) in enumerate(tracks):
    for c, (name, team, no) in enumerate(drivers):
        pos = c + 1
        rows.append({"Track": track, "Round": rnd, "SessionType": "Race",
                     "Driver": name, "Team": team, "No": no,
                     "Position": pos, "Points": points_table[r][c],
                     "Status": "Finished", "Finished": True,
                     "Starting Grid": pos})
        if rnd % 2 == 0:
            rows.append({"Track": track, "Round": rnd,
                         "SessionType": "Sprint", "Driver": name,
                         "Team": team, "No": no, "Position": pos,
                         "Points": max(0, 8 - c), "Status": "Finished",
                         "Finished": True, "Starting Grid": pos})
frame = pd.DataFrame(rows)

import shared  # noqa: E402
shared.load_race_data = lambda year: frame  # noqa: E731
shared.season_progress = lambda year: (6, 10)  # noqa: E731

from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(str(ROOT / "src" / "pages" / "2_HeadToHead.py"),
                       default_timeout=120)
at.session_state["selected_year"] = 2026
at.run()
print("exception:", at.exception)
print("plotly charts:", len(at.get("plotly_chart")))
print("markdown blocks:", len(at.markdown))
assert len(at.exception) == 0, at.exception
assert len(at.get("plotly_chart")) > 0
print("SMOKE OK")
