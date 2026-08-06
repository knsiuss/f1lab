# -*- coding: utf-8 -*-
"""
seasons.py
Season discovery and shared per-season helpers.

The supported years are derived from the data/ directory instead of being
hardcoded: adding a ``Formula1_{year}Season_RaceResults.csv`` file is all it
takes to opt the app into a new season.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import re
from pathlib import Path
from typing import List

# CSV files use the pattern Formula1_{year}Season_RaceResults.csv
_RACE_RESULTS_PATTERN = re.compile(r"Formula1_(\d{4})Season_RaceResults\.csv")


def discover_available_years(data_dir: Path) -> List[int]:
    """
    Return the seasons present in ``data_dir``, sorted ascending.

    A season is "available" when its race-results CSV exists. Sprint and
    qualifying files alone are not enough because every page keys off the
    race results.

    Args:
        data_dir: Path to the data directory.

    Returns:
        Sorted list of integer years; empty list when nothing is found.
    """
    try:
        data = Path(data_dir)
        if not data.is_dir():
            return []
        years = set()
        for path in data.glob("Formula1_*Season_RaceResults.csv"):
            match = _RACE_RESULTS_PATTERN.match(path.name)
            if match:
                years.add(int(match.group(1)))
    except OSError:
        return []
    return sorted(years)
