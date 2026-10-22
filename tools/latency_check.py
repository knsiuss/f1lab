# -*- coding: utf-8 -*-
"""
Latency check for the F1 Lab app.

Measures what the user actually waits for:

  * the data layer (season frame, grid, team colours),
  * every page, cold (first render in a process) and warm (the rerun a
    widget interaction triggers).

Run it from the project root::

    PYTHONPATH=src .venv/Scripts/python tools/latency_check.py

The warm numbers are the ones that matter for the feel of the app: they are
what a click on a dropdown costs. The cold numbers include the first download
of a season, which is bounded by the OpenF1 rate limit (30 requests/minute)
and happens once per season per machine.
"""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from streamlit.testing.v1 import AppTest  # noqa: E402

import openf1_results as ofr  # noqa: E402
import shared  # noqa: E402

YEAR = 2026
# A page that takes longer than this on a warm render is a regression.
WARM_BUDGET_S = 1.5
# A page whose panels are @st.fragment functions builds every panel on a full
# script run (the first paint, or a change to a control outside the panels), but
# a change *inside* a panel rebuilds only that panel. That interaction -- what
# the reader actually waits for -- is measured in the browser by
# tools/interact_check.py and is the number to hold to WARM_BUDGET_S. The full
# script run gets its own, higher allowance.
FRAGMENT_BUDGET_S = 4.5


def uses_fragments(page) -> bool:
    """True when the page builds its panels as @st.fragment functions."""
    try:
        return '@st.fragment' in page.read_text(encoding='utf-8')
    except OSError:
        return False


def timed(label, call):
    start = time.time()
    result = call()
    print(f'  {label:38s} {time.time() - start:6.2f}s')
    return result


def main() -> int:
    print(f'=== data layer ({YEAR}) ===')
    timed('season_sessions', lambda: ofr.season_sessions(YEAR))
    timed('calendar events', lambda: ofr._calendar_events(YEAR))
    frame = timed('build_season_results', lambda: ofr.build_season_results(YEAR))
    print(f'  {"":38s} {len(frame) if frame is not None else 0} rows')
    timed('build_season_results (again)', lambda: ofr.build_season_results(YEAR))
    timed('shared.load_race_data', lambda: shared.load_race_data(YEAR))
    timed('shared.season_progress', lambda: shared.season_progress(YEAR))
    timed('team colours', lambda: shared.season_team_colours(YEAR))

    print('\n=== pages (cold = first render, warm = rerun) ===')
    pages = sorted((ROOT / 'src' / 'pages').glob('*.py'))
    slow = []
    for page in pages:
        budget = FRAGMENT_BUDGET_S if uses_fragments(page) else WARM_BUDGET_S
        app = AppTest.from_file(str(page), default_timeout=600)
        start = time.time()
        app.run()
        cold = time.time() - start
        start = time.time()
        app.run()
        warm = time.time() - start
        errors = len(app.exception) + len(app.error)
        flag = '  <-- SLOW' if warm > budget else ''
        note = '  (fragments)' if budget != WARM_BUDGET_S else ''
        if warm > budget:
            slow.append(page.name)
        print(f'  {page.name:20s} cold {cold:6.2f}s   warm {warm:6.2f}s'
              f'   errors {errors}{note}{flag}')

    print()
    if slow:
        print(f'pages over their warm budget: {", ".join(slow)}')
        return 1
    print(f'every page renders inside its warm budget '
          f'({WARM_BUDGET_S}s; {FRAGMENT_BUDGET_S}s for the first full render of '
          f'a page whose panels are fragments)')
    print('run tools/interact_check.py for the in-panel interaction cost')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
