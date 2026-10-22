# -*- coding: utf-8 -*-
"""
Interaction check for the F1 Lab app.

Measures what the reader waits for after touching a control, in a real browser:

  * open the page,
  * change one widget inside a panel,
  * time how long until the page is idle again.

This is the number the panels are built for: the tab bodies are ``@st.fragment``
functions, so a change inside a panel rebuilds that panel only. A full script
run (the first paint) is measured by ``tools/latency_check.py``.

The app must already be running. Usage::

    .venv/Scripts/python tools/interact_check.py [base_url]

Budget: 2.5 s per interaction -- the reader waits for the panel, not for the page.
"""

import sys
import time

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8501'
# A panel that only redraws a chart answers in about a second. A panel that
# recomputes a comparison or a model (the head-to-head matchup, the forecast)
# needs a little more, because the numbers themselves have to be worked out
# again for the new selection.
BUDGET_S = 2.5
BUSY = ('[data-testid="stStatusWidget"], [data-testid="stSpinner"]')
PAGES = ['/HeadToHead', '/Predictor', '/Analysis']


def wait_idle(page, timeout_s=90.0):
    deadline = time.time() + timeout_s
    page.wait_for_timeout(400)
    while time.time() < deadline:
        if page.evaluate(
                f"() => document.querySelectorAll('{BUSY}').length") == 0:
            return True
        page.wait_for_timeout(200)
    return False


def first_panel_widget(page):
    """The first visible widget inside a tab panel.

    Widgets outside the panels (the race or season selector at the top of a
    page) change the data for every panel, so they are a full script run and
    belong to the latency tool's budget, not to this one.
    """
    for testid in ('stMultiSelect', 'stSelectbox'):
        boxes = page.locator(f'[data-testid=stTabs] [data-testid={testid}]')
        for index in range(min(boxes.count(), 6)):
            box = boxes.nth(index)
            try:
                if box.is_visible():
                    return box, testid
            except Exception:
                continue
    return None, None


def measure(page, url):
    page.goto(BASE + url, wait_until='domcontentloaded')
    page.wait_for_timeout(1200)
    if not wait_idle(page):
        return None, 'page never settled'
    page.wait_for_timeout(2500)
    box, testid = first_panel_widget(page)
    if box is None:
        return None, 'no visible widget'
    try:
        box.scroll_into_view_if_needed()
        box.click(timeout=8000)
        page.wait_for_timeout(600)
        option = page.locator('[role=option]')
        if not option.count():
            page.keyboard.press('Escape')
            return None, 'no option in the dropdown'
        start = time.time()
        option.first.click(timeout=8000)
        page.keyboard.press('Escape')
        if not wait_idle(page, timeout_s=30.0):
            return None, 'interaction never settled'
        return time.time() - start, testid
    except Exception as exc:  # a widget that cannot be driven is skipped
        return None, f'{type(exc).__name__}'


def main() -> int:
    slow = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome')
        page = browser.new_page(viewport={'width': 1500, 'height': 1400})
        print(f'=== interaction latency ({BASE}) ===')
        for url in PAGES:
            elapsed, note = measure(page, url)
            if elapsed is None:
                print(f'  {url:16s} skipped ({note})')
                continue
            flag = '  <-- SLOW' if elapsed > BUDGET_S else ''
            if elapsed > BUDGET_S:
                slow += 1
            print(f'  {url:16s} {elapsed:5.2f}s   ({note}){flag}')
        browser.close()
    print()
    if slow:
        print(f'{slow} interaction(s) over the {BUDGET_S}s budget')
        return 1
    print(f'every interaction answers inside {BUDGET_S}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
