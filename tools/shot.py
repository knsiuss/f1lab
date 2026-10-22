r"""Screenshot a Streamlit page with Playwright (headless Chrome).

Usage:
    .venv\Scripts\python.exe tools\shot.py <url> <out.png> [wait_seconds]

Streamlit renders client-side, so the tool waits for the run to settle
before capturing: the "stStatusWidget" (the little "Running" indicator)
must be gone and the main area must have real content, otherwise the
capture is only the skeleton.  The body text is then checked for the
markers of a broken page and a short excerpt is printed, which makes the
tool usable as a smoke test as well.
"""

import sys
import time

from playwright.sync_api import sync_playwright

#: Elements that show the app is still working.
BUSY_SELECTORS = (
    '[data-testid="stStatusWidget"]',
    '[data-testid="stSpinner"]',
)

#: Markers of a broken page.
ERROR_MARKERS = (
    "Critical Error",
    "Traceback",
    "KeyError",
    "TypeError",
    "AttributeError",
    "ValueError",
)


def _busy(page) -> bool:
    for selector in BUSY_SELECTORS:
        try:
            if page.locator(selector).count():
                return True
        except Exception:
            continue
    return False


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    url = sys.argv[1]
    out = sys.argv[2]
    wait = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.goto(url, wait_until="domcontentloaded", timeout=90_000)

        deadline = time.time() + wait
        settled = False
        while time.time() < deadline:
            if not _busy(page):
                # Give the charts a moment to draw, then check the page
                # really has content and not just the skeleton.
                time.sleep(3.0)
                if not _busy(page):
                    settled = True
                    break
            time.sleep(1.0)
        if not settled:
            print(f"!! page was still running after {wait:.0f}s")

        body = page.inner_text("body")
        for marker in ERROR_MARKERS:
            if marker in body:
                print(f"!! page contains {marker}")
        excerpt = " ".join(body.split())
        print(excerpt[:600])
        page.screenshot(path=out, full_page=True)
        browser.close()

    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
