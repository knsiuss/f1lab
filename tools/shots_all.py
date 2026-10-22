r"""Screenshot every page of the running app with Playwright.

Usage:
    .venv\Scripts\python.exe tools\shots_all.py [url] [out_dir]

The app is a Streamlit multi-page app built with ``st.navigation``. Every page
has its own URL path ("/Predictor", "/Analysis", ...), which the sidebar links
carry, so the pages are reached by following those links. Each page is captured
to a PNG (full page) so the layout can be reviewed in one go.
"""

import pathlib
import re
import sys
import time

from playwright.sync_api import sync_playwright

#: Elements that show the app is still working.  The sidebar spinner is part
#: of the page, so a screenshot taken while it is up shows an empty page.
BUSY_SELECTORS = (
    '[data-testid="stStatusWidget"]',
    '[data-testid="stSpinner"]',
)


def _busy(page) -> bool:
    for selector in BUSY_SELECTORS:
        try:
            if page.locator(selector).count():
                return True
        except Exception:
            continue
    return False


def wait_settled(page, timeout_s: float = 45.0) -> None:
    """Wait until Streamlit finished the run (no status widget, no spinner)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _busy(page):
            # Give the charts a moment to draw before the capture.
            time.sleep(3.0)
            if not _busy(page):
                return
        time.sleep(1.0)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def page_paths(page) -> list:
    """The URL paths of the app pages, taken from the sidebar links.

    The links are rendered by Streamlit, so this waits for them instead of
    reading an empty sidebar.
    """
    deadline = time.time() + 90.0
    paths = []
    while time.time() < deadline:
        hrefs = page.eval_on_selector_all(
            "a", "els => els.map(e => e.getAttribute('href') || '')")
        paths = []
        for href in hrefs:
            if not href or "://" in href or href.startswith("#"):
                continue
            if href.startswith("/"):
                path = href.split("?")[0]
            elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", href):
                path = "/" + href
            else:
                continue
            if path not in paths:
                paths.append(path)
        if paths:
            return paths
        time.sleep(1.0)
    return paths


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8501/"
    out_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "tools/shots")
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        # A tall viewport: Streamlit scrolls an inner container, so a normal
        # full-page screenshot would only capture the first screen.
        page = browser.new_page(viewport={"width": 1600, "height": 2600})
        page.goto(url, wait_until="domcontentloaded", timeout=90_000)
        wait_settled(page, 60)

        paths = page_paths(page)
        if not paths:
            page.screenshot(path=str(out_dir / "home.png"), full_page=True)
            print("saved", out_dir / "home.png")
        else:
            for path in paths:
                try:
                    page.goto(url.rstrip("/") + path,
                              wait_until="domcontentloaded", timeout=90_000)
                except Exception as exc:
                    print(f"!! could not open {path}: {exc}")
                    continue
                wait_settled(page, 90)
                target = out_dir / f"{slug(path or 'home')}.png"
                page.screenshot(path=str(target), full_page=True)
                body = page.inner_text("body")
                for marker in ("Critical Error", "Traceback", "KeyError"):
                    if marker in body:
                        print(f"!! {path} contains {marker}")
                print("saved", target)

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
