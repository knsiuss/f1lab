# -*- coding: utf-8 -*-
"""download_images.py — fetch driver portraits and country flags locally.

Driver portraits come from the F1 media CDN URLs already recorded in
``config.DRIVER_PROFILES`` (plus the 2026 debutants whose URLs follow the
same pattern). Flags come from flagcdn (flag data is public-domain source
material, Wikimedia). Run once; the app serves everything from
``src/assets/`` afterwards, no hotlinking at page-render time.

Provenance is recorded in ``src/assets/ATTRIBUTION.md``.
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DRIVER_PROFILES  # noqa: E402

ROOT = Path(__file__).parent.parent / "src" / "assets"
DRIVERS = ROOT / "drivers"
FLAGS = ROOT / "flags"
HEADERS = {"User-Agent": "F1Lab/1.0 (local dashboard, non-commercial)"}

# 2026 grid drivers missing from DRIVER_PROFILES, same CDN pattern.
EXTRA_DRIVERS = {
    "isack_hadjar": "https://media.formula1.com/content/dam/fom-website/"
                    "drivers/I/ISAHAD01_Isack_Hadjar/isahad01.png"
                    ".transform/2col/image.png",
}

# Country name (calendar spelling) -> ISO 3166-1 alpha-2 for flagcdn.
COUNTRIES = {
    "Australia": "au", "China": "cn", "Japan": "jp", "United States": "us",
    "USA": "us", "Canada": "ca", "Monaco": "mc", "Spain": "es",
    "Austria": "at", "Great Britain": "gb", "United Kingdom": "gb",
    "Belgium": "be", "Hungary": "hu", "Netherlands": "nl", "Italy": "it",
    "Azerbaijan": "az", "Bahrain": "bh", "Singapore": "sg", "Mexico": "mx",
    "Brazil": "br", "Qatar": "qa", "UAE": "ae", "United Arab Emirates": "ae",
    "Indonesia": "id", "Germany": "de", "France": "fr", "India": "in",
    "South Korea": "kr", "South Africa": "za", "Portugal": "pt",
    "Malaysia": "my", "Vietnam": "vn", "Saudi Arabia": "sa",
}


def fetch(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=20, headers=HEADERS)
        if resp.ok and "image" in resp.headers.get("content-type", ""):
            return resp.content
    except requests.RequestException:
        pass
    return None


def main() -> None:
    DRIVERS.mkdir(parents=True, exist_ok=True)
    FLAGS.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for name, profile in DRIVER_PROFILES.items():
        url = profile.get("image_url")
        if not url:
            continue
        slug = name.lower().replace(" ", "_")
        data = fetch(url)
        if data:
            (DRIVERS / f"{slug}.png").write_bytes(data)
            ok += 1
        else:
            print(f"MISS driver {name}")
            bad += 1
    for slug, url in EXTRA_DRIVERS.items():
        data = fetch(url)
        if data:
            (DRIVERS / f"{slug}.png").write_bytes(data)
            ok += 1
        else:
            print(f"MISS driver {slug}")
            bad += 1
    # Debutants whose CDN path does not follow the pattern: Wikipedia
    # thumbnails (CC BY-SA, see ATTRIBUTION.md).
    for slug, title in [
        ("andrea_kimi_antonelli", "Kimi Antonelli"),
        ("ryo_hirakawa", "Ryo Hirakawa"),
        ("arvid_lindblad", "Arvid Lindblad"),
    ]:
        try:
            page = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/"
                f"{title.replace(' ', '_')}",
                timeout=20, headers=HEADERS)
            thumb = (page.json().get("thumbnail") or {}).get("source")
            data = fetch(thumb) if thumb else None
            if data:
                (DRIVERS / f"{slug}.jpg").write_bytes(data)
                ok += 1
            else:
                print(f"MISS wiki {slug}")
                bad += 1
        except requests.RequestException:
            print(f"MISS wiki {slug}")
            bad += 1
    fok = fbad = 0
    for country, code in sorted(COUNTRIES.items()):
        data = fetch(f"https://flagcdn.com/w160/{code}.png")
        if data:
            (FLAGS / f"{code}.png").write_bytes(data)
            fok += 1
        else:
            print(f"MISS flag {country} ({code})")
            fbad += 1
    print(f"drivers ok={ok} miss={bad}; flags ok={fok} miss={fbad}")


if __name__ == "__main__":
    main()
