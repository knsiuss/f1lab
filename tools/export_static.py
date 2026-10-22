#!/usr/bin/env python3
"""Export a static HTML page per finished race (the site/ directory).

The Streamlit app is the product, but a race page that never changes is worth
publishing as plain HTML: it is linkable, indexable and costs nothing to serve.

    python tools/export_static.py --year 2026 --out site
    python tools/export_static.py --year 2025 --limit 3

What it writes:

    site/index.html                  the season index (rounds + standings table)
    site/<year>-<round>-<slug>.html  one page per race

Data comes from the same OpenF1 feed the app reads, through the same cache, so
an export offline of a warm cache makes no extra requests. Nothing here makes a
number up: when the season frame is unavailable the tool exits with code 2
instead of writing an empty site.

The pages carry no CSS framework and no JavaScript at all: the standings are
plain HTML bars, so a race page opens from a file system, works on a host that
blocks third-party CDNs, and renders before any script would have loaded.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import openf1_results  # noqa: E402
import report  # noqa: E402

SOURCE_NOTE = ('Results, lap times and standings from the OpenF1 public API '
               '(CC BY-NC-SA 4.0). Built with F1 Lab, a public-data analytics '
               'workbench; not a team tool and not official timing.')

PAGE_CSS = """
  :root { color-scheme: dark; }
  body { background:#0B0D12; color:#E8EAF0; margin:0; padding:2.5rem max(1rem, calc(50vw - 32rem));
         font-family:'Outfit', Inter, 'Segoe UI', sans-serif; line-height:1.6; }
  a { color:#E10600; }
  h1 { font-size:1.9rem; margin:.2rem 0 .4rem; }
  h2 { color:#E10600; font-size:1.05rem; margin:1.8rem 0 .5rem; letter-spacing:.02em; }
  h3 { font-size:.95rem; margin:1.2rem 0 .4rem; color:#cfd4de; }
  .eyebrow { color:#8b93a7; text-transform:uppercase; letter-spacing:.14em; font-size:.7rem; }
  .lead { color:#aeb6c6; margin:0 0 1.2rem; }
  pre { white-space:pre-wrap; font-family:ui-monospace, Consolas, monospace; font-size:.82rem;
        background:#12151C; border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:1rem; }
  .card { background:#12151C; border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:1rem 1.2rem; }
  ul { padding-left:1.1rem; }
  li { margin:.25rem 0; color:#c6ccd8; }
  table.md { border-collapse:collapse; width:100%; font-size:.82rem; margin:.4rem 0 1rem;
             background:#12151C; border:1px solid rgba(255,255,255,.08); border-radius:12px; overflow:hidden; }
  table.md th, table.md td { padding:.4rem .6rem; border-bottom:1px solid rgba(255,255,255,.06); }
  table.md th { text-align:left; color:#8b93a7; font-weight:600; text-transform:uppercase;
                letter-spacing:.06em; font-size:.68rem; }
  ol.bars { list-style:none; margin:.6rem 0 1rem; padding:0; }
  ol.bars li { display:flex; align-items:center; gap:.7rem; padding:.22rem 0; }
  ol.bars .who { flex:0 0 11.5rem; font-size:.85rem; color:#d8dce6;
                 white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  ol.bars .track { flex:1 1 auto; height:.85rem; background:rgba(255,255,255,.05); border-radius:2rem; }
  ol.bars .fill { display:block; height:100%; border-radius:2rem; }
  ol.bars .pts { flex:0 0 3rem; text-align:right; font-size:.82rem; font-variant-numeric:tabular-nums; }
  table.md tr:last-child td { border-bottom:0; }
  table.md td.num { text-align:right; font-variant-numeric:tabular-nums; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(16rem, 1fr)); gap:.6rem; }
  .round { display:block; padding:.7rem .9rem; border-radius:12px; text-decoration:none; color:#E8EAF0;
           background:#12151C; border:1px solid rgba(255,255,255,.08); }
  .round:hover { border-color:#E10600; }
  .round small { display:block; color:#8b93a7; font-size:.75rem; }
  footer { margin:3rem 0 1rem; color:#7d8596; font-size:.75rem; border-top:1px solid rgba(255,255,255,.08); padding-top:1rem; }
"""


def slugify(text: str) -> str:
    """Filesystem-safe slug for a race name."""
    cleaned = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return cleaned or 'race'


_SEPARATOR = re.compile(r'^:?-{2,}:?$')
_BOLD = re.compile(r'\*\*(.+?)\*\*')
_ITALIC = re.compile(r'(?<![\w*])([_*])(?!\s)(.+?)(?<!\s)\1(?![\w*])')
_CODE = re.compile(r'`([^`]+)`')


def _pipe_cells(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def _is_separator(line: str) -> bool:
    cells = _pipe_cells(line)
    return len(cells) >= 2 and all(_SEPARATOR.match(cell) for cell in cells)


def _inline(text: str) -> str:
    """Escape, then apply the three inline shapes the report uses."""
    out = html.escape(text)
    out = _CODE.sub(r'<code>\1</code>', out)
    out = _BOLD.sub(r'<strong>\1</strong>', out)
    out = _ITALIC.sub(r'<em>\2</em>', out)
    return out


def _table_html(header: str, separator: str, body: List[str]) -> str:
    """Render a pipe table as a real table."""
    numeric = [cell.endswith(':') for cell in _pipe_cells(separator)]
    head = ''.join(f'<th>{_inline(cell)}</th>' for cell in _pipe_cells(header))
    rows = []
    for line in body:
        cells = []
        for index, cell in enumerate(_pipe_cells(line)):
            cls = ' class="num"' if index < len(numeric) and numeric[index] else ''
            cells.append(f'<td{cls}>{_inline(cell)}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')
    return (f'<table class="md"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _md_block_to_html(block: str) -> str:
    """Render one block of the report markdown without a markdown library.

    The report is generated by ``src/report.py``, so the shapes are known:
    ``#``/``##``/``###`` headings, ``- `` bullets, pipe tables whose header
    line comes straight after a heading (so it does not start with ``|``),
    and short paragraphs with bold, italic or code spans.
    """
    lines = block.splitlines()
    if not lines:
        return ''
    out: List[str] = []
    paragraph: List[str] = []

    def flush() -> None:
        text = ' '.join(line.strip() for line in paragraph if line.strip())
        paragraph.clear()
        if text:
            out.append(f'<p>{_inline(text)}</p>')

    index = 0
    while index < len(lines):
        line = lines[index]
        if '|' in line and index + 1 < len(lines) and _is_separator(lines[index + 1]):
            flush()
            body: List[str] = []
            cursor = index + 2
            while cursor < len(lines) and lines[cursor].strip() and '|' in lines[cursor]:
                body.append(lines[cursor])
                cursor += 1
            out.append(_table_html(line, lines[index + 1], body))
            index = cursor
            continue
        if line.startswith('- '):
            flush()
            items: List[str] = []
            cursor = index
            while cursor < len(lines) and lines[cursor].startswith('- '):
                items.append(f'<li>{_inline(lines[cursor][2:].strip())}</li>')
                cursor += 1
            out.append(f'<ul>{"".join(items)}</ul>')
            index = cursor
            continue
        paragraph.append(line)
        index += 1
    flush()
    return ''.join(out)


def markdown_to_html(markdown: str) -> str:
    """Turn the report markdown into a small amount of safe HTML."""
    out: List[str] = []
    for block in re.split(r'\n\s*\n', markdown.strip()):
        stripped = block.strip()
        if not stripped:
            continue
        first, _, rest = stripped.partition('\n')
        if first.startswith('### '):
            out.append(f'<h3>{html.escape(first[4:].strip())}</h3>')
            if rest.strip():
                out.append(_md_block_to_html(rest))
            continue
        if first.startswith('## '):
            out.append(f'<h2>{html.escape(first[3:].strip())}</h2>')
            if rest.strip():
                out.append(_md_block_to_html(rest))
            continue
        if first.startswith('# '):
            out.append(f'<h1>{html.escape(first[2:].strip())}</h1>')
            if rest.strip():
                out.append(_md_block_to_html(rest))
            continue
        out.append(_md_block_to_html(stripped))
    return '\n'.join(out)


def standings_html(frame, upto_round: int, colours: Dict[str, str]) -> Optional[str]:
    """Top-ten championship bars as of a round, or None when there is no data.

    Rendered as HTML rather than a Plotly figure: the page then needs no
    JavaScript at all, so it works from a file system and on a static host
    that blocks third-party CDNs, and long driver names cannot be clipped by
    a chart margin.
    """
    rows = frame[frame['Round'] <= upto_round]
    if rows.empty or 'Points' not in rows.columns:
        return None
    points = rows.groupby('Driver')['Points'].sum().sort_values(ascending=False).head(10)
    if points.empty:
        return None
    teams = (rows.sort_values('Round').groupby('Driver')['Team'].last()
             .reindex(points.index).fillna(''))
    peak = float(points.max()) or 1.0
    bars = []
    for driver, value in points.items():
        colour = colours.get(teams.get(driver, ''), '#6c757d')
        width = max(1.5, 100.0 * float(value) / peak)
        bars.append(
            f'<li><span class="who">{html.escape(str(driver))}</span>'
            f'<span class="track"><span class="fill" style="width:{width:.1f}%;'
            f'background:{html.escape(colour)}"></span></span>'
            f'<span class="pts">{float(value):.0f}</span></li>')
    return f'<ol class="bars">{"".join(bars)}</ol>'


def page(title: str, eyebrow: str, body: str, subtitle: str = '') -> str:
    """One self-contained HTML page."""
    lead = f'<p class="lead">{html.escape(subtitle)}</p>' if subtitle else ''
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<div class="eyebrow">{html.escape(eyebrow)}</div>
<h1>{html.escape(title)}</h1>
{lead}
{body}
<footer>{html.escape(SOURCE_NOTE)}</footer>
</body>
</html>
"""


def race_page(year: int, track: str, round_number: int, frame, colours: Dict[str, str]) -> str:
    """The page for one race: the report, the standings chart, the round facts."""
    markdown = report.build_report(year, track, frame)
    chart = standings_html(frame, round_number, colours)
    body = ''
    if chart:
        body += f'<h2>Championship after this round</h2>{chart}'
    body += markdown_to_html(markdown)
    return page(title=f'{track} {year}', eyebrow=f'Round {round_number}',
                body=body, subtitle=f'The {year} season as it stood after round {round_number}.')


def index_page(year: int, races: List[Dict[str, Any]], frame, colours: Dict[str, str]) -> str:
    """The season index: every exported race plus the current standings."""
    cards = ''.join(
        f'<a class="round" href="{html.escape(item["file"])}">{html.escape(item["track"])}'
        f'<small>Round {item["round"]}</small></a>'
        for item in races)
    chart = standings_html(frame, 99, colours)
    body = '<h2>Races</h2><div class="grid">' + cards + '</div>'
    if chart:
        body += f'<h2>Championship</h2>{chart}'
    return page(title=f'F1 {year} season', eyebrow='F1 Lab',
                body=body, subtitle='Every finished race of the season, one page each.')


def export(year: int, out_dir: Path, limit: Optional[int] = None) -> int:
    """Write the site. Returns the number of race pages written."""
    frame = openf1_results.build_season_results(year)
    if frame is None or frame.empty:
        print(f'No results available for {year}: nothing to export.')
        return 0
    if 'SessionType' in frame.columns:
        races = frame[frame['SessionType'] == 'Race'].copy()
    else:
        races = frame.copy()
    if 'Round' not in races.columns:
        print('The season frame has no Round column: cannot order the races.')
        return 0
    races = races.drop_duplicates('Round').sort_values('Round')
    if limit:
        races = races.tail(int(limit))

    colours = openf1_results.team_colours(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Dict[str, Any]] = []
    for _, row in races.iterrows():
        track = str(row.get('Track', '')).strip()
        round_number = int(row.get('Round', 0))
        if not track:
            continue
        filename = f'{year}-{round_number:02d}-{slugify(track)}.html'
        (out_dir / filename).write_text(
            race_page(year, track, round_number, frame, colours), encoding='utf-8')
        written.append({'track': track, 'round': round_number, 'file': filename})
        print(f'wrote {out_dir / filename}')

    (out_dir / 'index.html').write_text(
        index_page(year, written, frame, colours), encoding='utf-8')
    print(f'wrote {out_dir / "index.html"} ({len(written)} race pages)')
    return len(written)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--year', type=int, default=None,
                        help='season to export (default: the current year)')
    parser.add_argument('--out', default='site',
                        help='output directory (default: site)')
    parser.add_argument('--limit', type=int, default=None,
                        help='export only the last N rounds')
    args = parser.parse_args(argv)

    from datetime import datetime
    year = args.year or datetime.now().year
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    written = export(year, out_dir, args.limit)
    return 0 if written else 2


if __name__ == '__main__':
    raise SystemExit(main())
