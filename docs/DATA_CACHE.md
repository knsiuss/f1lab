# The OpenF1 data cache (data-as-artifact)

The app reads OpenF1 through `src/openf1_client.py`, which keeps a disk cache in
`f1_cache/openf1/`. Normally that cache fills **on demand**: the first visitor of
a page pays for the download. That has two consequences the project does not
want:

* a cold machine (a fresh deployment, a new laptop, a cleared cache) needs the
  network before it can show anything, and the rate limiter means "a few
  minutes" of waiting;
* an OpenF1 outage — or a machine that is offline — leaves the pages empty
  even though the numbers have not changed since the race ended.

The data-as-artifact path fixes that by moving the download **out of the app**:

```
tools/refresh_cache.py  ──writes──>  f1_cache/openf1/  ──reads──>  the app
       (GitHub Actions)                  (one directory)         (artifacts.py)
```

A scheduled job downloads the data, publishes the directory, and the app reads
it from disk. The app never learns where the files came from: they are the same
files, under the same names, that it already wrote for itself.

---

## 1. What is in the archive

For every **started** session of a season (future sessions are skipped and
picked up by a later run):

| Query | Scope | Why the app needs it |
| --- | --- | --- |
| `sessions?year=` | once per year | the calendar of sessions |
| `session_result?session_key=` | per session | finishing order and points |
| `drivers?session_key=` | per session | who is in the session |
| `stints?session_key=` | per session | tyre stints |
| `pit?session_key=` | per session | pit stops |
| `laps?session_key=` | per session | lap times (pace, sectors, replay) |
| `weather?session_key=` | per session | track and air conditions |
| `starting_grid?meeting_key=` | once per meeting | the grid of a race |
| `drivers?meeting_key=` | once per meeting | the driver map of a weekend |

`drivers` is fetched twice on purpose: `openf1_session.py` asks by
`session_key` and `openf1_results.py` (the season frame) asks by
`meeting_key`. Those are two different cache entries, so archiving only one
leaves the other page on the network.

**Not** in the archive, deliberately: `car_data`, `position`, `intervals`, and
`race_control`. They are the large, per-driver, per-second feeds the replay
player ingests into SQLite (`src/telemetry.py`) and they sit behind an explicit
button. Archiving them would multiply the archive size for a feature that is
used on demand. The same applies to FastF1's own `f1_cache/` — this document is
about the OpenF1 archive only.

---

## 2. How the app prefers local data

`src/artifacts.py` is the loader. It imports **without Streamlit** (so tools and
tests can use it) and it maps a query to the *same file the client would write*:
it asks `openf1_client._cache_key` for the name, so the two can never disagree
about which file belongs to `laps?session_key=9158`.

### Freshness policy

| Query | Local file younger than | File older than that |
| --- | --- | --- |
| session the manifest marks `finished` | forever | served from disk anyway |
| anything else (a live session, an unknown session) | 12 h (`LIVE_MAX_AGE_S`) | refetched from the API |
| a past season (`year < current year`) | forever | — |

`FINISHED_MAX_AGE_S = None` means "never expires"; `FINISHED_GRACE_S` (3 h) is
the margin after `date_end` before a session counts as finished, so the job and
the app agree about a session that is still being written.

**"Finished" is not guessed from the clock.** The refresh job records every
session it archived in `f1_cache/openf1/_index/manifest.json`:

```json
{
  "cache_root": "…/f1_cache/openf1",
  "generated_at": "2026-10-23T03:17:04+00:00",
  "sessions": {
    "9158": {
      "year": 2026,
      "meeting_key": 1234,
      "session_name": "Race",
      "session_type": "Race",
      "date_start": "2026-03-08T15:00:00+00:00",
      "date_end": "2026-03-08T17:05:00+00:00",
      "finished": true
    }
  },
  "endpoints": { "laps": { "requests": 91, "hits": 0, "rows": 142384, … } },
  "last_run": { "years": [2025, 2026], "requests": 21, "cache_hits": 480, … }
}
```

A session is only marked `finished` when the session had ended **and** every
query for it came back cleanly. If one request failed, the session stays
"live": the next run retries it and the app keeps its 12 h rule for it, instead
of serving a partial file forever.

A meeting-scoped query (`starting_grid`, `drivers?meeting_key=`) is finished
only when **every** session of that meeting is finished — a grid fetched after
FP1 is empty, and caching that forever would be a lie.

Without a manifest (a developer's machine that never ran the job) the file is
just a cache again: 12 h for live data, refetch when stale. Nothing changes
for anyone who does not opt in.

### Two ways to load

```python
import artifacts

# One query, on demand: disk first, API only when missing or stale.
rows = artifacts.load_json('laps', session_key=9158)

# A status page, measured on disk:
status = artifacts.cache_status()
# {'root', 'exists', 'files', 'bytes', 'oldest', 'newest',
#  'sessions_indexed', 'archive_active', 'manifest_generated_at',
#  'last_run', 'endpoints'}
```

`load_json(..., allow_network=False)` never touches the network: it returns the
stale copy if one exists (for an offline demo) or `None`.

### The whole-app hook

Every data module reaches OpenF1 through `openf1_client.cached_json`, each
passing its own TTL (a day for a finished session). `artifacts.activate()`
wraps that function once: when the query belongs to a session the manifest
marks finished and the file exists, the file wins, however old it is;
everything else is delegated unchanged, so live sessions and cold misses behave
exactly as before. `deactivate()` restores the client's own function (tests and
tools that want the raw client).

The recommended integration in `src/shared.py` is two lines: `import artifacts`
with the other imports, and `artifacts.activate()` after the import block. That
one call covers the season frame, the session fallback, and the telemetry
ingest, because all of them go through the same client function.

---

## 3. Refreshing locally

```bash
# One season, as the nightly job does it (incremental: existing files are kept)
PYTHONPATH=src .venv/Scripts/python tools/refresh_cache.py --years 2026

# See the plan without a single request and without writing anything
PYTHONPATH=src .venv/Scripts/python tools/refresh_cache.py --dry-run --years 2026

# The four seasons OpenF1 has, first time (a backfill, ~1.5-2 h: see below)
PYTHONPATH=src .venv/Scripts/python tools/refresh_cache.py --years 2023 2024 2025 2026

# Only the last six started sessions of a season (a quick catch-up)
PYTHONPATH=src .venv/Scripts/python tools/refresh_cache.py --years 2026 --session-limit 6

# Fill a second archive without touching the app's cache
PYTHONPATH=src .venv/Scripts/python tools/refresh_cache.py --years 2026 --out /tmp/f1-archive
```

`--years 2023` and older is skipped with a note: OpenF1's archive starts with
the 2023 season.

The script prints a line per session, a per-endpoint table per year, and a
final tally:

```
Final tally
  network requests : 21
  cache hits       : 480
  empty results    : 2
  transport errors : 0
  rows read        : 142384
  bytes written    : 3.4 MB
  archive on disk  : 215 files, 4.7 MB
  sessions indexed : 31
  elapsed          : 1m 12s
```

A 404 is not a failure: OpenF1 answers 404 for a query that matches nothing
(a driver who never set a lap, a session without weather). The client turns it
into `[]`, the job counts it under `empty`, and the run continues. Transport
errors are counted under `errors` and the session is left un-finished so the
next run retries it. The job only exits non-zero when **every** requested year
failed, so a green run in CI means the archive was actually touched.

### How long a backfill takes

The client's rate limiter is shared and published at 30 requests/minute: no
second limiter, no way to go faster. One started session costs 6-8 requests
(6 session endpoints + 2 meeting endpoints for the first session of a weekend),
so a 110-session season is roughly **25-30 minutes** of wall clock, and the
four archived seasons are about **1.5-2 hours**. That is why:

* the nightly job is **incremental** — a finished session is already on disk
  and costs a cache hit, not a request, so a normal night is a handful of
  requests;
* `--session-limit` exists for a quick catch-up when only the latest weekend
  matters;
* the GitHub job has a 6-hour timeout.

---

## 4. The scheduled job

`.github/workflows/refresh-data.yml` runs at **03:17 UTC** every day and can be
started by hand (`workflow_dispatch`, with optional `years` and
`session_limit`).

1. It restores the previous archive from the `data-cache` branch, so the run is
   incremental.
2. It installs `requirements.txt` on Python 3.12 and runs the script for the
   current and previous season.
3. It force-pushes the refreshed directory to the `data-cache` branch as **one
   commit on top of the default branch**, so the branch is a snapshot rather
   than a 365-commit-a-year history. Nothing is lost: the previous snapshot was
   restored in step 1, and the run only publishes when the cache is non-empty
   and the job succeeded.

A branch was chosen over an uploaded artifact because the app reads files from
disk at runtime: an artifact is a zip a deploy job would have to download and
unpack, it expires after 90 days, and two deploy paths could disagree about
which archive is live. A branch is an ordinary git remote:

```bash
# On the deployment, before the app starts:
git clone --depth 1 --branch data-cache <repo> _archive
cp -r _archive/f1_cache/openf1 ./f1_cache/          # or symlink it
# or, from an existing checkout:
git archive data-cache f1_cache | tar -x
```

`f1_cache/` is in `.gitignore` (it is a cache and FastF1 writes into it too),
which is why the job uses `git add -f f1_cache/openf1`: the data branch is the
one place where the directory is tracked on purpose.

---

## 5. Licence: OpenF1 data is CC BY-NC-SA 4.0

The archive is a **derivative work** of data published by
[OpenF1](https://openf1.org), which is licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The consequences are not optional, and they apply to the cache as much as to a
live request:

* **Non-commercial only.** The cache may not be used in, or shipped with, a
  commercial product or a paid service. A demo, a research notebook, a
  non-commercial dashboard: fine. A SaaS product: not fine.
* **Attribution.** Anything that shows or republishes the data must credit
  OpenF1 (name, link to <https://openf1.org>, and a link to the licence). The
  app's Home page "About & Limitations" block carries the public-data framing;
  a page that reads the archive should keep the same credit.
* **Share-alike.** Publishing the `data-cache` branch is redistribution: the
  published archive stays under CC BY-NC-SA 4.0, and a modified archive must be
  offered under the same terms. Do not relicense it.
* **No endorsement.** The licence grants no rights to the F1 name, teams,
  drivers, or circuit marks.

FastF1 (the timing archive the app also uses) has its own licence; this
document does not change it. The refresh job only touches OpenF1.

If the deployment is ever commercial, the artifact path has to be dropped and
the app must read OpenF1 directly under its own terms — that is why
`artifacts.activate()` is a single call rather than something woven into every
data module.

---

## 6. Known limits

* **The archive is only as complete as the last run.** A session that OpenF1
  ingests late is picked up by the next run; a session that is currently
  running is re-fetched on every run (its files are never marked finished until
  it has ended and every query succeeded).
* **A past season's session list is treated as immutable.** If OpenF1 adds a
  round to a finished season after the archive was built, a normal re-run will
  not notice (the list is served from disk forever). Delete the file or run
  with `--out` into a fresh directory to re-enumerate that season.
* **A stale file is still a file.** With `activate()` the archive wins for a
  finished session even if a newer revision existed; "finished and cleanly
  archived" is the contract, and the manifest records the time it was built.
* **The manifest is written by the job, not by the app.** A deployment that
  copies only `f1_cache/openf1/*.json` and forgets `_index/` loses the
  "finished forever" rule and falls back to the 12 h policy. Copy the whole
  directory.
* **Disk, not memory.** `@st.cache_data` still fronts each page; the archive
  only removes the network from the path, not Streamlit's own caching.
