# OpenF1 golden fixtures

Small, trimmed copies of the OpenF1 responses this app reads. They are the
reference snapshot that `tests/test_api_contracts.py` and
`tools/contract_check.py --offline` validate against, so a payload change is
caught by an offline test instead of a page breaking in production.

The contract itself (which fields the app depends on, and their types) lives in
`tools/contract_check.py` - `ENDPOINTS`. This README only records provenance.

## Provenance

All files were captured from the public OpenF1 v1 API (`https://api.openf1.org/v1`)
on **2026-10-22 (UTC)**, 2025 season. No credentials, cookies or headers are
stored: these are the unauthenticated public responses. Only public sporting
data is present (session/driver/team names, lap times, telemetry values); the
`headshot_url` values are the public CDN links the API itself returns.

Every file is a JSON **list of objects** (OpenF1 paging shape), `indent=1`,
`ensure_ascii=False`, at most 40 rows.

| endpoint        | file                 | rows | query used                                        | trimming |
| --------------- | -------------------- | ---- | ------------------------------------------------- | -------- |
| `sessions`      | `sessions.json`      | 10   | `year=2025`                                       | kept the two meetings below (1254, 1255) only; all session types kept |
| `meetings`      | `meetings.json`      | 2    | `year=2025`                                       | kept meetings 1254 (Melbourne) and 1255 (Shanghai) |
| `session_result`| `session_result.json`| 20   | `session_key=9998` (Shanghai race)                | full classification kept; includes DNS rows (`position`/`number_of_laps` null) and a `'+1 LAP'` string gap |
| `starting_grid` | `starting_grid.json` | 40   | `meeting_key=1255`                                | full response: the meeting has **two** grids (sprint qualifying 9989, qualifying 9994), 20 rows each |
| `drivers`       | `drivers.json`       | 40   | `meeting_key=1255`                                | kept sessions 9993 (sprint) and 9998 (race), 20 drivers each |
| `laps`          | `laps.json`          | 40   | `session_key=9998`                                | first 40 rows (out laps of the race start) |
| `stints`        | `stints.json`        | 40   | `session_key=9998`                                | first 40 rows of the race |
| `pit`           | `pit.json`           | 25   | `session_key=9998`                                | full response; includes `stop_duration` null |
| `weather`       | `weather.json`       | 40   | `session_key=9998`                                | first 40 rows (start of the session) |
| `intervals`     | `intervals.json`     | 40   | `session_key=9998`                                | 36 rows from one mid-race window + 2 `'+1 LAP'` rows + 2 null-gap rows (all three gap shapes are represented) |
| `race_control`  | `race_control.json`  | 40   | `session_key=9998`                                | first 20 + last 20 rows, plus one real `Drs` row; categories present: `Flag`, `Other`, `Drs` |
| `position`      | `position.json`      | 40   | `session_key=9998`                                | first 40 rows (start of the race, chronological) |
| `car_data`      | `car_data.json`      | 40   | `session_key=9998&driver_number=1`                | first 40 samples of one car |

## Invariants these fixtures pin

- A sprint is still identifiable: `sessions.json` contains
  `session_key=9993` with `session_type='Race'` **and** `session_name='Sprint'`.
  This is exactly the shape that broke the app once; `_is_sprint` and
  `sprint_sessions` are covered by the contract test.
- `session_result.gap_to_leader` is polymorphic - `float`, `int` (`0`), the
  string `'+1 LAP'` and `null`. The parsers must accept all four.
- `starting_grid`, `drivers`, `weather` etc. keep all their keys on every row;
  a field that disappears (rather than arriving as `null`) is a breaking change.
- `starting_grid` for a sprint weekend is two grids, not one: the meeting
  response holds a separate grid per qualifying session (`9989` sprint
  qualifying, `9994` qualifying). There is no single "race grid" row set.
- `pit` rows carry `pit_duration` and `stop_duration`; the session fallback
  prefers `stop_duration` and falls back to `pit_duration`.

## Updating

- **Do not hand-edit values.** Re-capture with the shared client (it owns the
  process rate limiter) so the fixtures stay real responses:

  `PYTHONPATH=src .venv/Scripts/python.exe -c "import json; from openf1_client import openf1; print(json.dumps(openf1('sessions', year=2025)[:10], indent=1, ensure_ascii=False))"`

  then re-apply the same trimming described above and keep each file under 40 rows.
- If the upstream shape genuinely changed, update the contract in
  `tools/contract_check.py` (`ENDPOINTS`) in the same change - but only after
  checking that the app code still depends on the field.
- Verify with:

  `PYTHONPATH=src .venv/Scripts/python.exe tools/contract_check.py --offline`

  and `python tools/contract_check.py --live` when investigating a suspected
  upstream change (live mode hits the API through the shared rate limiter).
