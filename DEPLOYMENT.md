# Deploying F1 Lab

F1 Lab is one Streamlit process. There is no database and no background worker,
so "deploy" means: build an image, give it a writable cache directory, and put a
health check in front of port 8501.

Two things ship:

| What | Where | Why |
| --- | --- | --- |
| The app | `Dockerfile` → Render / Fly.io / any container host | the interactive dashboard |
| The race pages | `tools/export_static.py` → `site/` → GitHub Pages | one permanent HTML page per finished race: linkable, indexable, free |

---

## 1. Build and run locally

```bash
docker build -t f1lab .
docker run --rm -p 8501:8501 -v "${PWD}/docker-cache:/data/f1_cache" f1lab
# http://127.0.0.1:8501
```

Without docker:

```bash
pip install -r requirements.txt
PYTHONPATH=src streamlit run src/f1.py
```

The image runs as UID 10001 (not root) and writes only to `/data/f1_cache`
(OpenF1 responses) and `/app/logs`. Mount `/data` to keep the cache across
restarts; without a mount the container works, it just re-downloads.

## 2. Environment variables

| Variable | Default in the image | Meaning |
| --- | --- | --- |
| `F1LAB_CACHE_DIR` | `/data/f1_cache` | where OpenF1 responses are cached (see `src/openf1_client.py`) |
| `STREAMLIT_SERVER_PORT` | `8501` | listen port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | bind address |
| `STREAMLIT_SERVER_HEADLESS` | `true` | never try to open a browser |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | no telemetry |

No secrets are required: every data source is public and keyless. `.streamlit/config.toml`
already pins the dark theme, the port and `enableXsrfProtection = true`; the
image's environment only overrides what a host needs to change.

## 3. Health check

```bash
curl -fsS http://127.0.0.1:8501/_stcore/health   # -> "ok"
```

Streamlit answers 200 on `/_stcore/health` only once the server accepts
sessions, so a cold container is never marked healthy early. The Dockerfile
uses the Python standard library for this (a previous version called `curl`,
which `python:*-slim` does not ship, so the check could never pass).

## 4. Render

`deploy/render.yaml` is a blueprint: Render dashboard → **New** → **Blueprint**
→ pick the repository. It sets `F1LAB_CACHE_DIR`, the health check path and
automatic deploys from `main`.

On the free plan the filesystem is ephemeral, so every restart re-downloads the
season (about two minutes for 2025/2026 from a cold cache). A disk needs a paid
instance type; the blueprint carries the two lines commented out. Pages are
then warm in well under a second, which is the number `tools/latency_check.py`
holds the app to.

## 5. Fly.io

```bash
fly launch --copy-config --config deploy/fly.toml --no-deploy
fly volumes create f1lab_cache --size 1 --region sin
fly deploy --config deploy/fly.toml --dockerfile Dockerfile
```

`deploy/fly.toml` scales to zero between sessions (`auto_stop_machines = "stop"`),
runs one 1 GB machine in Singapore, mounts the cache volume at `/data` and
probes `/_stcore/health` every 30 s.

## 6. The static race pages

```bash
PYTHONPATH=src python tools/export_static.py --year 2026 --out site
```

Writes `site/index.html` plus one page per finished race
(`site/2026-17-singapore-grand-prix.html`). The pages contain the weekend
report, the championship snapshot and the season index; each is a single
self-contained file that loads Plotly from a CDN. Nothing is invented: when a
season has no results the tool exits with code 2 and writes nothing.

`.github/workflows/deploy.yml` exports both seasons on every push to `main` and
publishes `site/` to GitHub Pages (Settings → Pages → Source: GitHub Actions).

## 7. CI

`.github/workflows/tests.yml` runs ruff, mypy and pytest on Python 3.10–3.12.
`.github/workflows/deploy.yml` adds:

1. `docker build`, then start the container and poll `/_stcore/health` for 60 s.
2. Request `/` and require the Streamlit shell in the response.
3. Push `ghcr.io/<owner>/<repo>:latest` (main only).
4. Export the static site, upload it as an artifact, deploy it to Pages (main only).

A build that cannot answer its own health endpoint fails the workflow instead of
quietly shipping.

## 8. Honest limitations

- **Data licence.** Results and timing come from the OpenF1 API
  (CC BY-NC-SA 4.0). Anything built on it is **non-commercial**; do not put ads
  or a paywall on a deployment.
- **Upstream dependency.** If OpenF1 is rate limited or down, pages degrade to
  their "no data" state rather than inventing numbers. Clear the cache with
  `rm -rf /data/f1_cache` to force fresh requests.
- **Warm vs cold.** The first request after a restart downloads a season; the
  next ones are served from the cache. Size the host for the cold path.
- **Not official timing.** The app says so on every page; a deploy does not
  change that, and it must not be presented as a team or FIA tool.

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| container starts, health never 200 | port mismatch | `STREAMLIT_SERVER_PORT` must match the mapped port |
| "Connection error" in the browser | the host proxies the container's port, not Streamlit's | point the host at 8501 and keep websockets enabled |
| every page ready in ~2 min, then fast | cold cache | mount `/data`, or accept the first-load cost |
| `PermissionError: /data/f1_cache` | volume owned by another UID | `chown -R 10001:10001 /data` on the volume |
| static site is empty | the season has no finished races | run with `--year` of a finished season |
