# -*- coding: utf-8 -*-
"""Data-source priority and provenance (no UI, no network).

The single place that says which free source answers which question
(see ``docs/DATA_SOURCES.md`` for the rationale). Pages ask the
services in ``src/shared.py`` / ``src/jolpica.py``; this module only
records the priority order and the honesty labels, so changing a
priority never requires editing a page.

Datasets are never merged blindly: every result carries the
``source_name`` and a ``data_status`` (one of :data:`DATA_STATUSES`)
so a measured value can never be silently replaced by a forecast.
"""

SOURCE_PRIORITY = {
    # dataset: (primary, fallback)
    "race_results": ("FastF1", "Jolpica"),
    "lap_times": ("FastF1", "OpenF1"),
    "telemetry": ("FastF1", "OpenF1"),
    "tyre_stints": ("FastF1", "OpenF1"),
    "pit_stops": ("OpenF1", "Jolpica"),
    "standings": ("Jolpica", "computed_from_results"),
    "weather_history": ("FastF1", "OpenF1"),
    "weather_forecast": ("Open-Meteo", "Unavailable"),
    "rain_radar": ("RainViewer", "Unavailable"),
    "race_replay": ("OpenF1/FastF1", "position_data_if_available"),
    "championship_points": ("OpenF1_results", "Jolpica"),
}

# Data-status labels used across the app (see docs/DATA_SOURCES.md).
DATA_STATUSES = (
    "Verified",     # measured, cross-checked or official-style source
    "Available",    # measured, single source
    "Partial",      # some fields/rounds missing
    "Estimated",    # derived estimate, labelled as such in the UI
    "Unavailable",  # no source could provide the data
)

# Free-tier limits that shape the caching policy (no paid plans, no keys).
SOURCE_LIMITS = {
    "OpenF1": "3 req/s, 30 req/min (free historical tier, 2023+)",
    "FastF1": "official timing archive; session load is cached on disk",
    "Jolpica": "free Ergast successor; historical tables cached on disk",
    "Open-Meteo": "free non-commercial forecast, hourly (15-min where offered)",
    "RainViewer": "free radar index; frames may be unavailable per region",
}


def priority(dataset: str):
    """``(primary, fallback)`` for a dataset key, or ``(None, None)``."""
    return SOURCE_PRIORITY.get(dataset, (None, None))


def provenance(dataset: str, source_name: str, data_status: str) -> dict:
    """Provenance record attached to a returned dataset.

    ``data_status`` must be one of :data:`DATA_STATUSES`; anything else
    raises so a typo can never produce an unlabelled dataset.
    """
    if data_status not in DATA_STATUSES:
        raise ValueError(f"unknown data status: {data_status!r}")
    primary, fallback = priority(dataset)
    return {
        "dataset": dataset,
        "source_name": source_name,
        "data_status": data_status,
        "primary_source": primary,
        "fallback_source": fallback,
    }
