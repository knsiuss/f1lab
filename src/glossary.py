# -*- coding: utf-8 -*-
"""
glossary.py
~~~~~~~~~~~
Plain-language definitions for the Formula 1 words the app uses.

The app is read by people who do not follow the sport closely, so a label like
"Tyre degradation" or "Rolling average" has to explain itself. These
definitions are the single source for that wording: a page shows the short
version under a section title, and the glossary at the bottom of the page
shows the full list.

Nothing here is a fact about a specific season -- it is the vocabulary of the
sport, kept in one place so every page speaks the same way.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

from typing import Dict, List, Optional, Sequence

# The full vocabulary. ``short`` is the one-line version used under a section
# title; ``long`` is the explanation used in the glossary.
TERMS: Dict[str, Dict[str, str]] = {
    "race": {
        "short": "The main race on Sunday. Its points count towards the championship.",
        "long": "A Grand Prix is the main event of a race weekend. Every driver gets "
                "points for the finishing order, and the driver and the team with the "
                "most points at the end of the season win the championship.",
    },
    "sprint": {
        "short": "A short race on Saturday. It gives points, but fewer than the main race.",
        "long": "A sprint is a shorter race (roughly 100 km) held on Saturday. It awards "
                "points towards the championship, but fewer than the Grand Prix, and it "
                "does not give a grid position for the race.",
    },
    "qualifying": {
        "short": "The short session on Saturday that decides who starts where on Sunday.",
        "long": "Qualifying is a timed session on Saturday. The fastest lap decides the "
                "starting grid for the race: the quickest driver starts first (pole "
                "position), the second quickest starts second, and so on.",
    },
    "grid": {
        "short": "The starting order of the race, decided by qualifying.",
        "long": "The grid is the order in which the cars start the race. It comes from "
                "qualifying, and it is the baseline the app compares the forecast against.",
    },
    "points": {
        "short": "What the drivers and the teams are chasing. Winning is worth the most.",
        "long": "Points are awarded for finishing positions in every race and sprint. The "
                "driver with the most points wins the Drivers' Championship, the team "
                "with the most wins the Constructors' Championship.",
    },
    "stint": {
        "short": "The run of laps a driver does on one set of tyres.",
        "long": "A stint is the run of laps completed on one set of tyres before a pit "
                "stop. A driver usually does three stints in a race: a first one on the "
                "grid tyres, a long middle one, and a final one on fresh tyres.",
    },
    "pit stop": {
        "short": "A stop to change tyres. It costs a few seconds, so it is only worth "
                 "doing when the tyres are worn out.",
        "long": "A pit stop is a short stationary stop in the pit lane to change tyres. "
                "A normal stop costs a few seconds of track time, so a driver stops only "
                "when the tyre wear makes the lost time cheaper than the lost grip.",
    },
    "degradation": {
        "short": "How fast a tyre loses grip as it is used. Fast wear means more stops.",
        "long": "Tyre degradation is the rate at which a tyre loses performance as it is "
                "used. The softer the tyre, the faster it wears: a soft tyre is quick but "
                "short-lived, a hard tyre slower but usable for many laps.",
    },
    "undercut": {
        "short": "Stopping a few laps earlier than a rival to get fresher tyres and pass.",
        "long": "An undercut is a strategy: a driver stops a few laps before a rival, "
                "drives on fresher tyres and comes back in front. The reverse - staying "
                "out longer - is an overcut.",
    },
    "delta": {
        "short": "The time gap: to the car in front, or to your own best lap.",
        "long": "A delta is a difference in time. On a timing screen it shows the gap to "
                "the car in front, or the gap to the fastest lap of the session.",
    },
    "sector": {
        "short": "One third of the circuit. Comparing sector times shows where a driver "
                 "is faster or slower.",
        "long": "A circuit is split into three sectors. The time in each sector is timed "
                "separately, so the sector times of two drivers show where one of them "
                "gains or loses time.",
    },
    "rolling average": {
        "short": "The average of the last three results. A steadier number than one race.",
        "long": "A rolling average smooths the last three results into one number. It "
                "describes a run of form rather than a single lucky or unlucky race, and "
                "a lower number is better.",
    },
    "pace": {
        "short": "How fast a car goes on a lap, once the out-lap and the traffic are "
                 "removed.",
        "long": "Pace is the lap time a driver can hold. It is measured on clean laps "
                "only: the first lap out of the pits, the lap with the safety car, and "
                "laps that are not accurate are left out.",
    },
    "estimated": {
        "short": "Calculated from the data, not measured directly.",
        "long": "Anything in this app that is derived rather than reported is marked "
                "'estimated'. It comes with a confidence level that reflects how many "
                "clean laps the calculation was based on.",
    },
    "confidence": {
        "short": "How much data an estimate is based on.",
        "long": "The confidence level follows the size of the sample: 'good' means at "
                "least 15 clean laps, 'moderate' at least 8, 'low' at least 2. Below "
                "that the app says 'insufficient data' instead of guessing.",
    },
    "backtest": {
        "short": "The model is run over past races, learning only from earlier rounds, "
                 "and compared with a simple baseline.",
        "long": "A backtest replays the forecast over the races of a season. For every "
                "race the model only sees the rounds before it, so it cannot cheat by "
                "looking at the result it is predicting. The result is compared with a "
                "baseline that simply takes the starting order as the result.",
    },
    "baseline": {
        "short": "The simple comparison: the driver who starts first finishes first.",
        "long": "The baseline says the starting order decides the finishing order. A "
                "model is only worth using when it beats that, and the app says so "
                "plainly when it does not.",
    },
    "pit window": {
        "short": "The range of laps in which stopping is expected to be worth it.",
        "long": "The pit window is the stretch of laps where a driver loses less by "
                "stopping than by staying out on worn tyres. The app searches for it "
                "instead of assuming a fixed plan.",
    },
    "replay": {
        "short": "The animated view of the race: where every car was on track.",
        "long": "The replay shows the position of every car through a session, estimated "
                "from the public position feed. It is a position replay, not an official "
                "timing or on-board video.",
    },
    "setup fingerprint": {
        "short": "A rough reading of how a car was set up, from the way it turns in.",
        "long": "The setup fingerprint is a proxy, not a setup sheet: it groups the "
                "circuit into corners and compares how the car behaved in each of them "
                "to guess whether it was set up for a high-downforce or a low-downforce "
                "trim. It is an estimate from public telemetry.",
    },
    "h2h": {
        "short": "Head to head: the two drivers' record when they are compared directly.",
        "long": "A head-to-head record counts only the races the two drivers were in "
                "together. It is the clearest way to compare two drivers, because the "
                "weather, the track and the rules are the same for both of them.",
    },
    "podium": {
        "short": "The first three finishers of a race. Podium points are the biggest "
                 "prize in a race.",
        "long": "The podium is the top three finishing positions. The drivers on it "
                "collect the largest points of the race: 25 for the win, 18 for second "
                "and 15 for third.",
    },
    "championship": {
        "short": "The title: whoever has the most points when the season ends.",
        "long": "There are two championships. The Drivers' Championship goes to the "
                "driver with the most points; the Constructors' Championship goes to "
                "the team whose two drivers score the most points together.",
    },
    "fastest lap": {
        "short": "The quickest single lap of a session. It gives one extra point.",
        "long": "The fastest lap is the quickest single lap of a race. The driver who "
                "sets it takes one extra point, if they finish in the top ten.",
    },
    "tyre": {
        "short": "The rubber the car runs on. Soft tyres are quick but wear out fast.",
        "long": "Tyres come in compounds: soft, medium and hard. A soft tyre gives the "
                "most grip for a few laps and then wears off; a hard tyre lasts the "
                "whole race but is slower. Drivers switch between them in the pits.",
    },
    "constructor": {
        "short": "The team that enters the car. Two drivers score points for one team.",
        "long": "A constructor is the team behind the car - Mercedes, Ferrari, Red Bull "
                "Racing. Its two drivers share the constructors' points, so a team can "
                "win the title even when one of its drivers scores less than a rival.",
    },
    "clean lap": {
        "short": "A lap that counts for comparison: no pit lane, no safety car, no "
                 "mistake.",
        "long": "A clean lap is a lap with a valid time on a green track. The lap out "
                "of the pits, the lap behind the safety car and any lap the timing "
                "system flags as inaccurate are left out, because they are not "
                "representative of the car's speed.",
    },
    "forecast": {
        "short": "A guess about how the next race will end, worked out from the "
                 "starting order, the team's recent form and the driver's last results.",
        "long": "The forecast is a transparent model, not a black box: a "
                "regression over the starting order, how the team has been doing "
                "this season, the driver's last few results, the places gained on "
                "track and how reliably the driver finishes. It is run back over "
                "past races to see whether it beats simply copying the starting "
                "order, and the app says so plainly when it does not.",
    },
    "risk": {
        "short": "How likely it is that the forecast is wrong.",
        "long": "The risk comes from how close the numbers are: when two drivers are "
                "almost level in the model's score, either of them could finish in "
                "front of the other, and the forecast says so instead of pretending "
                "the order is certain.",
    },
    "margin": {
        "short": "The gap between the winner and the second driver, in seconds.",
        "long": "The winning margin is the time between the first and the second "
                "driver at the flag. A margin of a few tenths means the result was "
                "decided on the last lap.",
    },
}


#: The languages a reader can pick for this explanation layer.
LANGUAGES: Dict[str, str] = {'en': 'English', 'id': 'Bahasa Indonesia'}

#: What to use when nothing has been chosen, and what every helper falls back to.
DEFAULT_LANGUAGE = 'en'

try:  # the Indonesian edition is optional: without it the app stays English
    from glossary_id import TERMS as TERMS_ID  # type: ignore
except Exception:  # pragma: no cover - only when the file is missing
    TERMS_ID = {}


def language(code: Optional[str] = None) -> str:
    """Normalise a language code to one this module knows."""
    key = str(code or '').strip().lower()
    return key if key in LANGUAGES else DEFAULT_LANGUAGE


def _table(lang: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """The vocabulary table for one language (English when it is unknown)."""
    if language(lang) == 'id' and TERMS_ID:
        return TERMS_ID
    return TERMS


def _display_name(term: str, entry: Dict[str, str]) -> str:
    """The word to print for a term: the catalogue's own name when it has one."""
    if entry.get('term'):
        return entry['term']
    return 'H2H' if term == 'h2h' else term.title()


def short(term: str, default: str = '', lang: Optional[str] = None) -> str:
    """The one-line explanation of a term (used under a section title)."""
    entry = _table(lang).get(term.strip().lower())
    return entry['short'] if entry else default


def long(term: str, default: str = '', lang: Optional[str] = None) -> str:
    """The full explanation of a term (used in the glossary)."""
    entry = _table(lang).get(term.strip().lower())
    return entry['long'] if entry else default


def terms_for(names: Sequence[str],
              lang: Optional[str] = None) -> List[Dict[str, str]]:
    """Glossary entries for the given terms, skipping the ones we do not know.

    Args:
        names: the terms to look up (case-insensitive).
        lang: the language of the wording (``en`` or ``id``).

    Returns:
        A list of ``{'term', 'short', 'long'}`` dicts, in the order asked for.
    """
    table = _table(lang)
    out: List[Dict[str, str]] = []
    seen = set()
    for name in names or []:
        key = str(name).strip().lower()
        entry = table.get(key)
        if not entry or key in seen:
            continue
        seen.add(key)
        out.append({'term': _display_name(key, entry), 'short': entry['short'],
                    'long': entry['long']})
    return out


def all_entries(lang: Optional[str] = None) -> List[Dict[str, str]]:
    """Every glossary entry, alphabetically."""
    table = _table(lang)
    return [
        {'term': _display_name(term, entry), 'short': entry['short'],
         'long': entry['long']}
        for term, entry in sorted(table.items())
    ]
