# -*- coding: utf-8 -*-
"""The rain-radar panel: a self-contained map inside an iframe component.

The panel follows the replay player's pattern (see ``pages/5_Replay.py``): the
payload arrives as one JSON blob and the widget draws itself, so a pan, a zoom
or a frame change never triggers a Streamlit rerun.

Rendering the map in the browser is also what keeps the tiles off the server.
The Streamlit process only asks RainViewer for the 1.5 KB index document (see
``wx.radar_frames``); the image tiles are requested by the reader's browser for
the frame that is actually shown.

The template is a plain asset file with a dark palette that mirrors
``assets/style.css``, because an iframe cannot inherit the page's stylesheet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import styles

# The panel asset, next to the replay player it is styled after.
TEMPLATE = Path(__file__).parent / 'assets' / 'radar_scope.html'

# The iframe height lives with the other sizes in styles.py.
HEIGHT = styles.RADAR_HEIGHT


def radar_html(payload: Dict[str, Any],
               template: Optional[str] = None) -> str:
    """Fill the radar template with the payload, inlined as JSON.

    ``<``, ``>`` and ``&`` are escaped so that no value can close the script
    block early, the same rule ``replay_payload.player_html`` applies to the
    player. The template escapes every value again before it reaches the DOM,
    because the frame paths come from a third-party API.
    """
    mark = template if template is not None else TEMPLATE.read_text(
        encoding='utf-8')
    blob = json.dumps(payload)
    blob = (blob.replace('<', '\\u003c').replace('>', '\\u003e')
                .replace('&', '\\u0026'))
    return mark.replace('__PAYLOAD__', blob)
