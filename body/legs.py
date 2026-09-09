"""Leg naming, shared vocabulary between the body and the bridge.

Lives in `body/` because legs are a body concept. `bridge/` imports it from
here; `body/` must never import `bridge/` (that would drag torch into the
physics side).
"""

from __future__ import annotations

# NeuroMechFly leg order: left/right x fore/mid/hind.
LEG_ORDER = ("LF", "LM", "LH", "RF", "RM", "RH")
