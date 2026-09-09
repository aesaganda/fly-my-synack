"""Decode spiking activity into a descending drive for the CPG.

HOW MUCH OF THIS IS THE CONNECTOME
----------------------------------
Real: which neurons are grouped together. Leg motor neurons are selected by
`superclass in (vnc_motor, cb_motor)` and `subclass in (fl, ml, hl)`, split
left/right by `somaSide` - all genuine male-cns:v1.0 annotations, giving six
populations that line up with the six legs.

Not real: the mapping from those population rates to a gait. A connectome gives
connectivity, not tuned synaptic weights, so this network does not generate
walking on its own. The CPG produces the gait; the decoded rates steer it. That
is a documented hybrid, not emergent locomotion. The gains below are heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from brain.lif import LIFNetwork
from body.legs import LEG_ORDER
from brain.tables import DESCENDING, MOTOR

# male-cns subclass -> leg position. Verified against the live dataset:
# vnc_motor has fl=135, ml=116, hl=130 (plus non-leg ad/wm/nm/hm/xm).
_SUBCLASS_TO_POS = {"fl": "F", "ml": "M", "hl": "H"}

# Population rates are compared against a reference firing rate so the decoded
# drive is interpretable: a population firing at REFERENCE_RATE_HZ produces
# tanh(1) = 0.76 of full drive. PLACEHOLDER tuning knobs, not measurements.
REFERENCE_RATE_HZ = 60.0
TURN_REFERENCE_HZ = 25.0


@dataclass
class DescendingDrive:
    """What the CPG consumes."""

    forward: float = 1.0                 # overall gait speed multiplier
    turn: float = 0.0                    # -1 = full left, +1 = full right
    per_leg_gain: np.ndarray = field(default_factory=lambda: np.ones(6))
    stop: bool = False

    def as_dict(self) -> dict:
        return {
            "forward": round(float(self.forward), 4),
            "turn": round(float(self.turn), 4),
            "per_leg_gain": [round(float(g), 4) for g in self.per_leg_gain],
            "stop": bool(self.stop),
        }


class MotorDecoder:
    """Groups neurons once, then reads population rates every step."""

    def __init__(self, net: LIFNetwork, leg_order: tuple[str, ...] = LEG_ORDER) -> None:
        self.net = net
        self.leg_order = leg_order
        n = net.tables.neurons

        motor = n[n["superclass"].isin(MOTOR)]
        self.leg_groups: dict[str, torch.Tensor] = {}
        for leg in leg_order:
            side, pos = leg[0], leg[1]
            subclasses = [s for s, p in _SUBCLASS_TO_POS.items() if p == pos]
            sel = motor[motor["subclass"].isin(subclasses) & (motor["somaSide"] == side)]
            self.leg_groups[leg] = net.indices_for(sel["bodyId"])

        dns = n[n["superclass"].isin(DESCENDING)]
        self.dn_left = net.indices_for(dns[dns["somaSide"] == "L"]["bodyId"])
        self.dn_right = net.indices_for(dns[dns["somaSide"] == "R"]["bodyId"])
        self.dn_all = net.indices_for(dns["bodyId"])

        # Populated by the web UI's DN override; added to the decoded turn/forward.
        self.override: dict[str, float] = {}

        self.empty_groups = [k for k, v in self.leg_groups.items() if len(v) == 0]

    def group_sizes(self) -> dict[str, int]:
        return {
            **{k: len(v) for k, v in self.leg_groups.items()},
            "DN_L": len(self.dn_left),
            "DN_R": len(self.dn_right),
        }

    def _mean_rate(self, idx: torch.Tensor) -> float:
        """Mean firing rate of a population, in Hz."""
        if len(idx) == 0:
            return 0.0
        # `rate` is a low-passed spikes-per-step fraction; convert to Hz.
        return float(self.net.rate[idx].mean()) * 1000.0 / self.net.params.dt_ms

    def decode(self) -> DescendingDrive:
        leg_rates = np.array([self._mean_rate(self.leg_groups[l]) for l in self.leg_order])
        left = self._mean_rate(self.dn_left)
        right = self._mean_rate(self.dn_right)

        # tanh keeps a runaway network from producing an absurd drive.
        forward = float(np.tanh(leg_rates.mean() / REFERENCE_RATE_HZ)) if leg_rates.size else 0.0
        turn = float(np.tanh((right - left) / TURN_REFERENCE_HZ))

        forward = float(np.clip(forward + self.override.get("forward", 0.0), 0.0, 1.5))
        turn = float(np.clip(turn + self.override.get("turn", 0.0), -1.0, 1.0))

        per_leg = (
            1.0 + np.tanh((leg_rates - leg_rates.mean()) / REFERENCE_RATE_HZ)
            if leg_rates.size else np.ones(6)
        )
        return DescendingDrive(
            forward=forward,
            turn=turn,
            per_leg_gain=np.clip(per_leg, 0.2, 1.8),
            stop=bool(self.override.get("stop", 0.0) > 0.5),
        )

    def set_override(self, **kwargs: float) -> None:
        """Web-UI debug lever: bias the decoded drive directly.

        Not a biological mechanism - it exists so the demo does not depend on
        waiting for emergent behaviour.
        """
        self.override.update({k: float(v) for k, v in kwargs.items()})

    def clear_override(self) -> None:
        self.override.clear()
