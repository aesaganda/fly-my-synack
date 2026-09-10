"""Encode body state back into current injected at the network's input side.

APPROXIMATION NOTICE
--------------------
This is the weakest link in the whole pipeline and is labelled as such in the
README. Real flies have ~6,370 `vnc_sensory` neurons with specific modalities
and receptive fields. Here:

  * per-leg mechanical load - ground reaction plus actuator effort - is
    injected into that leg's own motor-neuron pool, standing in for
    proprioceptive/campaniform feedback without modelling any of the actual
    sensory afferents. Load rather than ground contact specifically, so the
    signal survives swimming, where nothing touches the floor;
  * a scalar "air motion" term stands in for Johnston's organ.

When the loaded subset does contain `vnc_sensory` neurons they are used as the
injection target instead. Otherwise the target is descending neurons, which is
anatomically backwards but keeps the loop closed. Either way the preset's
sensory gains are what change between environments.
"""

from __future__ import annotations

import numpy as np
import torch

from brain.lif import LIFNetwork
from body.legs import LEG_ORDER
from bridge.decode import MotorDecoder

SENSORY_SUPERCLASS = "vnc_sensory"

# Converts per-leg mechanical load (model units, uN) to injected current.
# Rescaled from 0.02 when the signal changed from ground contact alone (~86 per
# leg on land) to contact plus actuator effort (~142), so that walking on dry
# land receives about the same drive as before. PLACEHOLDER.
CONTACT_GAIN = 0.012
AIR_MOTION_GAIN = 0.001  # per mm/s of relative air speed. PLACEHOLDER.


class SensoryEncoder:
    def __init__(self, net: LIFNetwork, decoder: MotorDecoder) -> None:
        self.net = net
        self.decoder = decoder
        self.buffer = torch.zeros(net.n, device=net.device)

        n = net.tables.neurons
        sensory = n[n["superclass"] == SENSORY_SUPERCLASS]
        self.has_real_sensory = len(sensory) > 0
        self.air_target = (
            net.indices_for(sensory["bodyId"]) if self.has_real_sensory else decoder.dn_all
        )

    def encode(
        self,
        leg_load: np.ndarray,
        air_speed: float,
        mechanosensory_gain: float,
        johnstons_organ_gain: float,
    ) -> torch.Tensor:
        """Return the per-neuron external current for this step.

        leg_load:  (6,) mechanical load per leg, in LEG_ORDER order - ground
                   reaction plus actuator effort, so it is non-zero when
                   swimming as well as when walking.
        air_speed:      scalar magnitude of body velocity relative to the medium.
        """
        self.buffer.zero_()

        for leg, force in zip(LEG_ORDER, np.asarray(leg_load, dtype=float)):
            idx = self.decoder.leg_groups[leg]
            if len(idx):
                self.buffer[idx] += float(force) * CONTACT_GAIN * mechanosensory_gain

        if len(self.air_target):
            self.buffer[self.air_target] += (
                float(air_speed) * AIR_MOTION_GAIN * johnstons_organ_gain
            )
        return self.buffer
