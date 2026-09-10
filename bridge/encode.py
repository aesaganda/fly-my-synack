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
  * a scalar "air motion" term stands in for Johnston's organ;
  * bilateral olfaction, sampled at the two antennae, is injected into the LEFT
    and RIGHT descending populations - which is what turns the fly toward a
    smell, because `MotorDecoder` reads its turn command from exactly that
    left/right rate difference.

When the loaded subset does contain `vnc_sensory` neurons they are used as the
mechanosensory/air-motion target instead. Otherwise the target is descending
neurons, which is anatomically backwards but keeps the loop closed. Either way
the preset's sensory gains are what change between environments.

Olfaction is the one channel that goes to descending neurons ON PURPOSE rather
than as a fallback: the olfactory receptor neurons are in the antennae and the
antennal lobe, nowhere near the VNC, so `vnc_sensory` would be the wrong target
for a smell. Injecting at the DNs skips the entire antennal lobe / lateral horn
/ mushroom body pathway and pretends the descending command already carries the
olfactory decision. That is a big shortcut and it is the mechanism behind
everything the fly does in the kitchen world - see ODOUR_BILATERAL_GAIN.
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

# ---- olfaction -----------------------------------------------------------
#
# The two antennae are 0.28 mm apart. In a plume a centimetre wide that is a
# left/right concentration difference of at most ~1% of the concentration
# itself, so the
# differential and the common-mode parts of the signal are separated here and
# given their own gains. Lumping them together does not work: a gain large
# enough for the difference to steer saturates every DN with the common part.
#
# A real fly closes most of that gap with time rather than space - it casts its
# head and body and compares successive samples - which a body with no neck
# joint cannot do. ODOUR_BILATERAL_GAIN stands in for that missing temporal
# comparison, so it is deliberately large. PLACEHOLDER.
#
# Swept in-situ in the kitchen world, 40k steps, two seeds, measuring how close
# the fly gets to a source (dmin) and where it ENDS UP (dend) - the second
# number is the one that matters, since a fly that brushes past a smell has not
# tracked it:
#     gain   dmin mm      dend mm      mean |turn|
#        0   13.6 / 15.1  25.5 / 25.9     0.02     <- no olfaction: it walks past
#       12   13.3 / 13.4  26.8 / 27.3     0.09
#       30   13.0 / 13.4  24.4 / 24.3     0.23     <- steers, does not arrive
#       60   10.2 / 12.2  10.2 / 13.5     0.29     <- chosen
#      120   17.8 /  7.0  19.7 /  7.8     0.48     <- over-steers; seed-dependent
# It settles about a plume-width out rather than sitting on the source, which
# is what a gradient with no peak-holding mechanism does.
ODOUR_BILATERAL_GAIN = 60.0
# Common-mode: "there is a smell here", raising arousal a little. Small, because
# its only job is to keep the fly moving in a plume, not to steer it.
ODOUR_COMMON_GAIN = 0.05
# Receptor adaptation. Real ORNs respond to CHANGE and habituate to a steady
# background, and without that the fly orbits the first source it finds for
# ever: at the peak the gradient reverses on every pass and pulls it back in.
# Adaptation is what lets it arrive, lose interest, and go and find the next
# one - so this is the mechanism that turns chemotaxis into exploration.
# PLACEHOLDER: real ORN adaptation is multi-timescale and this is one leaky
# integrator.
ODOUR_ADAPT_TAU_S = 1.2


class SensoryEncoder:
    def __init__(self, net: LIFNetwork, decoder: MotorDecoder, dt_s: float = 1e-4) -> None:
        self.net = net
        self.decoder = decoder
        self.buffer = torch.zeros(net.n, device=net.device)
        # One leaky integrator per antenna, in the encoder rather than the world
        # because adaptation is a property of the receptor, not of the smell.
        self._odour_baseline = np.zeros(2)
        self._adapt_alpha = float(min(1.0, dt_s / ODOUR_ADAPT_TAU_S))
        self.last_odour = np.zeros(2)
        self.last_odour_adapted = np.zeros(2)

        n = net.tables.neurons
        sensory = n[n["superclass"] == SENSORY_SUPERCLASS]
        self.has_real_sensory = len(sensory) > 0
        self.air_target = (
            net.indices_for(sensory["bodyId"]) if self.has_real_sensory else decoder.dn_all
        )

    def reset(self) -> None:
        """Forget what the antennae have been smelling."""
        self._odour_baseline = np.zeros(2)
        self.last_odour = np.zeros(2)
        self.last_odour_adapted = np.zeros(2)

    def _olfaction(self, odour_lr: np.ndarray, gain: float) -> tuple[float, float]:
        """Adapt, then split into common and differential drive.

        Returns the current to add to the LEFT and RIGHT descending pools. A
        stronger smell on the right raises the right pool, which is what
        `MotorDecoder.decode` reads as a right turn.
        """
        c = np.asarray(odour_lr, dtype=float).reshape(2)
        self._odour_baseline += self._adapt_alpha * (c - self._odour_baseline)
        adapted = c - self._odour_baseline
        self.last_odour = c
        self.last_odour_adapted = adapted

        common = ODOUR_COMMON_GAIN * float(adapted.mean()) * gain
        half_diff = 0.5 * float(adapted[1] - adapted[0]) * ODOUR_BILATERAL_GAIN * gain
        return common - half_diff, common + half_diff

    def encode(
        self,
        leg_load: np.ndarray,
        air_speed: float,
        mechanosensory_gain: float,
        johnstons_organ_gain: float,
        odour_lr: np.ndarray | None = None,
        olfactory_gain: float = 1.0,
    ) -> torch.Tensor:
        """Return the per-neuron external current for this step.

        leg_load:  (6,) mechanical load per leg, in LEG_ORDER order - ground
                   reaction plus actuator effort, so it is non-zero when
                   swimming as well as when walking.
        air_speed:      scalar magnitude of body velocity relative to the medium.
        odour_lr:       (2,) concentration at the left and right antenna, or
                        None in a world with nothing to smell.
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

        if odour_lr is not None:
            left, right = self._olfaction(odour_lr, olfactory_gain)
            if len(self.decoder.dn_left):
                self.buffer[self.decoder.dn_left] += left
            if len(self.decoder.dn_right):
                self.buffer[self.decoder.dn_right] += right
        return self.buffer
