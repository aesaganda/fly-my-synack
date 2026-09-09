"""Six coupled phase oscillators producing a tripod gait.

This is the part that actually generates locomotion. The connectome does not -
it steers. Keeping the CPG here, small and readable, rather than reaching into
`flygym_demo.complex_terrain.turning_controller`, means the gait does not break
when FlyGym reorganises its example modules.

Tripod gait: (LF, RM, LH) step together, antiphase to (RF, LM, RH).
"""

from __future__ import annotations

import numpy as np

from body.legs import LEG_ORDER

# Antiphase grouping. Index into LEG_ORDER = (LF, LM, LH, RF, RM, RH).
_TRIPOD_A = (0, 4, 2)  # LF, RM, LH
_TRIPOD_B = (3, 1, 5)  # RF, LM, RH

BASE_FREQ_HZ = 18.0   # within the ~10-20 Hz stride frequency of a walking fly
COUPLING = 8.0        # phase-locking strength between legs


class TripodCPG:
    def __init__(self, dt_s: float, base_freq_hz: float = BASE_FREQ_HZ, seed: int = 0) -> None:
        self.dt = dt_s
        self.base_freq = base_freq_hz
        rng = np.random.default_rng(seed)

        self.phase = np.zeros(6)
        self.phase[list(_TRIPOD_B)] = np.pi
        self.phase += rng.normal(0.0, 0.05, 6)  # break perfect symmetry

        self.target = np.zeros(6)
        self.target[list(_TRIPOD_B)] = np.pi
        self.amplitude = np.ones(6)

    def step(self, forward: float, turn: float, per_leg_gain: np.ndarray, stop: bool = False):
        """Advance the oscillators one timestep.

        forward: gait frequency multiplier (0 = frozen).
        turn:    -1 full left .. +1 full right. Implemented as a left/right
                 stride-amplitude asymmetry, which is how a fly actually turns.
        """
        freq = 0.0 if stop else self.base_freq * float(np.clip(forward, 0.0, 1.5))

        # Kuramoto coupling toward the tripod phase offsets:
        #   dphi_i/dt += K/N * sum_j sin( (phi_j - phi_i) - (target_j - target_i) )
        # The sign matters: negating this term makes the coupling REPULSIVE and
        # the six legs slowly fan out instead of locking into two tripods.
        dphase = self.phase[None, :] - self.phase[:, None]
        dtarget = self.target[None, :] - self.target[:, None]
        phase_err = np.sin(dphase - dtarget).sum(axis=1)

        self.phase = (self.phase + self.dt * (2 * np.pi * freq + COUPLING * phase_err / 6.0)) % (2 * np.pi)

        left = np.clip(1.0 - float(turn), 0.0, 2.0)
        right = np.clip(1.0 + float(turn), 0.0, 2.0)
        side = np.array([left, left, left, right, right, right])  # LEG_ORDER is L,L,L,R,R,R
        self.amplitude = np.clip(side * np.asarray(per_leg_gain, dtype=float), 0.0, 2.0)
        if stop:
            self.amplitude = np.zeros(6)
        return self.phase, self.amplitude

    def swing_stance(self) -> np.ndarray:
        """True where a leg is in swing (foot off the ground)."""
        return np.sin(self.phase) > 0.0

    def gait_regularity(self) -> float:
        """Kuramoto-style order parameter over the two tripods.

        1.0 = the two tripods are cleanly antiphase-locked; near 0 = the gait
        has fallen apart. Used as a `--compare-envs` metric.
        """
        a = np.exp(1j * self.phase[list(_TRIPOD_A)]).mean()
        b = np.exp(1j * self.phase[list(_TRIPOD_B)]).mean()
        return float(np.abs(a - b) / 2.0)


assert len(LEG_ORDER) == 6, "CPG assumes six legs in LEG_ORDER"
