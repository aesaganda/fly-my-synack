"""The CPG is what actually generates the gait, so it gets its own checks."""

from __future__ import annotations

import numpy as np
import pytest

from body.cpg import TripodCPG
from body.legs import LEG_ORDER


def run(cpg, n=600, **kw):
    kw.setdefault("forward", 1.0)
    kw.setdefault("turn", 0.0)
    kw.setdefault("per_leg_gain", np.ones(6))
    for _ in range(n):
        out = cpg.step(**kw)
    return out


def test_tripods_settle_antiphase():
    cpg = TripodCPG(dt_s=1e-3)
    phase, _ = run(cpg)
    # LF (index 0) and RF (index 3) are in opposite tripods.
    sep = abs(np.angle(np.exp(1j * (phase[0] - phase[3]))))
    assert sep > 2.5, f"tripods not antiphase: separation {sep:.2f} rad"
    assert cpg.gait_regularity() > 0.8


def test_swing_stance_splits_into_two_tripods():
    cpg = TripodCPG(dt_s=1e-3)
    run(cpg)
    swing = cpg.swing_stance()
    assert 2 <= swing.sum() <= 4, "a tripod gait should have ~3 legs in swing"


def test_turn_produces_left_right_asymmetry():
    cpg = TripodCPG(dt_s=1e-3)
    _, amp = run(cpg, turn=0.6)
    left = amp[[LEG_ORDER.index(l) for l in ("LF", "LM", "LH")]].mean()
    right = amp[[LEG_ORDER.index(l) for l in ("RF", "RM", "RH")]].mean()
    assert right > left, "positive turn must lengthen right-side strides"

    _, amp_l = run(TripodCPG(dt_s=1e-3), turn=-0.6)
    assert amp_l[0] > amp_l[3], "negative turn must lengthen left-side strides"


def test_stop_and_zero_forward_freeze_the_gait():
    cpg = TripodCPG(dt_s=1e-3)
    _, amp = cpg.step(1.0, 0.0, np.ones(6), stop=True)
    assert not amp.any()

    # With forward=0 the coupling still tidies the legs into their tripods, so
    # individual phases may move; what must NOT happen is the cycle advancing.
    frozen = TripodCPG(dt_s=1e-3)
    run(frozen, n=800)                       # settle into the locked tripod
    before = np.angle(np.exp(1j * frozen.phase[[0, 4, 2]]).mean())
    for _ in range(2000):
        frozen.step(forward=0.0, turn=0.0, per_leg_gain=np.ones(6))
    after = np.angle(np.exp(1j * frozen.phase[[0, 4, 2]]).mean())
    advance = abs(np.angle(np.exp(1j * (after - before))))
    assert advance < 0.05, f"gait advanced {advance:.3f} rad with zero forward drive"


def test_forward_drive_sets_stride_frequency():
    slow, fast = TripodCPG(dt_s=1e-3), TripodCPG(dt_s=1e-3)
    for _ in range(300):
        slow.step(0.25, 0.0, np.ones(6))
        fast.step(1.0, 0.0, np.ones(6))
    assert fast.phase[0] != pytest.approx(slow.phase[0], abs=0.2)
