"""Running the controller slower than the physics.

The physics timestep is pinned at 1e-4 because the gait needs it. The controller
is not, and decoupling the two is what makes real time reachable. Two things
have to hold: `control_every=1` must reproduce the original loop exactly, and a
decimated rate must not quietly break the closed loop - which is exactly what it
did, in a way that looked like a physics problem.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flygym")
pytest.importorskip("torch")

SMALL = dict(connectome="synthetic", subset="motor", synthetic_size=3000,
             connectome_dir="/tmp/flysim-test")


def _run(control_every: int, steps: int = 12_000):
    from session import Session

    sess = Session(preset="dry_land", world="kitchen", seed=0,
                   control_every=control_every, **SMALL)
    try:
        sess.run(steps)
        return sess, sess.metrics.summary()
    finally:
        pass


def test_control_every_one_is_the_original_loop():
    sess, summary = _run(1, steps=2000)
    try:
        assert sess.control_every == 1
        assert sess.brain_every == 2          # 0.2 ms of LIF against a 0.1 ms physics step
        assert summary["steps"] == 2000
        assert len(sess.metrics.path) == 20   # a sample every 100 physics steps
    finally:
        sess.body.close()


def test_the_sensory_loop_stays_closed_when_the_controller_is_decimated():
    """The bug this is here for.

    Sensory input is encoded on the control step just before the network reads
    it. Written as `(step_count + 1) % brain_every == 0`, that condition can
    never fire once control steps arrive N apart - so the network saw only its
    tonic drive, the decoded drive fell from 0.90 to 0.30, and the fly shuffled
    on the spot. It looked like a physics problem.
    """
    for control_every in (1, 10):
        sess, _ = _run(control_every, steps=8000)
        try:
            assert np.any(sess._external.cpu().numpy() != 0.0), (
                f"nothing reached the network at control_every={control_every}"
            )
            assert sess._last_drive.forward > 0.6, (
                f"decoded drive collapsed to {sess._last_drive.forward:.2f} "
                f"at control_every={control_every}"
            )
        finally:
            sess.body.close()


@pytest.mark.parametrize("control_every", (10, 20))
def test_the_gait_survives_a_decimated_control_rate(control_every):
    """A stride is 55 ms, so 2 ms control updates still sample it 27 times."""
    sess, summary = _run(control_every, steps=30_000)
    try:
        assert not summary["fell_over"]
        assert summary["net_speed_mm_s"] > 4.0, summary["net_speed_mm_s"]
        assert summary["slip_ratio"] < 0.75, summary["slip_ratio"]
        assert summary["gait_regularity"] > 0.9
        assert len(sess.metrics.path) > 10, "metrics were never sampled"
    finally:
        sess.body.close()


def test_the_cpg_keeps_its_stride_frequency():
    """The oscillator is advanced once per control step, so its dt is the
    control interval - not the physics timestep, or the stride silently divides
    by control_every."""
    from session import Session

    fast = Session(preset="dry_land", world="kitchen", seed=0, control_every=1, **SMALL)
    slow = Session(preset="dry_land", world="kitchen", seed=0, control_every=10, **SMALL)
    try:
        assert slow.cpg.dt == pytest.approx(10 * fast.cpg.dt)
        assert slow.cpg.base_freq == fast.cpg.base_freq
    finally:
        fast.body.close()
        slow.body.close()


def test_raising_the_timestep_is_available_and_documented_as_a_footgun():
    """`--timestep` is real, and the README says what it costs. Here we only
    check the plumbing reaches MuJoCo and the brain clock follows it."""
    from session import Session

    sess = Session(preset="dry_land", world="flat", seed=0, timestep=2e-4, **SMALL)
    try:
        assert sess.body.sim.mj_model.opt.timestep == pytest.approx(2e-4)
        # The LIF's own dt tracks the physics so the membrane time constants
        # stay right in real terms.
        assert sess.base_params.dt_ms == pytest.approx(sess.brain_every * 2e-4 * 1000)
        assert sess.metrics.timestep == pytest.approx(2e-4)
    finally:
        sess.body.close()
