"""Does the whole stack run? Kept fast and dependency-light on purpose.

Uses the synthetic connectome so it needs no credentials and no network.
"""

from __future__ import annotations

import pytest

pytest.importorskip("flygym")
pytest.importorskip("torch")

from session import Session  # noqa: E402

SMALL = dict(connectome="synthetic", subset="motor", synthetic_size=3000,
             connectome_dir="/tmp/flysim-test")


def test_100_steps_in_dry_land():
    sess = Session(preset="dry_land", **SMALL)
    summary = sess.run(100)
    assert sess.step_count == 100
    assert summary["steps"] == 100
    assert not summary["fell_over"]
    sess.body.close()


def test_preset_reaches_both_physics_and_neurons():
    """The whole point of a preset: it must move BOTH sides of the model."""
    sess = Session(preset="dry_land", **SMALL)
    dry_density = sess.body.sim.mj_model.opt.density
    dry_tau = sess.net.params.tau_m_ms

    sess.switch_preset("submerged_water")
    assert sess.body.sim.mj_model.opt.density > 100 * dry_density
    sess.switch_preset("cold")
    assert sess.net.params.tau_m_ms > dry_tau      # colder -> slower
    sess.switch_preset("hot")
    assert sess.net.params.tau_m_ms < dry_tau      # hotter -> faster
    sess.body.close()


def test_mid_run_preset_switch_keeps_running():
    """Environment-transition experiments switch presets without a rebuild."""
    sess = Session(preset="dry_land", **SMALL)
    sess.run(50)
    sess.switch_preset("submerged_water")
    sess.run(50)
    assert sess.step_count == 100
    assert sess.preset.name == "submerged_water"
    sess.body.close()


def test_reset_returns_to_zero():
    sess = Session(preset="dry_land", **SMALL)
    sess.run(60)
    sess.reset()
    assert sess.step_count == 0
    sess.run(10)
    assert sess.step_count == 10
    sess.body.close()


def test_live_metrics_are_serialisable():
    """The web UI JSON-encodes these every 0.4 s."""
    import json

    sess = Session(preset="windy", **SMALL)
    sess.run(20)
    json.dumps(sess.live_metrics())
    assert sess.live_metrics()["preset"] == "windy"
    sess.body.close()


def test_metrics_are_geometrically_possible():
    """Guards a bug this metric has actually had.

    Straightness is displacement / path length, so it cannot exceed 1. An
    earlier version decimated the path for the length but measured displacement
    over the full array, so the endpoints did not match and it reported 1.06.
    """
    sess = Session(preset="dry_land", **SMALL)
    sess.run(600)
    s = sess.metrics.summary()
    assert 0.0 <= s["path_straightness"] <= 1.0
    assert s["net_speed_mm_s"] >= 0.0
    assert s["net_speed_mm_s"] <= s["mean_speed_mm_s"] + 1e-6, (
        "net speed cannot exceed mean instantaneous speed"
    )
    sess.body.close()
