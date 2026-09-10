"""Connectome tables, subsetting, and the LIF engine."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from brain.lif import LIFNetwork, LIFParams  # noqa: E402
from brain.sources import synthetic  # noqa: E402
from brain.subset import apply_subset  # noqa: E402
from brain.tables import DESCENDING, MOTOR, NT_SIGN, sign_from_nt  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return synthetic.build(n_neurons=4000, mean_degree=20, seed=1)


def test_synthetic_matches_real_role_counts(tables):
    """Synthetic must mirror male-cns:v1.0's population sizes, or code that works
    on fake data silently misbehaves on real data."""
    assert len(tables.ids_by_superclass(DESCENDING)) == 1314
    assert len(tables.ids_by_superclass(MOTOR)) == 708 + 107


def test_glutamate_and_histamine_are_inhibitory_in_flies():
    """The vertebrate intuition is wrong here and would flip ~30% of the net."""
    assert NT_SIGN["glutamate"] == -1.0
    assert NT_SIGN["histamine"] == -1.0
    assert NT_SIGN["acetylcholine"] == +1.0

    import pandas as pd

    s = sign_from_nt(pd.Series(["acetylcholine", "GABA", "glutamate", "unclear", None]))
    assert list(s[:4]) == [1.0, -1.0, -1.0, 1.0]  # unknown falls back to excitatory


def test_subset_keeps_only_internal_edges(tables):
    motor = apply_subset(tables, "motor")
    ids = set(motor.neurons["bodyId"])
    assert len(motor.neurons) == 1314 + 815
    assert set(motor.edges["pre"]) <= ids and set(motor.edges["post"]) <= ids
    assert len(apply_subset(tables, "all").neurons) == len(tables.neurons)


def test_unknown_subset_rejected(tables):
    with pytest.raises(ValueError, match="unknown subset"):
        apply_subset(tables, "cerebellum")


def test_network_spikes_and_stays_finite(tables):
    net = LIFNetwork(apply_subset(tables, "motor"), seed=0)
    drive = torch.full((net.n,), 1.5)
    total = 0
    for _ in range(300):
        total += int(net.step(drive).sum())
    assert total > 0, "network never spiked under strong drive"
    assert torch.isfinite(net.v).all()
    assert (net.rate >= 0).all() and (net.rate <= 1).all()


def test_refractory_period_caps_firing_rate(tables):
    """No neuron may fire faster than 1/refractory, however hard it is driven."""
    net = LIFNetwork(apply_subset(tables, "motor"), LIFParams(noise_std=0.0), seed=0)
    steps = 500
    counts = torch.zeros(net.n)
    for _ in range(steps):
        counts += net.step(torch.full((net.n,), 50.0)).float()
    max_allowed = steps / net._refrac_steps + 1
    assert counts.max() <= max_allowed


def test_silent_without_drive(tables):
    net = LIFNetwork(apply_subset(tables, "motor"), LIFParams(noise_std=0.0), seed=0)
    assert sum(int(net.step().sum()) for _ in range(200)) == 0


def test_q10_taus_change_dynamics(tables):
    """Hot presets must actually make the network faster, not just parse."""
    sub = apply_subset(tables, "motor")
    from env.loader import load_preset

    def spikes_at(preset_name):
        p = load_preset(preset_name)
        base = LIFParams(noise_std=0.0)
        net = LIFNetwork(sub, base.scaled(p.scaled_taus(base.base_taus())), seed=0)
        return sum(int(net.step(torch.full((net.n,), 1.2)).sum()) for _ in range(400))

    assert spikes_at("hot") > spikes_at("cold")


def test_checkpoint_roundtrip_is_exact(tables, tmp_path):
    """--compare-envs depends on replaying the SAME brain across presets."""
    sub = apply_subset(tables, "motor")
    a = LIFNetwork(sub, seed=3)
    ckpt = tmp_path / "brain.pt"
    a.save(ckpt)

    b = LIFNetwork(sub, seed=3)
    b.load_weights(ckpt)
    assert torch.allclose(a.W.to_dense(), b.W.to_dense())

    out_a = [int(a.step(torch.full((a.n,), 1.5)).sum()) for _ in range(50)]
    out_b = [int(b.step(torch.full((b.n,), 1.5)).sum()) for _ in range(50)]
    assert out_a == out_b, "same seed + same weights must replay identically"


def test_checkpoint_rejects_mismatched_neuron_set(tables, tmp_path):
    ckpt = tmp_path / "small.pt"
    LIFNetwork(apply_subset(tables, "motor"), seed=0).save(ckpt)
    with pytest.raises(ValueError, match="different neuron set"):
        LIFNetwork(apply_subset(tables, "vnc"), seed=0).load_weights(ckpt)


def test_checkpoint_does_not_overwrite_preset_temperature(tables, tmp_path):
    """Loading a brain snapshot must not carry its Q10-scaled taus with it.

    --compare-envs saves one snapshot and replays it under every preset. When
    load_weights also restored LIFParams, every preset inherited the FIRST
    preset's temperature, so hot and cold produced identical behaviour - the
    comparison silently measured nothing.
    """
    from env.loader import load_preset

    sub = apply_subset(tables, "motor")
    base = LIFParams()

    cold_p = load_preset("cold")
    cold = LIFNetwork(sub, base.scaled(cold_p.scaled_taus(base.base_taus())), seed=0)
    ckpt = tmp_path / "cold.pt"
    cold.save(ckpt)

    hot_p = load_preset("hot")
    hot = LIFNetwork(sub, base.scaled(hot_p.scaled_taus(base.base_taus())), seed=0)
    hot_tau = hot.params.tau_m_ms
    assert hot_tau < cold.params.tau_m_ms

    hot.load_weights(ckpt)
    assert hot.params.tau_m_ms == hot_tau, "checkpoint overwrote the preset's temperature"
    assert torch.allclose(hot.W.to_dense(), cold.W.to_dense()), "weights should be shared"
