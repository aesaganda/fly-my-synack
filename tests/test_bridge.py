"""Motor decoding: grouping comes from the connectome, gains do not."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from body.legs import LEG_ORDER  # noqa: E402
from brain.lif import LIFNetwork, LIFParams  # noqa: E402
from brain.sources import synthetic  # noqa: E402
from brain.subset import apply_subset  # noqa: E402
from bridge.decode import MotorDecoder  # noqa: E402
from bridge.encode import SensoryEncoder  # noqa: E402


@pytest.fixture(scope="module")
def net():
    tables = apply_subset(synthetic.build(n_neurons=4000, mean_degree=20, seed=2), "motor")
    return LIFNetwork(tables, LIFParams(noise_std=0.0), seed=0)


def test_every_leg_gets_a_motor_neuron_population(net):
    """Six populations from subclass fl/ml/hl x somaSide - all must be non-empty,
    or a leg is silently undriveable."""
    dec = MotorDecoder(net)
    assert not dec.empty_groups, f"legs with no motor neurons: {dec.empty_groups}"
    sizes = dec.group_sizes()
    for leg in LEG_ORDER:
        assert sizes[leg] > 0
    # fl+ml+hl = 135+116+130 = 381 leg MNs, split across both sides.
    assert sum(sizes[l] for l in LEG_ORDER) == 381
    assert sizes["DN_L"] + sizes["DN_R"] > 1000


def test_drive_is_bounded_even_when_the_network_saturates(net):
    """The ceiling is a runaway guard, so assert against the constant itself -
    a hardcoded number here goes stale the moment the ceiling is retuned."""
    from bridge.decode import MAX_FORWARD_DRIVE, PER_LEG_DEPTH

    dec = MotorDecoder(net)
    for _ in range(200):
        net.step(torch.full((net.n,), 80.0))
    d = dec.decode()
    assert 0.0 <= d.forward <= MAX_FORWARD_DRIVE
    assert -1.0 <= d.turn <= 1.0
    assert np.all(d.per_leg_gain >= 1.0 - PER_LEG_DEPTH)
    assert np.all(d.per_leg_gain <= 1.0 + PER_LEG_DEPTH)


def test_silent_network_gives_no_forward_drive(net):
    net.reset()
    dec = MotorDecoder(net)
    assert dec.decode().forward == pytest.approx(0.0, abs=1e-6)


def test_override_biases_drive_and_clears(net):
    dec = MotorDecoder(net)
    dec.set_override(turn=0.8, forward=0.5)
    d = dec.decode()
    assert d.turn == pytest.approx(0.8, abs=1e-6)
    assert d.forward >= 0.5

    dec.set_override(stop=1.0)
    assert dec.decode().stop is True

    dec.clear_override()
    assert dec.decode().stop is False


def test_sensory_gain_scales_injected_current(net):
    """The preset's sensory gains are what differ between environments."""
    dec = MotorDecoder(net)
    enc = SensoryEncoder(net, dec)
    forces = np.full(6, 5.0)

    low = enc.encode(forces, air_speed=0.0, mechanosensory_gain=1.0,
                     johnstons_organ_gain=0.0).abs().sum().item()
    high = enc.encode(forces, air_speed=0.0, mechanosensory_gain=1.6,
                      johnstons_organ_gain=0.0).abs().sum().item()
    assert high == pytest.approx(1.6 * low, rel=1e-5)

    none = enc.encode(np.zeros(6), air_speed=0.0, mechanosensory_gain=1.0,
                      johnstons_organ_gain=0.0).abs().sum().item()
    assert none == pytest.approx(0.0)


def test_air_speed_reaches_the_network(net):
    dec = MotorDecoder(net)
    enc = SensoryEncoder(net, dec)
    still = enc.encode(np.zeros(6), 0.0, 1.0, 1.0).abs().sum().item()
    windy = enc.encode(np.zeros(6), 500.0, 1.0, 1.5).abs().sum().item()
    assert windy > still
