"""The wings, the flight model, and the room.

The honest claim is narrow and worth pinning: the wings are REAL articulated
bodies beating at a real wingbeat frequency, and they generate none of the lift.
The forces come from a controller. Both halves of that are tested here, along
with the things that actually went wrong while building it - a frame mismatch
that flipped the fly onto its back, and a moment arm that spun it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flygym")
pytest.importorskip("torch")

from body import flight  # noqa: E402
from body.world import ROOM_HALF, make_world  # noqa: E402

SMALL = dict(connectome="synthetic", subset="motor", synthetic_size=3000,
             connectome_dir="/tmp/flysim-test")


# ---------- the wingbeat, with no MuJoCo involved ----------

def test_the_wingbeat_strokes_and_feathers():
    period = 1.0 / flight.WINGBEAT_HZ
    samples = [flight.wing_angles(period * f) for f in np.linspace(0, 1, 64, endpoint=False)]
    stroke = np.array([s for s, _ in samples])
    feather = np.array([f for _, f in samples])
    assert stroke.max() > 0.9 * flight.STROKE_RAD
    assert stroke.min() < -0.9 * flight.STROKE_RAD
    # The wing must turn OVER, not just wave: the feather angle has to reach
    # both extremes, or the down- and upstroke are the same stroke backwards.
    assert feather.max() > 0.8 * flight.FEATHER_RAD
    assert feather.min() < -0.8 * flight.FEATHER_RAD


def test_wings_can_be_stilled():
    assert flight.wing_angles(0.003, amplitude=0.0) == (0.0, 0.0)


# ---------- the controller ----------

def _level():
    return np.eye(3)


def test_hovering_holds_the_fly_up():
    force, torque = flight.body_wrench(
        rotation=_level(), velocity=np.zeros(3), angular_velocity=np.zeros(3),
        weight=10.0, climb_mm_s=0.0, forward_mm_s=0.0, yaw_rate=0.0,
    )
    assert force[2] == pytest.approx(10.0, rel=1e-6)
    assert np.allclose(force[:2], 0.0)
    assert np.allclose(torque, 0.0)


def test_a_tilted_fly_asks_for_more_lift_but_not_unboundedly():
    """Lift is divided by the vertical component of the body's up axis.

    Without a clamp, a fly knocked past 70 deg would command unbounded lift
    trying to hold its height.
    """
    tipped = np.eye(3)
    tipped[:, 2] = [0.0, 0.0, 0.02]        # almost on its side
    force, _ = flight.body_wrench(
        rotation=tipped, velocity=np.zeros(3), angular_velocity=np.zeros(3),
        weight=10.0, climb_mm_s=0.0, forward_mm_s=0.0, yaw_rate=0.0,
    )
    assert np.linalg.norm(force) < 10.0 / flight.MIN_UP_COMPONENT + 1e-6


def test_manoeuvre_force_is_capped():
    """A 250 mm/s velocity error asks for 7.5 body weights uncapped, which is
    not a manoeuvre any animal makes and drove the fly into the floor."""
    weight = 10.0
    force, _ = flight.body_wrench(
        rotation=_level(), velocity=np.zeros(3), angular_velocity=np.zeros(3),
        weight=weight, climb_mm_s=0.0, forward_mm_s=250.0, yaw_rate=0.0,
    )
    assert np.linalg.norm(force[:2]) <= weight * flight.MAX_MANOEUVRE_WEIGHTS + 1e-9


def test_attitude_damping_does_not_fight_the_yaw_command():
    """Damping the whole angular velocity damps yaw too, and at these gains that
    swamps the turn command completely - a commanded 1.5 rad/s turn produced a
    path indistinguishable from flying straight."""
    spinning = np.array([0.0, 0.0, 1.5])          # already turning at the rate asked
    _, torque = flight.body_wrench(
        rotation=_level(), velocity=np.zeros(3), angular_velocity=spinning,
        weight=10.0, climb_mm_s=0.0, forward_mm_s=0.0, yaw_rate=1.5,
    )
    assert abs(torque[2]) < 1e-9, "a fly already turning at the commanded rate needs no torque"


# ---------- the room ----------

def test_only_the_room_offers_flight():
    assert make_world("flat").flight is False
    assert make_world("kitchen").flight is False
    room = make_world("room")
    assert room.flight is True
    assert room.flight_bounds == (ROOM_HALF[0], ROOM_HALF[1], 2 * ROOM_HALF[2])


def test_the_room_keeps_one_ground_geom():
    """Same trap as the kitchen: FlyGym drops the per-leg contact sensors when a
    world has more than one ground geom, and the walls are not ground."""
    assert len(make_world("room").ground_geoms) == 1


def test_the_ceiling_has_a_hole_in_it():
    """A closed room hides the digital rain, which is the point of the sky."""
    room = make_world("room")
    ceiling = [n for n in room.prop_geoms if n.startswith("ceil_")]
    assert len(ceiling) == 4, "the ceiling should be a frame, not a slab"


# ---------- wings only where they are needed ----------

@pytest.fixture(scope="module")
def room_body():
    from body.sim import FlyBody
    from env.loader import load_preset

    body = FlyBody(load_preset("temperate"), world="room")
    yield body
    body.close()


def test_wings_are_added_only_in_a_flyable_world(room_body):
    """Six extra DoFs of dynamics would change every walking number in the
    README, so a world with no air space above it does not get them."""
    from body.sim import FlyBody
    from env.loader import load_preset

    assert room_body.can_fly
    assert len(room_body._wing_slots) == 6
    assert room_body.n_actuators == 48

    ground = FlyBody(load_preset("temperate"), world="kitchen")
    try:
        assert not ground.can_fly
        assert ground._wing_slots == {}
        assert ground.n_actuators == 42
    finally:
        ground.close()


def test_a_hovering_fly_stays_where_it_is(room_body):
    """The integration test for the whole flight model.

    This is where a body-frame angular velocity fed to a world-frame torque
    showed up: level, the two frames coincide and it looks perfect, so the bug
    only appears once the fly tilts.
    """
    import mujoco

    from body.cpg import TripodCPG

    cpg = TripodCPG(dt_s=room_body.timestep, seed=0)
    for _ in range(2000):
        phase, amplitude = cpg.step(1.0, 0.0, np.ones(6))
        room_body.step(phase, amplitude)

    room_body.takeoff()
    room_body.set_flight_command(climb_mm_s=200.0, forward_mm_s=0.0, yaw_rate=0.0)
    while room_body._prev_pos[2] < 250.0:
        phase, amplitude = cpg.step(0.0, 0.0, np.ones(6), stop=True)
        room_body.step(phase, amplitude)

    room_body.set_flight_command(0.0, 0.0, 0.0)
    start = room_body._prev_pos.copy()
    worst_tilt = 0.0
    for _ in range(20000):
        phase, amplitude = cpg.step(0.0, 0.0, np.ones(6), stop=True)
        obs = room_body.step(phase, amplitude)
        up_z = room_body._body_frame()[2, 2]
        worst_tilt = max(worst_tilt, float(np.degrees(np.arccos(np.clip(up_z, -1, 1)))))

    drift = float(np.linalg.norm((obs["position"] - start)[:2]))
    assert worst_tilt < 20.0, f"hovering fly tilted {worst_tilt:.0f} deg"
    assert abs(obs["position"][2] - start[2]) < 25.0, "hovering fly did not hold its height"
    assert drift < 60.0, f"hovering fly drifted {drift:.0f} mm"
    assert not obs["fell_over"]
    room_body.land()


def test_landing_clears_the_applied_wrench(room_body):
    """`xfrc_applied` persists across `mj_step`, so a stale force would keep
    flying a fly that has already touched down."""
    from body.cpg import TripodCPG

    cpg = TripodCPG(dt_s=room_body.timestep, seed=0)
    room_body.takeoff()
    room_body.set_flight_command(100.0, 0.0, 0.0)
    for _ in range(500):
        room_body.step(*cpg.step(0.0, 0.0, np.ones(6), stop=True))
    assert np.any(room_body.sim.mj_data.xfrc_applied[room_body._thorax_body] != 0.0)

    room_body.land()
    for _ in range(50):
        room_body.step(*cpg.step(0.6, 0.0, np.ones(6)))
    assert np.all(room_body.sim.mj_data.xfrc_applied[room_body._thorax_body] == 0.0)
