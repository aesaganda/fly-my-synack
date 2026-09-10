"""The kitchen world, the Matrix sky, and the olfactory loop that explores it.

Three things are worth guarding here and they are all things that have gone
wrong or would go wrong silently:

  * the kitchen must not break the per-leg ground-contact sensors, which FlyGym
    only installs when the world has exactly ONE ground geom;
  * the sky has to actually change between frames, or "animated" is a lie;
  * the left/right odour sign has to steer the fly TOWARDS a smell. Getting it
    backwards produces a fly that flees food, which looks like plausible
    behaviour and is the opposite of what the code claims.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flygym")
pytest.importorskip("torch")

from body.world import (  # noqa: E402
    KITCHEN_ODOURS,
    MatrixRain,
    OdourSource,
    WORLDS,
    odour_at,
)

SMALL = dict(connectome="synthetic", subset="motor", synthetic_size=3000,
             connectome_dir="/tmp/flysim-test")


# ---------- the odour field (no MuJoCo needed) ----------

def test_odour_peaks_at_the_source_and_decays():
    src = (OdourSource("sugar", 10.0, 0.0, strength=1.0, sigma_mm=10.0),)
    at_source, one_sigma, far = odour_at(src, [[10, 0], [10, 10], [10, 60]])
    assert at_source == pytest.approx(1.0)
    assert one_sigma == pytest.approx(np.exp(-0.5), rel=1e-6)
    assert far < 1e-3


def test_odour_sources_sit_on_real_props():
    """A smell you cannot walk up to is a bug, not a feature."""
    from body.world import KITCHEN_PROPS

    props = {name: pos for name, _k, _s, pos, _c, _solid in KITCHEN_PROPS}
    for src in KITCHEN_ODOURS:
        matches = [n for n, p in props.items()
                   if abs(p[0] - src.x) < 1e-6 and abs(p[1] - src.y) < 1e-6]
        assert matches, f"odour source {src.name!r} at ({src.x}, {src.y}) has nothing there"


# ---------- the sky ----------

def test_rain_is_animated_and_reproducible():
    sky = MatrixRain(face=32, seed=3)
    a, b = sky.frame(0.0), sky.frame(0.35)
    assert a.shape == (6 * 32, 32, 3) and a.dtype == np.uint8
    assert not np.array_equal(a, b), "the sky is not moving"
    assert np.array_equal(a, MatrixRain(face=32, seed=3).frame(0.0)), "same seed, same sky"


def test_rain_is_green_and_never_pure_black():
    """It has to read as digital rain, not as a dark grey wall."""
    f = MatrixRain(face=48, seed=0).frame(1.0).astype(float)
    r, g, b = f[..., 0].mean(), f[..., 1].mean(), f[..., 2].mean()
    assert g > 2 * r and g > 2 * b, f"not green: rgb means {r:.1f}/{g:.1f}/{b:.1f}"
    assert f[..., 1].min() > 0, "green channel bottoms out at pure black"


def test_rain_installs_as_the_skybox_replacing_the_default():
    import mujoco as mj

    from body.world import make_world

    world = make_world("kitchen")
    skyboxes = [t for t in world.mjcf_root.textures
                if t.type == mj.mjtTexture.mjTEXTURE_SKYBOX]
    assert len(skyboxes) == 1, "left FlyGym's white gradient skybox in place"
    assert skyboxes[0].name == world.sky.name


# ---------- the world in MuJoCo ----------

@pytest.fixture(scope="module")
def kitchen_body():
    from body.sim import FlyBody
    from env.loader import load_preset

    body = FlyBody(load_preset("dry_land"), world="kitchen")
    yield body
    body.close()


def test_kitchen_keeps_the_ground_contact_sensors(kitchen_body):
    """Props must stay OUT of ground_geoms.

    FlyGym skips the per-leg contact sensors when a world has more than one
    ground geom, and `get_ground_contact_info` - which body/sim.py calls every
    step - then raises. Adding the furniture to ground_geoms would have been
    the obvious way to make it collide, and it would have broken the loop.
    """
    assert len(kitchen_body.world.ground_geoms) == 1
    found, forces, *_ = kitchen_body.sim.get_ground_contact_info(kitchen_body.name)
    assert np.asarray(found).shape == (6,)


def test_the_fly_can_bump_into_the_furniture(kitchen_body):
    """Scenery with no collision pairs is a hologram: nothing in this model has
    contype/conaffinity set, so contact comes only from explicit pairs."""
    import mujoco as mj

    model = kitchen_body.sim.mj_model
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_PAIR, i) for i in range(model.npair)]
    prop_pairs = [n for n in names if n and n.endswith("-prop")]
    assert prop_pairs, "no fly/prop collision pairs were emitted"
    assert any("tarsus5" in n for n in prop_pairs), "the feet cannot touch anything"


def test_antennae_are_two_real_places_that_move_with_the_fly(kitchen_body):
    """Bilateral olfaction needs two positions, not one.

    With LEGS_ONLY joints the antennae are geoms welded to the thorax, and
    `get_body_positions` reports body id -1 for them - which silently indexes
    the LAST body and gives both antennae the same wrong coordinate.
    """
    from body.cpg import TripodCPG

    left, right = kitchen_body.antenna_positions()
    separation = float(np.linalg.norm(left - right))
    assert 0.1 < separation < 1.0, f"antenna separation {separation:.3f} mm is not anatomical"

    cpg = TripodCPG(dt_s=kitchen_body.timestep, seed=0)
    before = kitchen_body.antenna_positions().copy()
    for _ in range(2000):
        phase, amplitude = cpg.step(1.0, 0.0, np.ones(6))
        kitchen_body.step(phase, amplitude)
    after = kitchen_body.antenna_positions()
    assert np.linalg.norm(after - before) > 0.05, "antennae did not move with the fly"
    assert float(np.linalg.norm(after[0] - after[1])) == pytest.approx(separation, rel=0.2)


def test_flat_world_smells_of_nothing():
    """The default world must be exactly what it was before olfaction existed."""
    from body.sim import FlyBody
    from env.loader import load_preset

    body = FlyBody(load_preset("dry_land"), world="flat")
    try:
        assert body.world.odour_sources == ()
        assert not body.has_odour
        assert np.all(body.odour_lr() == 0.0)
        obs = body.step(np.zeros(6), np.zeros(6))
        assert np.all(obs["odour_lr"] == 0.0)
    finally:
        body.close()


def test_flat_world_gets_none_of_the_new_machinery():
    """`--world flat` has to behave exactly as it did before worlds existed.

    Verified once by hand, and it is worth saying what "exactly" means: a
    12,000-step Session in dry_land, windy and cold produced bit-identical
    metrics before and after this feature landed. The conditions that make that
    true is the one asserted here: no odour sources, so `encode` is called with
    `odour_lr=None` and the injection is skipped entirely.
    """
    from session import Session

    sess = Session(preset="dry_land", world="flat", seed=0, **SMALL)
    try:
        assert sess.body.world.odour_sources == ()
        sess.run(50)
        assert np.all(sess.encoder.last_odour == 0.0)
    finally:
        sess.body.close()


def test_every_world_is_reachable_from_the_cli():
    import run

    for name in WORLDS:
        assert run.build_parser().parse_args(["--world", name]).world == name
    assert set(WORLDS) == {"flat", "kitchen", "room"}


# ---------- the olfactory loop ----------

def _settle(sess, odour_lr, steps=3000):
    """Drive the network with a fixed smell, everything else held constant."""
    sess.net.reset()
    sess.encoder.reset()
    for _ in range(steps):
        sess.net.step(sess.encoder.encode(
            leg_load=np.full(6, 140.0), air_speed=0.0,
            mechanosensory_gain=1.0, johnstons_organ_gain=1.0, odour_lr=odour_lr,
        ))
    return sess.decoder.decode().turn


@pytest.fixture(scope="module")
def kitchen_session():
    from session import Session

    sess = Session(preset="dry_land", world="kitchen", seed=0, **SMALL)
    yield sess
    sess.body.close()


def test_a_smell_on_the_right_produces_a_right_turn(kitchen_session):
    """The sign that decides whether the fly seeks food or flees it.

    `MotorDecoder` reads turn as tanh((DN_R - DN_L) / ref) and the CPG takes
    positive turn as a right turn - measured, not assumed: driving the body at
    turn=+0.4 leaves the yaw 35 deg CLOCKWISE of where turn=-0.4 leaves it.
    So more odour on the right has to raise DN_R.
    """
    base = 0.70
    delta = 0.01                      # the real bilateral difference near a plume
    right = _settle(kitchen_session, [base - delta, base + delta])
    left = _settle(kitchen_session, [base + delta, base - delta])
    assert right > left, f"turn {right:.3f} toward a right-side smell vs {left:.3f} left"
    assert right > 0.05 and left < -0.05, (
        f"olfactory steering is too weak to bend the path: {left:.3f} / {right:.3f}"
    )


def test_receptors_habituate_to_a_steady_smell(kitchen_session):
    """Without this the fly orbits the first source it finds for ever.

    At the peak of a plume the gradient reverses on every pass, so a fly with
    non-adapting receptors is pulled straight back in. Adaptation is what turns
    chemotaxis into exploration.
    """
    from bridge.encode import ODOUR_ADAPT_TAU_S

    enc = kitchen_session.encoder
    enc.reset()
    steps = int(3 * ODOUR_ADAPT_TAU_S / kitchen_session.body.timestep)
    first = None
    for _ in range(steps):
        left, right = enc._olfaction(np.array([1.0, 1.0]), 1.0)
        if first is None:
            first = abs(left)
    assert first > 0
    assert abs(left) < 0.05 * first, f"still responding after 3 tau: {left:.5f} vs {first:.5f}"
    # ...and it must recover: a NEW smell after habituation still gets through.
    fresh, _ = enc._olfaction(np.array([2.0, 2.0]), 1.0)
    assert abs(fresh) > 10 * abs(left)


def test_the_fly_ends_up_near_a_smell_only_when_it_can_smell(monkeypatch):
    """The whole point: closed-loop chemotaxis, not a scripted waypoint.

    Same world, same seed, same props - the ONLY difference is whether the
    bilateral olfactory gain is on. Measured over 40k steps the fly finishes
    ~10 mm from a source with it and ~25 mm away without.
    """
    import bridge.encode as encode
    from session import Session

    def run(gain):
        monkeypatch.setattr(encode, "ODOUR_BILATERAL_GAIN", gain)
        sess = Session(preset="dry_land", world="kitchen", seed=0, **SMALL)
        try:
            sess.run(40_000)
            x, y = sess.body._prev_pos[:2]
            return min(float(np.hypot(s.x - x, s.y - y))
                       for s in sess.body.world.odour_sources)
        finally:
            sess.body.close()

    with_smell = run(encode.ODOUR_BILATERAL_GAIN)
    without = run(0.0)
    assert with_smell < without - 5.0, (
        f"olfaction bought nothing: {with_smell:.1f} mm from a source with it, "
        f"{without:.1f} mm without"
    )


# ---------- the activity readout ----------

def test_neural_frame_is_a_decodable_raster(kitchen_session):
    import base64
    import json

    kitchen_session.reset()
    kitchen_session.run(4000)
    frame = kitchen_session.neural_frame()
    json.dumps(frame)                      # the control socket does this every 0.4 s

    raw = base64.b64decode(frame["raster"])
    assert len(raw) == frame["rows"] * frame["cols"]
    assert sum(g["rows"] for g in frame["groups"]) == frame["rows"]
    assert {g["name"] for g in frame["groups"]} >= {"DN_L", "DN_R", "LF", "RH"}
    assert np.frombuffer(raw, dtype=np.uint8).sum() > 0, "the raster is empty - nothing spiked"
    assert set(frame["rates_hz"]) == {g["name"] for g in frame["groups"]}


def test_reset_clears_the_raster(kitchen_session):
    import base64

    kitchen_session.run(2000)
    kitchen_session.reset()
    raw = base64.b64decode(kitchen_session.neural_frame()["raster"])
    assert np.frombuffer(raw, dtype=np.uint8).sum() == 0
