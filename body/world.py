"""The scenery the fly walks in, and where its odour comes from.

Two worlds:

  ``flat``     FlatGroundWorld, untouched. Every kinematics number in the README
               was measured here, so it stays the default for ``run.py``.
  ``kitchen``  a worktop vignette under a Matrix-style digital-rain sky, with
               odour sources the fly can actually find. This is what the web UI
               serves by default, and it is the only world that emits odour.

The two are deliberately separable: the kitchen adds scenery, collision pairs
and an odour field, and NOTHING else. Every environment preset (windy, hot,
cold, submerged, ...) still applies to either world, because a preset writes to
``mjOption`` and a world writes geometry - they do not overlap.

`body/` never imports torch or anything from `brain/`/`bridge/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from flygym.compose import ContactParams, FlatGroundWorld
from flygym.utils.mjcf import GEOM_TYPES

# --------------------------------------------------------------------------
# Camera framing
#
# The default FlyGym tracking camera sits 7.5 mm out and looks 37 deg DOWN, so
# the horizon is off the top of the frame and the sky is never visible. That is
# the right framing for watching a gait and the wrong one for a world with
# scenery in it, so the kitchen pulls back and tilts up until about the top
# THIRD of the frame is sky.
#
# `xyaxes` gives the camera's x and y axes; the optical axis is -(x cross y).
# (0, 0.203, 0.979) as "up" points the camera 11.7 deg below horizontal, so with
# fovy=58 the top of the frame sits 17 deg ABOVE the horizon.
# --------------------------------------------------------------------------
CAMERA_GAIT = {"pos_offset": (-0.5, -7.5, 5.0), "xyaxes": (1, 0, 0, 0, 0.6, 0.8),
               "fovy": 30.0}
CAMERA_SCENE = {"pos_offset": (-0.5, -9.5, 2.0), "xyaxes": (1, 0, 0, 0, 0.203, 0.979),
                "fovy": 58.0}
# A second framing, used while the fly is airborne. Looking 12 deg DOWN is right
# for a fly on a worktop and wrong for one at 460 mm under a ceiling: it points
# at the far wall and the skylight never appears. This one looks 14 deg UP.
CAMERA_FLIGHT = {"pos_offset": (-0.5, -11.0, -1.4), "xyaxes": (1, 0, 0, 0, -0.242, 0.970),
                 "fovy": 62.0}

# Clipping planes for the kitchen, in mm (the model's unit extent is 1.0, so
# these are used as-is). See KitchenWorld.apply_visuals.
FAR_CLIP_MM = 1200.0
NEAR_CLIP_MM = 0.05

# Worktop tile pitch. Small enough that a walking fly crosses one every couple
# of strides, which is what makes the ground read as a surface rather than fog.
TILE_MM = 6.25


@dataclass(frozen=True)
class OdourSource:
    """A smell the fly can steer by. Gaussian, because a still-air diffusion
    plume is roughly Gaussian and anything better needs advection the physics
    engine is not solving.

    `sigma_mm` decides whether the fly can find it at all, from both sides. Too
    narrow and it is a needle the fly stumbles into rather than navigates to -
    the animal walks ~10 mm/s and its bilateral baseline is 0.28 mm. Too WIDE
    and neighbouring plumes overlap into a smooth field with no gradient left in
    it: measured, fifteen sources at sigma 15-22 mm and ~50 mm spacing flattened
    the worktop so completely that the fly walked 150k steps in a straight line
    and never came within 12 mm of anything. At sigma 11 mm against the same
    spacing the plumes stay separate.
    """

    name: str
    x: float
    y: float
    strength: float = 1.0
    sigma_mm: float = 11.0


def source_arrays(sources: tuple[OdourSource, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack sources into (centres, amplitudes, -1/2sigma^2).

    Split out so a world can pack once instead of per physics step: with a
    dozen sources, rebuilding these lists 10,000 times a second of sim time
    costs more than the arithmetic does.
    """
    return (
        np.array([[s.x, s.y] for s in sources], dtype=float).reshape(-1, 2),
        np.array([s.strength for s in sources], dtype=float),
        -0.5 / np.array([s.sigma_mm for s in sources], dtype=float) ** 2,
    )


def gaussian_sum(packed, points_xy) -> np.ndarray:
    """Concentration at each (x, y), summed over sources. Arbitrary units."""
    centres, amplitude, k = packed
    pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
    if len(centres) == 0:
        return np.zeros(len(pts))
    d2 = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
    return (amplitude * np.exp(k * d2)).sum(axis=1)


def odour_at(sources: tuple[OdourSource, ...], points_xy) -> np.ndarray:
    """Unpacked convenience form. The kitchen uses the packed path."""
    return gaussian_sum(source_arrays(sources), points_xy)


# --------------------------------------------------------------------------
# The sky
# --------------------------------------------------------------------------
class MatrixRain:
    """An animated digital-rain skybox.

    MuJoCo compiles a skybox as six square faces stacked vertically in one
    texture, in the order right, left, up, down, front, back, with row 0 at the
    TOP of each face (verified by rendering a gradient). So a drop falling is a
    bright head at increasing row index with a fading trail above it.

    The texture is repainted in `mj_model.tex_data` and re-uploaded to the GL
    context each rendered frame; nothing about the model is recompiled. Phase
    comes from `mj_data.time`, not a wall clock, so a run renders identically
    twice.
    """

    FACES = 6

    def __init__(self, face: int = 96, seed: int = 0, name: str = "matrix_rain") -> None:
        self.face = face
        self.name = name
        rng = np.random.default_rng(seed)
        shape = (self.FACES, 1, face)                      # (face, row-broadcast, column)
        # Rows per second. A drop crosses a 96-row face in 1-3 s.
        self._speed = rng.uniform(35.0, 110.0, shape)
        self._tail = rng.uniform(6.0, 26.0, shape)
        # Cycle longer than the face so columns spend part of the time empty -
        # otherwise every column always has a drop in it and it reads as static
        # noise rather than rain.
        self._cycle = face * rng.uniform(1.15, 2.6, shape)
        self._offset = rng.uniform(0.0, 1.0, shape) * self._cycle
        # A fixed binary glyph field. Rain sliding past a static field of glyphs
        # is what gives the flicker; regenerating it per frame just looks noisy.
        self._glyph = (rng.random((self.FACES, face, face)) > 0.42).astype(np.float32)
        self._texid = None
        self._adr = None

    # ---------- construction ----------

    def install(self, spec) -> None:
        """Add the skybox texture to an MjSpec (replacing any existing one)."""
        import mujoco as mj

        for existing in list(spec.textures):
            if existing.type == mj.mjtTexture.mjTEXTURE_SKYBOX:
                spec.delete(existing)
        tex = spec.add_texture()
        tex.name = self.name
        tex.type = mj.mjtTexture.mjTEXTURE_SKYBOX
        tex.width = self.face
        tex.height = self.FACES * self.face
        tex.nchannel = 3
        tex.data = self.frame(0.0).tobytes()

    # ---------- animation ----------

    def frame(self, t: float) -> np.ndarray:
        """(6*face, face, 3) uint8 rain at simulation time `t` seconds."""
        f = self.face
        rows = np.arange(f, dtype=float)[None, :, None]
        head = (self._offset + self._speed * float(t)) % self._cycle
        # Distance from the drop head UP the trail. Negative means the row is
        # below the head, i.e. the drop has not reached it yet.
        d = head - rows
        trail = np.where(d >= 0.0, np.exp(-d / self._tail), 0.0) * self._glyph
        glow = ((d >= 0.0) & (d < 1.6) & (head < f)).astype(float)

        rgb = np.empty((self.FACES, f, f, 3), dtype=float)
        rgb[..., 0] = 0.10 * trail + 0.90 * glow
        rgb[..., 1] = 0.035 + 0.85 * trail + 0.90 * glow     # 0.035 = never pure black
        rgb[..., 2] = 0.18 * trail + 0.90 * glow
        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(-1, f, 3)

    def upload(self, mj_model, mjr_context, t: float) -> None:
        """Repaint the compiled texture and push it to the GPU."""
        import mujoco as mj

        if self._texid is None:
            self._texid = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_TEXTURE, self.name)
            if self._texid < 0:
                raise RuntimeError(f"skybox texture {self.name!r} is not in the model")
            self._adr = int(mj_model.tex_adr[self._texid])
        data = self.frame(t).reshape(-1)
        mj_model.tex_data[self._adr:self._adr + data.size] = data
        mj.mjr_uploadTexture(mj_model, mjr_context, self._texid)


# --------------------------------------------------------------------------
# Kitchen scenery
#
# Everything is in millimetres, and box/ellipsoid sizes are HALF-extents, so
# `(5, 5, 5)` is a 10 mm sugar cube - about three and a half body lengths of
# the 2.8 mm fly standing next to it.
#
# A kitchen at true relative scale is all wall: a mug is 90 mm of sheer
# ceramic, a worktop is a plain. So the scene is built the way a fly would
# actually meet one - crumbs, spills and sugar within a few body lengths, and
# the furniture as far-off silhouettes on the horizon. Nothing here is a
# measurement; it is set dressing, chosen so the fly has somewhere to go.
#
#   name, geom type, size (mm), position (mm), rgba, solid
#
# WHAT `solid` MEANS, AND WHY MOST PROPS ARE NOT
# ----------------------------------------------
# This gait cannot climb and cannot reverse. Measured: a wall only 0.4 mm high
# - half the fly's standing height - stops it dead, and running the CPG phase
# backwards does not walk it backwards, it walks it sideways (the duty-factor
# asymmetry means the cycle is not time-reversible). So any obstacle wide
# enough that the fly cannot slide off the end of it is a permanent trap. The
# rim of a 92 mm plate is exactly that: measured, the fly pressed against it for
# the remaining 40k steps of a run, stepping at a healthy drive and covering
# 0.6 mm per 4k steps instead of 3.5.
#
# So `solid` is reserved for props small enough that sliding along the face
# carries the fly off the end of it - verified: it clears a 3.6 mm cube after
# about 13k steps of contact and walks away. Everything bigger is scenery the
# fly passes through. At the camera's 9 mm standoff a 0.7 mm-thick plate reads
# as a plate whether or not it has collision.
#
# An escape reflex was tried instead and made it WORSE - see the README.
#
# `solid` only decides whether collision pairs are emitted against the fly.
# Every prop is contype=conaffinity=0, matching the fly's own geoms: in this
# model contact comes exclusively from explicit <pair> elements.
# --------------------------------------------------------------------------
_WOOD = (0.62, 0.44, 0.26, 1.0)
_CERAMIC = (0.93, 0.93, 0.90, 1.0)
_CRUMB = (0.78, 0.62, 0.35, 1.0)

KITCHEN_PROPS = (
    # -- within a few body lengths: what the fly actually walks among --------
    # Solid, and deliberately SMALL. See the note above _solid_props.
    ("sugar_cube",   "box",       (1.75, 1.75, 1.75), (22, 11, 1.75),    (0.97, 0.97, 0.99, 1.0), True),
    ("sugar_chip",   "box",       (1.1, 1.1, 1.1),    (13, 17, 1.1),     (0.97, 0.97, 0.99, 1.0), True),
    ("crumb_a",      "sphere",    (1.15, 1.15, 1.15), (-19, -13, 1.1),   _CRUMB, True),
    ("crumb_b",      "sphere",    (0.75, 0.75, 0.75), (-22, -9, 0.72),   _CRUMB, True),
    ("crumb_c",      "sphere",    (0.55, 0.55, 0.55), (-15, -16, 0.52),  _CRUMB, True),
    ("crumb_d",      "sphere",    (0.9, 0.9, 0.9),    (-25, -17, 0.86),  _CRUMB, True),
    ("rice_grain",   "ellipsoid", (1.8, 0.6, 0.6),    (7, -21, 0.6),     (0.95, 0.94, 0.88, 1.0), True),

    # -- spills: walked ON, not into, so they are flat and pass-through ------
    ("jam_drop",     "ellipsoid", (5.5, 4.5, 0.35),   (-8, 26, 0.3),     (0.45, 0.06, 0.10, 1.0), False),
    ("syrup_pool",   "ellipsoid", (7.0, 5.0, 0.3),    (48, -38, 0.28),   (0.55, 0.33, 0.06, 1.0), False),
    ("droplet",      "sphere",    (1.4, 1.4, 1.4),    (30, 24, 1.0),     (0.55, 0.75, 0.95, 0.55), False),
    ("coffee_ring",  "cylinder",  (9.0, 0.04, 0.0),   (34, -16, 0.04),   (0.35, 0.22, 0.13, 0.9), False),

    # -- crockery: big enough to trap the fly against its rim, so scenery ----
    ("chopping_board", "box",     (45, 30, 0.9),      (-58, 42, 0.9),    _WOOD, False),
    ("plate",        "cylinder",  (46, 1.1, 0.0),     (62, 58, 1.1),     _CERAMIC, False),
    ("saucer",       "cylinder",  (28, 0.8, 0.0),     (-95, -52, 0.8),   _CERAMIC, False),
    ("spoon_bowl",   "ellipsoid", (9, 5.5, 1.2),      (18, 74, 1.2),     (0.78, 0.80, 0.84, 1.0), False),
    ("spoon_handle", "box",       (22, 1.6, 0.5),     (48, 74, 0.5),     (0.78, 0.80, 0.84, 1.0), False),

    # -- horizon: silhouettes against the rain, ~500 mm out so they subtend
    #    4-12 deg. Any closer and a mug fills the whole sky band. -----------
    ("backsplash",   "box",       (600, 4.0, 22),     (0, 620, 22),      (0.40, 0.44, 0.41, 1.0), False),
    ("mug",          "cylinder",  (32, 42, 0.0),      (-215, 400, 42),   (0.20, 0.36, 0.55, 1.0), False),
    ("jar",          "cylinder",  (21, 30, 0.0),      (125, 490, 30),    (0.45, 0.62, 0.42, 0.85), False),
    ("jar_lid",      "cylinder",  (22, 3.0, 0.0),     (125, 490, 63),    (0.55, 0.45, 0.20, 1.0), False),
    ("bottle",       "cylinder",  (12, 52, 0.0),      (300, 455, 52),    (0.16, 0.22, 0.16, 1.0), False),
    ("bottle_neck",  "cylinder",  (5.5, 9.0, 0.0),    (300, 455, 113),   (0.16, 0.22, 0.16, 1.0), False),
    ("toaster",      "box",       (34, 20, 17),       (-55, 530, 17),    (0.72, 0.74, 0.78, 1.0), False),
    ("pan",          "cylinder",  (38, 9.0, 0.0),     (430, 360, 9),     (0.18, 0.18, 0.20, 1.0), False),

    # -- the rest of the mess, scattered out to ~150 mm ----------------------
    # A worktop is not tidy, and this is not decoration: the fly habituates to
    # a source in a second or two and walks off, so without something else to
    # find it simply leaves the scene in a straight line and never comes back.
    ("crumb_e",      "sphere",    (1.0, 1.0, 1.0),    (105, 35, 0.95),   _CRUMB, True),
    ("crumb_f",      "sphere",    (1.2, 1.2, 1.2),    (30, 120, 1.15),   _CRUMB, True),
    ("crumb_g",      "sphere",    (0.85, 0.85, 0.85), (-88, 108, 0.8),   _CRUMB, True),
    ("oil_smear",    "ellipsoid", (9.0, 6.0, 0.3),    (-125, 20, 0.28),  (0.72, 0.62, 0.28, 0.95), False),
    ("milk_spill",   "ellipsoid", (11.0, 8.0, 0.3),   (-60, -95, 0.28),  (0.92, 0.92, 0.90, 0.9), False),
    ("tea_stain",    "cylinder",  (12.0, 0.04, 0.0),  (118, -78, 0.04),  (0.42, 0.28, 0.15, 0.9), False),
)


# The rest of the worktop.
#
# The hand-placed vignette above covers about 150 mm. The fly nets ~9 mm/s, so
# it walks out of that in under half a minute and off a 1000 mm ground plane in
# under two - measured, a 200k-step run ended 186 mm from the spawn and still
# heading out. A browser demo runs for as long as the tab is open, so the mess
# is generated out to SCATTER_RADIUS_MM instead of being typed out.
#
# All of it is flat and pass-through: a scatter this dense would otherwise be a
# minefield of things the fly can wedge against, and the reason only small props
# are solid is written out above.
#
# The count is a RENDER budget, not an aesthetic one. MuJoCo's cost per frame
# tracks the number of geoms in the scene and not their size, and these are
# specks: 300 of them took the frame from 6.3 ms to 13.9 ms, which comes
# straight out of the simulation's share of the wall clock. 120 over a smaller
# radius keeps the same spacing - about 70 mm between things worth walking to -
# for a third of the geometry.
SCATTER_SEED = 7
SCATTER_RADIUS_MM = 420.0
SCATTER_INNER_MM = 45.0      # leave the hand-placed vignette alone
SCATTER_COUNT = 120
SCATTER_SMELLS_EVERY = 3     # one prop in three is worth walking to

_SCATTER_KINDS = (
    # geom type, half-size, rgba          - crumb dust, a spill, a dried stain
    ("ellipsoid", (2.2, 1.6, 0.22), (0.76, 0.60, 0.33, 1.0)),
    ("ellipsoid", (6.5, 4.8, 0.26), (0.58, 0.34, 0.12, 0.95)),
    ("cylinder", (8.0, 0.05, 0.0), (0.40, 0.26, 0.15, 0.85)),
    ("ellipsoid", (4.0, 3.2, 0.24), (0.86, 0.84, 0.78, 0.95)),
)


def _scatter() -> tuple[tuple, tuple]:
    """Props and odour sources for the rest of the worktop, seeded."""
    rng = np.random.default_rng(SCATTER_SEED)
    props, smells = [], []
    for i in range(SCATTER_COUNT):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        # sqrt keeps the scatter uniform per unit AREA rather than crowding the
        # middle, which is where the hand-placed props already are.
        radius = float(np.sqrt(rng.uniform(SCATTER_INNER_MM ** 2, SCATTER_RADIUS_MM ** 2)))
        x, y = round(radius * np.cos(angle), 1), round(radius * np.sin(angle), 1)
        kind, size, rgba = _SCATTER_KINDS[i % len(_SCATTER_KINDS)]
        props.append((f"bit_{i:03d}", kind, size, (x, y, size[2]), rgba, False))
        if i % SCATTER_SMELLS_EVERY == 0:
            smells.append(OdourSource(
                f"bit_{i:03d}", x, y,
                strength=round(float(rng.uniform(0.7, 1.0)), 2),
                sigma_mm=round(float(rng.uniform(11.0, 15.0)), 1),
            ))
    return tuple(props), tuple(smells)


_SCATTER_PROPS, _SCATTER_ODOURS = _scatter()
KITCHEN_PROPS = KITCHEN_PROPS + _SCATTER_PROPS

# Where the smells come from. Placed on real props, so following a gradient
# takes the fly to something it can see.
KITCHEN_ODOURS = (
    OdourSource("sugar", 22.0, 11.0, strength=1.0, sigma_mm=11.0),
    OdourSource("jam", -8.0, 26.0, strength=0.9, sigma_mm=11.0),
    OdourSource("crumbs", -19.0, -13.0, strength=0.75, sigma_mm=11.0),
    OdourSource("syrup", 48.0, -38.0, strength=1.0, sigma_mm=11.0),
    OdourSource("coffee", 34.0, -16.0, strength=0.8, sigma_mm=11.0),
    OdourSource("butter", 62.0, 58.0, strength=0.9, sigma_mm=11.0),
    OdourSource("toast", -58.0, 42.0, strength=0.85, sigma_mm=11.0),
    OdourSource("spoon", 18.0, 74.0, strength=0.8, sigma_mm=11.0),
    OdourSource("grease", -95.0, -52.0, strength=0.9, sigma_mm=11.0),
    OdourSource("biscuit", 105.0, 35.0, strength=0.9, sigma_mm=11.0),
    OdourSource("cake", 30.0, 120.0, strength=0.95, sigma_mm=11.0),
    OdourSource("seed", -88.0, 108.0, strength=0.8, sigma_mm=11.0),
    OdourSource("oil", -125.0, 20.0, strength=0.9, sigma_mm=11.0),
    OdourSource("milk", -60.0, -95.0, strength=0.85, sigma_mm=11.0),
    OdourSource("tea", 118.0, -78.0, strength=0.8, sigma_mm=11.0),
) + _SCATTER_ODOURS

# Fly geoms given collision pairs against the solid props: the six foot tips
# plus the two segments that hit a wall first. Pairing every geom would be
# ~1200 extra narrow-phase checks per step for no visible difference.
PROP_CONTACT_SEGMENTS = (
    "c_thorax", "c_head",
    "lf_tarsus5", "lm_tarsus5", "lh_tarsus5", "rf_tarsus5", "rm_tarsus5", "rh_tarsus5",
)
# Props big enough that only the BODY needs to notice them. A wall has to stop
# the fly; it does not have to be felt by six individual feet, and pairing all
# eight geoms against the room shell and its furniture is 168 extra
# narrow-phase checks every physics step for no visible difference. The feet
# keep their pairs against the small props they walk among.
BULK_CONTACT_SEGMENTS = ("c_thorax", "c_head")


# --------------------------------------------------------------------------
# Worlds
# --------------------------------------------------------------------------
class FlatWorld(FlatGroundWorld):
    """The bare checkerboard plane, unchanged. No scenery, no smells."""

    camera = CAMERA_GAIT
    sky = None
    flight = False          # is there air space worth giving the fly wings for?
    # (half-x, half-y, ceiling) of the flyable volume, mm. None where there is
    # no room to be inside of.
    flight_bounds: tuple[float, float, float] | None = None
    flight_camera: dict | None = None      # framing used while airborne
    odour_sources: tuple[OdourSource, ...] = ()
    prop_geom_names: tuple[str, ...] = ()

    def apply_visuals(self, mj_model) -> None:
        """FlyGym's defaults, left alone."""

    def odour_at(self, points_xy) -> np.ndarray:
        return np.zeros(len(np.atleast_2d(np.asarray(points_xy, dtype=float))))


class KitchenWorld(FlatGroundWorld):
    """A worktop under a Matrix sky, with things on it that smell."""

    camera = CAMERA_SCENE
    flight = False
    odour_sources = KITCHEN_ODOURS

    # Far larger than FlyGym's 1000 mm default: at ~9 mm/s net the fly reaches
    # the edge of a 1000 mm plane in under two minutes, and the browser demo
    # runs indefinitely. One plane geom, so the size costs nothing.
    def __init__(self, name: str = "kitchen_world", *, half_size: float = 5_000,
                 sky_seed: int = 0) -> None:
        super().__init__(name=name, half_size=half_size)
        self.sky = MatrixRain(seed=sky_seed)
        self.sky.install(self.mjcf_root)
        self._retile_worktop(half_size)
        self.prop_geoms = {}
        self.solid_props = []
        self.bulk_props = []
        self._add_props()
        self._packed_odour = source_arrays(self.odour_sources)

    def _retile_worktop(self, half_size: float) -> None:
        """Repaint the ground as worktop tiles rather than the grey checker.

        FlatGroundWorld's material is already attached to the ground geom, so
        this edits the existing texture/material in place instead of adding a
        second one the geom would not reference.
        """
        for tex in self.mjcf_root.textures:
            if tex.name == "checker":
                tex.rgb1 = (0.80, 0.78, 0.73)
                tex.rgb2 = (0.71, 0.69, 0.65)
        for mat in self.mjcf_root.materials:
            if mat.name == "grid":
                # texrepeat is per half-size, so it has to track the plane or
                # a bigger worktop silently gets metre-wide tiles.
                repeat = half_size / TILE_MM
                mat.texrepeat = (repeat, repeat)
                # No reflectance: a mirror worktop turns every prop into a
                # second, upside-down prop and the scene stops being readable.
                mat.reflectance = 0.0

    @property
    def prop_geom_names(self) -> tuple[str, ...]:
        """Names of the solid props, for the body's collision check."""
        return tuple(g.name for g in self.solid_props)

    def _add_props(self) -> None:
        for name, kind, size, pos, rgba, solid in KITCHEN_PROPS:
            geom = self.mjcf_root.worldbody.add_geom(
                type=GEOM_TYPES[kind], name=f"prop_{name}", size=size, pos=pos,
                rgba=rgba, contype=0, conaffinity=0,
            )
            self.prop_geoms[name] = geom
            if solid:
                self.solid_props.append(geom)

    @override
    def _attach_fly_mjcf(self, fly, spawn_position, spawn_rotation, *args, **kwargs):
        """Attach the fly, then let it bump into the furniture.

        The props are kept OUT of `self.ground_geoms` on purpose: FlyGym only
        installs the per-leg ground-contact sensors when there is exactly one
        ground geom, and `Simulation.get_ground_contact_info()` - which
        `body/sim.py` reads every step - raises without them.
        """
        dofs = super()._attach_fly_mjcf(fly, spawn_position, spawn_rotation, *args, **kwargs)
        params = kwargs.get("ground_contact_params") or ContactParams()
        bulk = {g.name for g in getattr(self, "bulk_props", [])}
        for seg, geoms in fly.bodyseg_to_mjcfgeom.items():
            if seg.name in BULK_CONTACT_SEGMENTS:
                props = self.solid_props
            elif seg.name in PROP_CONTACT_SEGMENTS:
                props = [g for g in self.solid_props if g.name not in bulk]
            else:
                continue
            for body_geom in geoms:
                for prop in props:
                    self.mjcf_root.add_pair(
                        geomname1=body_geom.name, geomname2=prop.name,
                        name=f"{body_geom.name}-{prop.name}-prop",
                        friction=params.get_friction_tuple(),
                        solref=params.get_solref_tuple(),
                        solimp=params.get_solimp_tuple(),
                        margin=params.margin,
                    )
        return dofs

    def apply_visuals(self, mj_model) -> None:
        """Runtime-only visual settings, applied AFTER `Simulation` is built.

        Like the physics options in `env/loader.apply_physics`, these live in
        mutable model memory and have to be written after `add_fly`, which
        overwrites `<visual>` from the fly's own mujoco_globals.yaml.

        The far clipping plane is the one that matters: FlyGym ships
        `zfar = 250` (x the unit model extent, so 250 mm), which silently
        deletes anything further away. The first version of this scene had a
        mug and a toaster on the horizon that simply never rendered.
        """
        mj_model.vis.map.zfar = FAR_CLIP_MM
        # znear has to come up with it or the depth buffer, spanning 0.5 um to
        # 1.2 m, z-fights across the whole worktop. 50 um is still far closer
        # than the 11 mm tracking camera ever gets.
        mj_model.vis.map.znear = NEAR_CLIP_MM
        # Haze fades distant geometry into the skybox, which both sells the
        # depth and hides the ground plane's own edge at 1000 mm.
        mj_model.vis.map.haze = 0.9
        # Dimmer and cooler than the default flat 0.5/0.6 white, so the green
        # sky is the brightest thing in the frame rather than the worktop.
        mj_model.vis.headlight.ambient[:] = (0.30, 0.34, 0.31)
        mj_model.vis.headlight.diffuse[:] = (0.52, 0.55, 0.52)
        mj_model.vis.headlight.specular[:] = (0.05, 0.06, 0.05)

    def odour_at(self, points_xy) -> np.ndarray:
        return gaussian_sum(self._packed_odour, points_xy)


# --------------------------------------------------------------------------
# The room
#
# Walls, furniture and a ceiling with a hole in it, wrapped around the kitchen
# worktop - so the crumbs, the spills and the smells are all still there, and
# now there is somewhere to fly.
#
# It is 1.8 x 1.4 x 1.0 m, which is a small room and a very small kitchen. That
# is deliberate: the flight controller cruises at 200 mm/s, so a real 4 m
# kitchen would take 20 s of simulated time to cross and the simulation runs at
# about a sixth of real time. This one crosses in about eight.
#
# The ceiling is a frame rather than a slab, leaving a skylight over the middle.
# A closed room would hide the digital rain completely, and the rain is the
# point; an open-topped room would let the fly leave. A hole in the ceiling
# keeps both.
# --------------------------------------------------------------------------
ROOM_HALF = (900.0, 700.0, 500.0)      # x, y, half-height in mm
WALL_THICKNESS = 20.0
# Most of the ceiling, not a porthole: from 460 mm below a small hole is a
# postage stamp and the digital rain may as well not be there.
SKYLIGHT_HALF = (620.0, 470.0)
# A window in the +y wall - the one the tracking camera faces. The skylight
# alone was not enough: a fly cruising at 460 mm in a room with 1000 mm walls is
# looking at the far wall, not at the ceiling, so the digital rain never
# appeared in a single frame. Through a window it is straight ahead.
WINDOW_HALF_X = 450.0
WINDOW_Z = (250.0, 850.0)


def _room_shell() -> tuple:
    hx, hy, hz = ROOM_HALF
    t = WALL_THICKNESS
    sx, sy = SKYLIGHT_HALF
    wall = (0.72, 0.74, 0.70, 1.0)
    ceil = (0.55, 0.57, 0.54, 1.0)
    wx, (wz0, wz1) = WINDOW_HALF_X, WINDOW_Z
    side = (hx + t - wx) / 2
    shell = [
        ("wall_xp", "box", (t, hy + t, hz), (hx, 0, hz), wall, True),
        ("wall_xn", "box", (t, hy + t, hz), (-hx, 0, hz), wall, True),
        ("wall_yn", "box", (hx + t, t, hz), (0, -hy, hz), wall, True),
        # +y wall, as four pieces around the window opening.
        ("wall_yp_sill", "box", (hx + t, t, wz0 / 2), (0, hy, wz0 / 2), wall, True),
        ("wall_yp_head", "box", (hx + t, t, (2 * hz - wz1) / 2), (0, hy, (2 * hz + wz1) / 2), wall, True),
        ("wall_yp_left", "box", (side, t, (wz1 - wz0) / 2), (-(wx + side), hy, (wz0 + wz1) / 2), wall, True),
        ("wall_yp_right", "box", (side, t, (wz1 - wz0) / 2), (wx + side, hy, (wz0 + wz1) / 2), wall, True),
    ]
    # Ceiling as four slabs around a rectangular hole.
    z = 2 * hz
    shell += [
        ("ceil_xp", "box", ((hx - sx) / 2, hy, t), ((hx + sx) / 2, 0, z), ceil, True),
        ("ceil_xn", "box", ((hx - sx) / 2, hy, t), (-(hx + sx) / 2, 0, z), ceil, True),
        ("ceil_yp", "box", (sx, (hy - sy) / 2, t), (0, (hy + sy) / 2, z), ceil, True),
        ("ceil_yn", "box", (sx, (hy - sy) / 2, t), (0, -(hy + sy) / 2, z), ceil, True),
    ]
    return tuple(shell)


# Furniture, in the half of the room the fly does not start in, so it has
# somewhere to fly TO. Sizes are ordinary kitchen sizes; at 2.8 mm of fly they
# are cliffs.
ROOM_FURNITURE = (
    ("counter",     "box",      (420, 150, 150),  (-450, 520, 150),  (0.80, 0.78, 0.72, 1.0), True),
    ("counter_top", "box",      (430, 160, 12),   (-450, 520, 312),  (0.30, 0.31, 0.33, 1.0), True),
    ("fridge",      "box",      (190, 175, 400),  (620, 480, 400),   (0.86, 0.87, 0.89, 1.0), True),
    ("fridge_door", "box",      (8, 165, 380),    (425, 480, 410),   (0.78, 0.79, 0.82, 1.0), False),
    ("table_top",   "box",      (300, 200, 12),   (400, -350, 380),  _WOOD, True),
    ("table_leg_a", "cylinder", (14, 190, 0.0),   (680, -180, 190),  _WOOD, True),
    ("table_leg_b", "cylinder", (14, 190, 0.0),   (120, -180, 190),  _WOOD, True),
    ("table_leg_c", "cylinder", (14, 190, 0.0),   (680, -520, 190),  _WOOD, True),
    ("table_leg_d", "cylinder", (14, 190, 0.0),   (120, -520, 190),  _WOOD, True),
    # The one thing every fly in every room ends up circling.
    ("lamp_flex",   "cylinder", (4, 130, 0.0),    (0, 0, 870),       (0.25, 0.25, 0.27, 1.0), False),
    ("lamp_shade",  "cylinder", (95, 60, 0.0),    (0, 0, 680),       (0.94, 0.90, 0.72, 1.0), True),
    ("bin",         "cylinder", (110, 170, 0.0),  (-700, -450, 170), (0.30, 0.33, 0.36, 1.0), True),
)


class RoomWorld(KitchenWorld):
    """The kitchen, indoors, with air space above it."""

    camera = CAMERA_SCENE
    flight_camera = CAMERA_FLIGHT
    flight = True
    flight_bounds = (ROOM_HALF[0], ROOM_HALF[1], 2 * ROOM_HALF[2])

    def __init__(self, name: str = "room_world", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def _add_props(self) -> None:
        super()._add_props()
        for name, kind, size, pos, rgba, solid in _room_shell() + ROOM_FURNITURE:
            geom = self.mjcf_root.worldbody.add_geom(
                type=GEOM_TYPES[kind], name=f"prop_{name}", size=size, pos=pos,
                rgba=rgba, contype=0, conaffinity=0,
            )
            self.prop_geoms[name] = geom
            if solid:
                self.solid_props.append(geom)
                self.bulk_props.append(geom)

    def apply_visuals(self, mj_model) -> None:
        super().apply_visuals(mj_model)
        # The room is 1.8 m across and the kitchen's 1.2 m far plane would cut
        # the far wall in half.
        mj_model.vis.map.zfar = 4000.0
        # Haze fades the far ground into the sky, which is wrong indoors - the
        # far wall should stay solid.
        mj_model.vis.map.haze = 0.0
        mj_model.vis.headlight.ambient[:] = (0.34, 0.36, 0.34)
        mj_model.vis.headlight.diffuse[:] = (0.50, 0.52, 0.50)


WORLDS = {"flat": FlatWorld, "kitchen": KitchenWorld, "room": RoomWorld}
DEFAULT_WORLD = "flat"


def make_world(kind: str = DEFAULT_WORLD, **kwargs):
    if kind not in WORLDS:
        raise ValueError(f"unknown world {kind!r}, expected one of {sorted(WORLDS)}")
    return WORLDS[kind](**kwargs)
