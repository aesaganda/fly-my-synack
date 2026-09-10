"""FlyGym / MuJoCo wrapper.

Written against flygym 2.1.0's current API (`Simulation`, `compose.NeuroMechFly`,
state getters). The 1.x Gymnasium-style `env.step(action) -> obs` interface no
longer exists.

`body/` never imports torch or anything from `brain/`/`bridge/` - it takes CPG
phase/amplitude in and hands observations back out.
"""

from __future__ import annotations

import numpy as np

from body.legs import LEG_ORDER
from env.loader import EnvPreset, apply_physics
from env.q10 import q10_factor

# 7 actuated DoFs per leg, in the order flygym reports them:
#   0 coxa yaw, 1 coxa pitch, 2 coxa roll,
#   3 femur pitch, 4 femur roll, 5 tibia pitch, 6 tarsus pitch
DOFS_PER_LEG = 7
COXA_PITCH, FEMUR_PITCH, TIBIA_PITCH, TARSUS_PITCH = 1, 3, 5, 6

# The step cycle, derived from the model's measured foot Jacobian rather than
# guessed. Displacement of the front-left foot relative to the thorax, per
# +0.5 rad on each joint (mm):
#
#     joint          fore/aft      height
#     coxa  pitch +   +0.116       -0.089     -> protracts the foot
#     femur pitch +   -0.195       -0.467     -> drives the foot DOWN
#     femur pitch -   -0.024       +0.312     -> lifts the foot
#     tibia pitch +   -0.287       -0.268     -> retracts the foot, strongly
#
# So fore-aft travel comes from coxa AND tibia together, and the femur is the
# lift. phase 0 = foot fully forward and planted; stance is sin(phase) >= 0,
# over which the coxa retracts and the tibia extends, carrying the foot
# backwards and the body forwards. During swing the femur flexes to lift the
# foot clear.
#
# Two earlier versions of this were wrong in instructive ways: clipping the
# extension to max(0, cos) left the swing leg sitting at its neutral STANDING
# pose, so it dragged along the ground and cancelled the stance legs' thrust;
# and defining stance as the other half-cycle made the leg extend into the
# ground with adhesion off, levering the fly onto its back.
# NeuroMechFly is built with mirror_left2right=True, so the left and right leg
# joint axes point in opposite directions and the same joint offset sweeps the
# two sides opposite ways. Without this sign the fly walks in a circle: yaw
# drifts about -345 deg over 30k steps (a ~3.5 mm radius). With it, the drift
# falls to roughly +26 deg over 20k steps and forward speed nearly doubles.
SIDE_SIGN = np.array([1.0 if leg.startswith("L") else -1.0 for leg in LEG_ORDER])

# The tibia carries the fore-aft sweep together with the coxa, but its effect
# REVERSES between leg pairs. Measured foot travel per +0.5 rad of tibia pitch:
#     front  dx = -0.275   (foot moves backward)
#     mid    dx = -0.094
#     hind   dx = +0.234   (foot moves FORWARD)
# so it needs a per-pair sign; a single tibia term propels the front legs while
# fighting the hind ones, which shows up as a ~30 deg nose-up pitch bias. With
# the sign right the tibia is worth having twice over: it lengthens the stride,
# and because tibia pitch also moves the foot vertically it partly cancels the
# height change the coxa sweep causes - so the fly goes FASTER and nods LESS
# than with the coxa alone (11.2 mm/s at 7.1 deg pitch sd, against 3.0 mm/s at
# 11.6 deg).
TIBIA_POS_SIGN = np.array([1.0 if leg[1] in "FM" else -1.0 for leg in LEG_ORDER])

COXA_SWING_RAD = 2.6     # fore-aft sweep at the thorax-coxa joint
TIBIA_SWING_RAD = 2.4    # tibia's share of the same sweep (per-pair sign)
FEMUR_LIFT_RAD = 1.2     # femur flexion lifting the foot during swing

# Fraction of the cycle a leg spends in stance. At 0.5 (a pure sinusoid) the two
# tripods hand over instantaneously and, because the stance legs are themselves
# moving, the fly ends up with fewer than three feet down 65% of the time - it
# bounces rather than walks, nodding +/-20 deg every stride. Above 0.5 the
# tripods overlap, which restores real support. Insects likewise use a duty
# factor above 0.5 at walking speeds.
DUTY_FACTOR = 0.65
# PLACEHOLDER amplitudes: chosen by a sweep against forward speed AND postural
# stability. This gait tops out around 6 mm/s in a straight line, against the
# 10-20 mm/s a real fly walks, and the coxa excursion is larger than a real
# fly's. It moves the model convincingly; it is not a reproduction of measured
# Drosophila kinematics.

# Swimming stroke. Nothing to push against, so thrust comes purely from drag
# asymmetry: a fast power stroke with the legs spread, then a slow recovery with
# them folded in. Verified in open water with the fly suspended clear of the
# floor - the asymmetric stroke gives ~0.7 mm/s while a symmetric, unfolded one
# gives 0.005, i.e. essentially nothing. That is the whole propulsive mechanism.
#
# It is slow, and honestly so: a millimetre-scale body rowing in water sits at a
# Reynolds number of order 1-10, where viscosity dominates and rowing is a poor
# way to travel. Real adult Drosophila are bad swimmers too. Do not expect
# walking speeds here.
SWIM_SWEEP_RAD = 2.5      # fore-aft sweep of the rowing stroke
SWIM_FOLD_RAD = 1.8       # femur/tibia flexion during recovery, to cut drag
SWIM_POWER_FRACTION = 0.35  # fraction of the cycle spent on the power stroke

# Muscle kinetics are temperature-dependent too, not just neural ones. An
# ectotherm in the cold has slower, weaker muscle as well as slower neurons, and
# leaving that out made the cold preset merely step less often rather than look
# sluggish. The position actuators stand in for muscle here, so their gain gets
# the same Q10 treatment as the membrane time constants - inverted, because
# warm muscle is FASTER while a warm membrane time constant is SHORTER.
#
# Clamped: below ~4 the fly cannot hold itself up and simply collapses, and
# above ~40 the extra stiffness buys nothing. Both the reuse of the neural Q10
# and these bounds are PLACEHOLDERS - muscle and membrane need not share a Q10.
MUSCLE_GAIN_CLAMP = (4.0, 40.0)

THORAX_SEGMENT = "c_thorax"
FALL_HEIGHT_MM = 0.25   # thorax centre below this = collapsed onto the ground
# A height check alone is blind to the failure that actually happens: the fly
# rolls onto its back while its thorax stays well above FALL_HEIGHT_MM. Check
# orientation too.
FALL_ROLL_DEG = 90.0
FALL_PITCH_DEG = 75.0


def mujoco_bias_affine() -> int:
    """mjBIAS_AFFINE, which is what a position actuator uses."""
    import mujoco

    return int(mujoco.mjtBias.mjBIAS_AFFINE)


class FlyBody:
    def __init__(
        self,
        preset: EnvPreset,
        render: bool = False,
        seed: int = 0,
        actuator_gain: float = 20.0,
        adhesion_gain: float = 200.0,
        camera_res: tuple[int, int] = (360, 480),
    ) -> None:
        # Imported here so `import body.legs` stays cheap and flygym-free.
        from flygym import Simulation
        from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
        from flygym.compose import (
            ActuatorType,
            FlatGroundWorld,
            KinematicPosePreset,
            NeuroMechFly,
        )
        from flygym.utils.math import Rotation3D

        self._ActuatorType = ActuatorType
        neutral = KinematicPosePreset.NEUTRAL

        self.fly = NeuroMechFly()
        skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
        self.fly.add_joints(skeleton, neutral_pose=neutral)
        dofs = self.fly.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        self.fly.add_actuators(
            dofs, actuator_type=ActuatorType.POSITION, kp=actuator_gain, neutral_input=neutral
        )
        # Far stronger than the default 1.0: the foot otherwise slips during
        # stance and a stride delivers a fraction of the travel its geometry
        # implies. Sweeping this was worth ~3x in speed. Beyond ~400 the foot
        # sticks hard enough to drag the fly off a straight line.
        self.fly.add_leg_adhesion(gain=adhesion_gain)
        self._camera = self.fly.add_tracking_camera() if render else None

        world = FlatGroundWorld()
        world.add_fly(self.fly, [0.0, 0.0, 0.7], Rotation3D(format="quat", values=[1, 0, 0, 0]))
        self.sim = Simulation(world, timestep=1e-4)
        self.timestep = float(self.sim.timestep)
        self.name = self.fly.name

        # Neutral targets for the 42 actuated DoFs, in actuator order.
        lookup = self.fly.get_pose_lookup(neutral)
        order = self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        self.neutral_angles = np.array(
            [lookup.get(self._dof_key(d), 0.0) for d in order], dtype=float
        )
        self.n_actuators = len(self.neutral_angles)
        if self.n_actuators != DOFS_PER_LEG * len(LEG_ORDER):
            raise RuntimeError(
                f"expected {DOFS_PER_LEG * len(LEG_ORDER)} actuated DoFs, got {self.n_actuators}"
            )

        segs = list(self.fly.get_bodysegs_order())
        self._thorax_idx = next(
            i for i, s in enumerate(segs) if THORAX_SEGMENT in str(s)
        )

        # Preset must be applied AFTER add_fly: add_fly overwrites <option> from
        # the fly's own mujoco_globals.yaml.
        self.swimming = False
        self._kp_ref = actuator_gain
        self._adhesion_ref = adhesion_gain
        self._adhesion_actuators = [
            i for i in range(self.sim.mj_model.nu)
            if self.sim.mj_model.actuator_biastype[i] != mujoco_bias_affine()
        ]
        self._position_actuators = [
            i for i in range(self.sim.mj_model.nu)
            if self.sim.mj_model.actuator_biastype[i] == mujoco_bias_affine()
        ]
        self.apply_preset(preset)

        # The renderer is created lazily on first use, NOT here: the GL context
        # must be created on the same thread that renders with it. The web
        # service steps the simulation on a worker thread, and building the
        # context on the main thread then rendering from the worker deadlocks.
        self._renderer = None
        self._render_enabled = render
        self._camera_res = camera_res
        self._camera_id = -1

        self.last_speed = 0.0
        self._prev_pos = None
        self.sim.warmup()
        self._prev_pos = self._thorax_position()

    @staticmethod
    def _dof_key(dof) -> str:
        """flygym's pose lookup is keyed 'parent-child-axis'."""
        return f"{dof.parent.name}-{dof.child.name}-{dof.axis.value}"

    # ---------- environment ----------

    def apply_preset(self, preset: EnvPreset) -> None:
        self.preset = preset
        apply_physics(self.sim.mj_model, preset)
        self._wind_mean = np.asarray([float(w) for w in preset.physics["wind"]])
        gust = preset.wind_gust
        self._gust_amp = None if gust is None else np.asarray(gust, dtype=float)
        self._gust_hz = preset.wind_gust_hz
        self._wind = self._wind_mean.copy()
        self.swimming = preset.locomotion == "swim"
        self._apply_adhesion(preset)
        self._apply_muscle_gain(preset)

    def _apply_adhesion(self, preset: EnvPreset) -> None:
        """Set tarsal grip from the preset (see EnvPreset.adhesion_gain)."""
        gain = preset.adhesion_gain
        self.adhesion_gain = self._adhesion_ref if gain is None else float(gain)
        idx = self._adhesion_actuators
        if idx:
            self.sim.mj_model.actuator_gainprm[idx, 0] = self.adhesion_gain

    def _apply_muscle_gain(self, preset: EnvPreset) -> None:
        """Scale the position actuators' gain with temperature (see above)."""
        factor = q10_factor(preset.temperature_c, float(preset.neural["q10"]))
        self.muscle_gain = float(np.clip(self._kp_ref / factor, *MUSCLE_GAIN_CLAMP))
        idx = self._position_actuators
        self.sim.mj_model.actuator_gainprm[idx, 0] = self.muscle_gain
        self.sim.mj_model.actuator_biasprm[idx, 1] = -self.muscle_gain

    # ---------- rendering ----------

    def _init_renderer(self) -> None:
        import mujoco

        # MuJoCo's own Renderer rather than flygym's video-buffering one: the web
        # UI needs one frame on demand, not an encoded clip.
        self._renderer = mujoco.Renderer(
            self.sim.mj_model, height=self._camera_res[0], width=self._camera_res[1]
        )
        for i in range(self.sim.mj_model.ncam):
            name = mujoco.mj_id2name(self.sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name and "trackcam" in name:
                self._camera_id = i
                break

    def render_frame(self) -> np.ndarray | None:
        if not self._render_enabled:
            return None
        if self._renderer is None:
            self._init_renderer()
        self._renderer.update_scene(self.sim.mj_data, camera=self._camera_id)
        return self._renderer.render()

    # ---------- loop ----------

    def _thorax_position(self) -> np.ndarray:
        return np.asarray(self.sim.get_body_positions(self.name)[self._thorax_idx], dtype=float)

    def orientation_deg(self) -> tuple[float, float]:
        """Thorax roll and pitch in degrees, from the body quaternion (wxyz)."""
        w, x, y, z = np.asarray(
            self.sim.get_body_rotations(self.name)[self._thorax_idx], dtype=float
        )
        roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
        return float(roll), float(pitch)

    @staticmethod
    def _cycle(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split the cycle into stance and swing at DUTY_FACTOR.

        Returns (sweep, lift, is_stance). `sweep` runs +1 -> -1 across stance
        (carrying the planted foot backwards, driving the body forwards) and
        back over the shorter swing; `lift` is a half-sine confined to swing.
        """
        u = (phase / (2 * np.pi)) % 1.0
        stance = u < DUTY_FACTOR
        frac = np.where(stance, u / DUTY_FACTOR, (u - DUTY_FACTOR) / (1.0 - DUTY_FACTOR))
        sweep = np.where(stance, np.cos(np.pi * frac), -np.cos(np.pi * frac))
        lift = np.where(stance, 0.0, np.sin(np.pi * frac))
        return sweep, lift, stance

    @classmethod
    def stance_mask(cls, phase: np.ndarray) -> np.ndarray:
        """True where the foot is planted and pushing."""
        return cls._cycle(phase)[2]

    def _swim_targets(self, phase: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
        """Rowing stroke: sweep the legs back fast, fold them, bring them forward."""
        u = (phase / (2 * np.pi)) % 1.0
        power = u < SWIM_POWER_FRACTION
        frac = np.where(
            power, u / SWIM_POWER_FRACTION,
            (u - SWIM_POWER_FRACTION) / (1.0 - SWIM_POWER_FRACTION),
        )
        stroke = np.where(power, np.cos(np.pi * frac), -np.cos(np.pi * frac))
        fold = np.where(power, 0.0, -SWIM_FOLD_RAD)

        offsets = np.zeros((len(phase), DOFS_PER_LEG))
        offsets[:, COXA_PITCH] = SWIM_SWEEP_RAD * stroke * SIDE_SIGN
        offsets[:, FEMUR_PITCH] = fold
        offsets[:, TIBIA_PITCH] = fold
        offsets *= amplitude[:, None]
        return self.neutral_angles + offsets.reshape(-1)

    def joint_targets(self, phase: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
        """Turn six leg phases into 42 joint-angle targets."""
        if self.swimming:
            return self._swim_targets(phase, amplitude)
        offsets = np.zeros((len(phase), DOFS_PER_LEG))
        sweep, lift, _ = self._cycle(phase)
        offsets[:, COXA_PITCH] = COXA_SWING_RAD * sweep * SIDE_SIGN
        offsets[:, TIBIA_PITCH] = (
            -TIBIA_SWING_RAD * sweep * SIDE_SIGN * TIBIA_POS_SIGN
            - 0.5 * FEMUR_LIFT_RAD * lift
        )
        # The femur lift is NOT mirrored: "up" is the same direction on both
        # sides, so flexing it needs the same sign left and right.
        offsets[:, FEMUR_PITCH] = -FEMUR_LIFT_RAD * lift
        offsets *= amplitude[:, None]
        return self.neutral_angles + offsets.reshape(-1)

    def _gust_wave(self) -> np.ndarray:
        """Two incommensurate sines per axis, so gusts do not look metronomic.

        Amplitude stays within [-1, 1] because the two terms are halved.
        """
        t = self.sim.mj_data.time
        f = self._gust_hz
        slow = np.sin(2 * np.pi * f * t + np.array([0.0, 1.9, 3.4]))
        fast = np.sin(2 * np.pi * f * 2.7 * t + np.array([1.1, 0.3, 2.2]))
        return 0.5 * slow + 0.5 * fast

    def _update_wind(self) -> None:
        if self._gust_amp is None:
            return
        self._wind = self._wind_mean + self._gust_amp * self._gust_wave()
        self.sim.mj_model.opt.wind[:] = self._wind

    def step(self, phase: np.ndarray, amplitude: np.ndarray) -> dict:
        self._update_wind()
        self.sim.set_actuator_inputs(
            self.name, self._ActuatorType.POSITION, self.joint_targets(phase, amplitude)
        )
        # Adhesion on during stance, off during swing - otherwise the fly drags
        # its feet and cannot lift them. Swimming has nothing to grip, so it is
        # off entirely; leaving it on lets the fly haul itself along the floor,
        # which is exactly the walking-underwater look this mode replaces.
        adhesion = (
            np.zeros(6) if self.swimming else self.stance_mask(phase).astype(float)
        )
        self.sim.set_leg_adhesion_states(self.name, adhesion)

        self.sim.step()

        contact_found, contact_forces, *_ = self.sim.get_ground_contact_info(self.name)
        # Per-leg mechanical load. Ground reaction alone is useless when
        # swimming - measured 0.01 per leg underwater against 85.9 on land -
        # because there is nothing to push on. Actuator effort is ~57 per leg in
        # BOTH media, since underwater the legs are working against fluid drag
        # instead. Real proprioceptors sense limb load, not specifically ground
        # contact, so summing the two gives a signal that survives both modes.
        actuator_force = np.abs(
            np.asarray(self.sim.get_actuator_forces(self.name, self._ActuatorType.POSITION))
        ).reshape(len(LEG_ORDER), DOFS_PER_LEG).sum(axis=1)
        contact_magnitude = np.linalg.norm(np.asarray(contact_forces), axis=1)
        pos = self._thorax_position()
        velocity = (pos - self._prev_pos) / self.timestep if self._prev_pos is not None else np.zeros(3)
        self._prev_pos = pos
        self.last_speed = float(np.linalg.norm(velocity[:2]))

        roll, pitch = self.orientation_deg()
        return {
            "position": pos,
            "speed": self.last_speed,
            "contact_forces": contact_magnitude,
            "leg_load": contact_magnitude + actuator_force,
            "contact_found": np.asarray(contact_found) > 0,  # float sensor, not a bool
            "air_speed": float(np.linalg.norm(velocity - self._wind)),
            "joint_angles": np.asarray(self.sim.get_joint_angles(self.name)),
            "roll_deg": roll,
            "pitch_deg": pitch,
            # "Fell over" is a walking concept: it means the gait collapsed and
            # the animal is on its back or its belly. A swimming fly is
            # suspended in fluid with no feet down, and pitching or rolling is
            # ordinary behaviour there, not failure - so the check is not
            # applied in swim mode rather than firing spuriously.
            "fell_over": (
                False if self.swimming
                else bool(
                    pos[2] < FALL_HEIGHT_MM
                    or abs(roll) > FALL_ROLL_DEG
                    or abs(pitch) > FALL_PITCH_DEG
                )
            ),
        }

    def reset(self) -> None:
        self.sim.reset()
        self.apply_preset(self.preset)
        self.sim.warmup()
        self._prev_pos = self._thorax_position()
        self.last_speed = 0.0

    def close(self) -> None:
        try:
            self.sim.close()
        except Exception:
            pass
