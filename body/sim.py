"""FlyGym / MuJoCo wrapper.

Written against flygym 2.1.0's current API (`Simulation`, `compose.NeuroMechFly`,
state getters). The 1.x Gymnasium-style `env.step(action) -> obs` interface no
longer exists.

`body/` never imports torch or anything from `brain/`/`bridge/` - it takes CPG
phase/amplitude in and hands observations back out.
"""

from __future__ import annotations

import numpy as np

from body import flight
from body.legs import LEG_ORDER
# body/world.py imports flygym at module scope, and so does this module in all
# but name - it is the flygym wrapper. `body/legs.py` is the one that stays
# cheap and dependency-free, and it still is.
from body.world import DEFAULT_WORLD, make_world
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

# Physics timestep. FlyGym's default, and the value every documented number in
# the README was measured at. See `run.py --timestep` for what raising it buys
# and what it costs.
DEFAULT_TIMESTEP = 1e-4
# Physics steps per control update. 1 reproduces the original loop exactly.
DEFAULT_CONTROL_EVERY = 1


THORAX_SEGMENT = "c_thorax"
# The third antennal segment carries the olfactory receptor neurons, so this is
# where the odour is sampled. The two sit 0.28 mm apart, which is the entire
# spatial baseline the fly has for a left/right comparison - see
# ODOUR_BILATERAL_GAIN in bridge/encode.py.
ANTENNA_GEOMS = ("l_funiculus", "r_funiculus")

# Lateral offset, in mm, beyond which a contact counts as being on one side of
# the body rather than dead ahead.
TOUCH_HALF_WIDTH_MM = 0.6

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
        world: str = DEFAULT_WORLD,
        timestep: float = DEFAULT_TIMESTEP,
    ) -> None:
        # Imported here so `import body.legs` stays cheap and flygym-free.
        from flygym import Simulation
        from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
        from flygym.compose import ActuatorType, ContactParams, KinematicPosePreset, NeuroMechFly
        from flygym.utils.math import Rotation3D

        self._ActuatorType = ActuatorType
        neutral = KinematicPosePreset.NEUTRAL

        # The world comes first: it owns the camera framing, and a world with
        # scenery in it needs a wider, less top-down shot than the close gait
        # view FlyGym ships.
        self.world = make_world(world)
        self.world_name = world

        # Wings are only given joints in a world that can be flown in. They are
        # six extra DoFs of dynamics on a model that is otherwise rigid above
        # the coxae, so adding them unconditionally would change every walking
        # number in the README for the sake of scenery.
        self.can_fly = bool(getattr(self.world, "flight", False))
        self.fly = NeuroMechFly()
        joints = JointPreset.LEGS_ONLY.to_joint_list()
        if self.can_fly:
            biological = {
                (j.parent.name, j.child.name): j
                for j in JointPreset.ALL_BIOLOGICAL.to_joint_list()
            }
            joints = joints + [biological[("c_thorax", w)] for w in ("l_wing", "r_wing")]
        skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=joints)
        self.fly.add_joints(skeleton, neutral_pose=neutral)
        dofs = self.fly.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        if self.can_fly:
            dofs = dofs + [
                d for d in self.fly.get_jointdofs_order() if "wing" in d.child.name
            ]
        self.fly.add_actuators(
            dofs, actuator_type=ActuatorType.POSITION, kp=actuator_gain, neutral_input=neutral
        )
        # Far stronger than the default 1.0: the foot otherwise slips during
        # stance and a stride delivers a fraction of the travel its geometry
        # implies. Sweeping this was worth ~3x in speed. Beyond ~400 the foot
        # sticks hard enough to drag the fly off a straight line.
        self.fly.add_leg_adhesion(gain=adhesion_gain)
        self._camera = None
        if render:
            for cam_name, spec in (("trackcam", self.world.camera),
                                   ("flycam", getattr(self.world, "flight_camera", None))):
                if spec is None:
                    continue
                self._camera = self.fly.add_tracking_camera(
                    name=cam_name,
                    pos_offset=spec["pos_offset"],
                    rotation=Rotation3D(format="xyaxes", values=spec["xyaxes"]),
                    fovy=spec["fovy"],
                )

        # MuJoCo needs the contact reference time constant to be at least twice
        # the timestep or contacts go unstable, and FlyGym's default is tuned
        # for 1e-4. It has to follow the timestep up.
        self.world.add_fly(
            self.fly, [0.0, 0.0, 0.7], Rotation3D(format="quat", values=[1, 0, 0, 0]),
            ground_contact_params=ContactParams(
                solver_refaccl_timeconst=max(2e-4, 2.0 * timestep)
            ),
        )
        self.sim = Simulation(self.world, timestep=timestep)
        # Visual settings, like the physics options, must be written after
        # add_fly - it overwrites <visual> from the fly's mujoco_globals.yaml.
        self.world.apply_visuals(self.sim.mj_model)
        self.timestep = float(self.sim.timestep)
        self.name = self.fly.name

        # Neutral targets for the 42 actuated DoFs, in actuator order.
        lookup = self.fly.get_pose_lookup(neutral)
        order = self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        self.neutral_angles = np.array(
            [lookup.get(self._dof_key(d), 0.0) for d in order], dtype=float
        )
        self.n_actuators = len(self.neutral_angles)
        self._leg_slots = [
            i for i, d in enumerate(order)
            if any(k in d.child.name for k in ("coxa", "trochanterfemur", "tibia", "tarsus"))
        ]
        if len(self._leg_slots) != DOFS_PER_LEG * len(LEG_ORDER):
            raise RuntimeError(
                f"expected {DOFS_PER_LEG * len(LEG_ORDER)} actuated leg DoFs, "
                f"got {len(self._leg_slots)}"
            )
        self._wing_slots = {
            f"{d.child.name[0]}_{d.axis.value}": i
            for i, d in enumerate(order) if "wing" in d.child.name
        }

        segs = list(self.fly.get_bodysegs_order())
        self._thorax_idx = next(
            i for i, s in enumerate(segs) if THORAX_SEGMENT in str(s)
        )
        self._antenna_geoms = self._geom_ids(ANTENNA_GEOMS)
        self._prop_mask = self._prop_geom_mask()
        self._thorax_body = self.sim._internal_bodyids_by_fly[self.name][self._thorax_idx]
        self.has_odour = bool(self.world.odour_sources)

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
        self._flight_camera_id = -1

        self.airborne = False
        self._steps_since_obs = 0
        self._held_targets = self.neutral_angles.copy()
        self.flight_command = (0.0, 0.0, 0.0)   # climb mm/s, forward mm/s, yaw rate
        self._weight = float(self.sim.mj_model.body_mass.sum()) * abs(
            float(self.sim.mj_model.opt.gravity[2]) or 9810.0
        )
        if self.can_fly:
            self._prepare_wings()

        self.last_speed = 0.0
        self._prev_pos = None
        self.sim.warmup()
        self._prev_pos = self._thorax_position()

    @staticmethod
    def _dof_key(dof) -> str:
        """flygym's pose lookup is keyed 'parent-child-axis'."""
        return f"{dof.parent.name}-{dof.child.name}-{dof.axis.value}"

    def _geom_ids(self, names: tuple[str, ...]) -> list[int]:
        import mujoco

        ids = [
            mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.name}/{n}")
            for n in names
        ]
        if any(i < 0 for i in ids):
            raise RuntimeError(f"missing geom(s) among {names} on fly {self.name!r}")
        return ids

    def _prepare_wings(self) -> None:
        """Retune the wing joints and actuators so a 200 Hz beat can happen.

        As shipped the wing joints have damping 0.5 and stiffness 10, which is
        right for a folded wing and gives an actuator bandwidth of kp/damping =
        800 rad/s. At 200 Hz that attenuated the stroke to 0.18 mm; with these
        values it is 6.6 mm. Measured both ways.
        """
        import mujoco

        model = self.sim.mj_model
        for side in ("l", "r"):
            for axis in ("yaw", "pitch", "roll"):
                jid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.name}/c_thorax-{side}_wing-{axis}"
                )
                if jid >= 0:
                    model.dof_damping[model.jnt_dofadr[jid]] = flight.WING_DAMPING
                    model.jnt_stiffness[jid] = 0.0
        for slot in self._wing_slots.values():
            model.actuator_gainprm[slot, 0] = flight.WING_KP
            model.actuator_biasprm[slot, 1] = -flight.WING_KP
        # Direct indices into mj_data.ctrl for the four wing DoFs the wingbeat
        # drives. `set_actuator_inputs` rewrites all 48 every substep and looks
        # its own indices up each time; at 200 Hz this runs every physics step.
        ctrl_ids = self.sim._intern_actuatorids_by_type_by_fly[
            self._ActuatorType.POSITION][self.name]
        self._wing_ctrl = {
            key: (int(ctrl_ids[slot]), float(self.neutral_angles[slot]))
            for key, slot in self._wing_slots.items()
        }

    def _prop_geom_mask(self) -> np.ndarray | None:
        """Boolean over geom ids, True for the world's solid props.

        None where there are none, so `bumped_into` costs nothing in a world
        with nothing to bump into.
        """
        import mujoco

        names = getattr(self.world, "prop_geom_names", ())
        if not names:
            return None
        mask = np.zeros(self.sim.mj_model.ngeom, dtype=bool)
        for name in names:
            gid = mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                mask[gid] = True
        return mask

    def bumped_into(self) -> float:
        """0 if nothing is in the way, otherwise -1 (obstacle left) or +1 (right).

        Dead-ahead contacts return +1 rather than 0 so that a head-on hit still
        produces a turn instead of a stall.
        """
        if self._prop_mask is None:
            return 0.0
        data = self.sim.mj_data
        n = int(data.ncon)
        if n == 0:
            return 0.0
        hit = self._prop_mask[data.contact.geom1[:n]] | self._prop_mask[data.contact.geom2[:n]]
        if not hit.any():
            return 0.0
        lateral = (data.contact.pos[:n][hit] - self._thorax_position()) @ self._body_frame()[:, 1]
        mean = float(np.mean(lateral))
        if mean > TOUCH_HALF_WIDTH_MM:
            return -1.0
        return 1.0

    def _body_frame(self) -> np.ndarray:
        """3x3 body-to-world matrix; columns are forward, left and up."""
        import mujoco

        mat = np.zeros(9)
        mujoco.mju_quat2Mat(
            mat, np.asarray(self.sim.get_body_rotations(self.name)[self._thorax_idx], dtype=float)
        )
        return mat.reshape(3, 3)

    def set_flight_command(self, climb_mm_s: float, forward_mm_s: float, yaw_rate: float) -> None:
        self.flight_command = (float(climb_mm_s), float(forward_mm_s), float(yaw_rate))

    def takeoff(self) -> bool:
        """Leave the ground. No-op in a world with no air space above it."""
        if not self.can_fly:
            return False
        self.airborne = True
        return True

    def land(self) -> None:
        self.airborne = False

    def _apply_flight(self) -> None:
        """One step of the lumped flight model. See body/flight.py."""
        rotation = self._body_frame()
        # MuJoCo reports a free joint's angular velocity in the BODY frame while
        # `xfrc_applied` takes a torque in the WORLD frame. Mixing them is
        # invisible while the fly is level - the frames coincide - and turns the
        # attitude damping into positive feedback the moment it tilts, which
        # flipped the fly onto its back every time it was asked to fly forwards.
        force, torque = flight.body_wrench(
            rotation=rotation,
            velocity=np.asarray(self.sim.mj_data.qvel[:3], dtype=float),
            angular_velocity=rotation @ np.asarray(self.sim.mj_data.qvel[3:6], dtype=float),
            weight=self._weight,
            climb_mm_s=self.flight_command[0],
            forward_mm_s=self.flight_command[1],
            yaw_rate=self.flight_command[2],
        )
        # `xfrc_applied` acts at the THORAX body's centre of mass, but the fly's
        # mass is spread over legs, head, abdomen and wings and its true centre
        # of mass sits ~0.5 mm behind it. A horizontal force at the wrong point
        # is a pitching moment: measured, 1 body weight of pure +x force spun
        # the fly up to 9.3 rad/s of pitch in 0.3 s, and in flight that showed
        # up as a 13 rev/s spin the moment any thrust was commanded. Move the
        # force to the real centre of mass by adding the moment it would have
        # produced there.
        data = self.sim.mj_data
        com, ipos = data.subtree_com[self._thorax_body], data.xipos[self._thorax_body]
        ax, ay, az = com[0] - ipos[0], com[1] - ipos[1], com[2] - ipos[2]
        fx, fy, fz = force
        applied = data.xfrc_applied[self._thorax_body]
        applied[0], applied[1], applied[2] = fx, fy, fz
        applied[3] = torque[0] + ay * fz - az * fy
        applied[4] = torque[1] + az * fx - ax * fz
        applied[5] = torque[2] + ax * fy - ay * fx

    def antenna_positions(self) -> np.ndarray:
        """World (x, y, z) of the two third antennal segments, in mm.

        With JointPreset.LEGS_ONLY the head has no joint of its own, so the
        antennae are geoms welded to the thorax body - `get_body_positions`
        returns nothing useful for them (their body id is -1, which silently
        indexes the LAST body). Read the geom frames instead.
        """
        return np.asarray(self.sim.mj_data.geom_xpos[self._antenna_geoms], dtype=float)

    def odour_lr(self) -> np.ndarray:
        """(left, right) odour concentration at the antennae. Zeros with no sources."""
        if not self.has_odour:
            return np.zeros(2)
        return self.world.odour_at(self.antenna_positions()[:, :2])

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
        if self.world.sky is not None:
            # Fades the far ground into the skybox, so the worktop recedes into
            # the rain instead of ending at the far clipping plane.
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
        for i in range(self.sim.mj_model.ncam):
            name = mujoco.mj_id2name(self.sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i) or ""
            if name.endswith("trackcam"):
                self._camera_id = i
            elif name.endswith("flycam"):
                self._flight_camera_id = i

    def render_frame(self) -> np.ndarray | None:
        if not self._render_enabled:
            return None
        if self._renderer is None:
            self._init_renderer()
        if self.world.sky is not None:
            # Repaint the skybox for this frame. `_mjr_context` is private but
            # there is no public handle on the render context, and uploading a
            # texture is the only way to animate one without recompiling.
            self.world.sky.upload(
                self.sim.mj_model, self._renderer._mjr_context, self.sim.mj_data.time
            )
        camera = self._camera_id
        if self.airborne and self._flight_camera_id >= 0:
            camera = self._flight_camera_id
        self._renderer.update_scene(self.sim.mj_data, camera=camera)
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

    def _swim_offsets(self, phase: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
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
        return offsets * amplitude[:, None]

    def joint_targets(self, phase: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
        """Turn six leg phases into a full vector of joint-angle targets.

        The vector is 42 long without wings and 48 with them; leg offsets are
        written into the leg slots by index rather than by assuming the leg
        actuators come first.
        """
        offsets = self._swim_offsets(phase, amplitude) if self.swimming \
            else self._walk_offsets(phase, amplitude)
        targets = self.neutral_angles.copy()
        targets[self._leg_slots] += offsets.reshape(-1)
        if self.can_fly:
            stroke, feather = flight.wing_angles(
                self.sim.mj_data.time, amplitude=1.0 if self.airborne else 0.0
            )
            for side in ("l", "r"):
                targets[self._wing_slots[f"{side}_yaw"]] += stroke
                targets[self._wing_slots[f"{side}_pitch"]] += feather
        return targets

    def _walk_offsets(self, phase: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
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
        return offsets * amplitude[:, None]

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

    def substep(self) -> None:
        """Advance the physics one step with the last control command held.

        The physics timestep is pinned at 1e-4 because the gait needs it - at
        3e-4 the fly falls over and at 2e-4 its walking speed changes threefold.
        The CONTROLLER does not need that rate: a stride is 55 ms long, so
        updating joint targets every millisecond still samples it fifty times.
        Splitting the two is what makes real time reachable, because `mj_step`
        is only about 62 us of the 290 us a full control step used to cost.

        Only the genuinely time-varying parts are updated here: the wingbeat
        (200 Hz, so it would visibly stutter if held), the flight wrench, and
        gusts.
        """
        self._update_wind()
        self._steps_since_obs += 1
        if self.can_fly and self.airborne:
            stroke, feather = flight.wing_angles(self.sim.mj_data.time)
            ctrl = self.sim.mj_data.ctrl
            for side in ("l", "r"):
                index, neutral = self._wing_ctrl[f"{side}_yaw"]
                ctrl[index] = neutral + stroke
                index, neutral = self._wing_ctrl[f"{side}_pitch"]
                ctrl[index] = neutral + feather
            # The wrench genuinely does need the physics rate - held at the
            # control rate the fly tips over. It is written in scalars for
            # exactly that reason; see body/flight.py.
            self._apply_flight()
        self.sim.step()

    def step(self, phase: np.ndarray, amplitude: np.ndarray) -> dict:
        self._update_wind()
        self._held_targets = self.joint_targets(phase, amplitude)
        self.sim.set_actuator_inputs(
            self.name, self._ActuatorType.POSITION, self._held_targets
        )
        # Adhesion on during stance, off during swing - otherwise the fly drags
        # its feet and cannot lift them. Swimming has nothing to grip, so it is
        # off entirely; leaving it on lets the fly haul itself along the floor,
        # which is exactly the walking-underwater look this mode replaces.
        adhesion = (
            np.zeros(6) if (self.swimming or self.airborne)
            else self.stance_mask(phase).astype(float)
        )
        self.sim.set_leg_adhesion_states(self.name, adhesion)

        # The wrench has to be rewritten every step and cleared on landing:
        # xfrc_applied persists across mj_step, so a stale force would keep
        # flying a fly that has already touched down.
        if self.airborne:
            self._apply_flight()
        elif self.can_fly:
            self.sim.mj_data.xfrc_applied[self._thorax_body] = 0.0

        self.sim.step()

        contact_found, contact_forces, *_ = self.sim.get_ground_contact_info(self.name)
        # Per-leg mechanical load. Ground reaction alone is useless when
        # swimming - measured 0.01 per leg underwater against 85.9 on land -
        # because there is nothing to push on. Actuator effort is ~57 per leg in
        # BOTH media, since underwater the legs are working against fluid drag
        # instead. Real proprioceptors sense limb load, not specifically ground
        # contact, so summing the two gives a signal that survives both modes.
        # Leg slots only: with wings there are 48 position actuators, not 42.
        actuator_force = np.abs(
            np.asarray(self.sim.get_actuator_forces(self.name, self._ActuatorType.POSITION))
        )[self._leg_slots].reshape(len(LEG_ORDER), DOFS_PER_LEG).sum(axis=1)
        contact_magnitude = np.linalg.norm(np.asarray(contact_forces), axis=1)
        pos = self._thorax_position()
        # Divide by the time ACTUALLY elapsed since the last observation, not by
        # one timestep: with the controller running slower than the physics
        # there are several substeps in between, and dividing by one of them
        # reports a velocity several times too large - which then feeds the
        # air-speed the antennae sense.
        elapsed = self.timestep * (1 + self._steps_since_obs)
        velocity = (pos - self._prev_pos) / elapsed if self._prev_pos is not None else np.zeros(3)
        self._steps_since_obs = 0
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
            "odour_lr": self.odour_lr(),
            "joint_angles": np.asarray(self.sim.get_joint_angles(self.name)),
            "roll_deg": roll,
            "pitch_deg": pitch,
            # "Fell over" is a walking concept: it means the gait collapsed and
            # the animal is on its back or its belly. A swimming fly is
            # suspended in fluid with no feet down, and pitching or rolling is
            # ordinary behaviour there, not failure - so the check is not
            # applied in swim mode rather than firing spuriously.
            "airborne": self.airborne,
            "fell_over": (
                False if (self.swimming or self.airborne)
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
