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

# 7 actuated DoFs per leg, in the order flygym reports them:
#   0 coxa yaw, 1 coxa pitch, 2 coxa roll,
#   3 femur pitch, 4 femur roll, 5 tibia pitch, 6 tarsus pitch
DOFS_PER_LEG = 7
COXA_PITCH, FEMUR_PITCH, TIBIA_PITCH, TARSUS_PITCH = 1, 3, 5, 6

# The step cycle.
#
# Fore-aft foot position follows sin(phase). The leg EXTENDS (femur/tibia/tarsus
# pitch up) over the half-cycle where cos(phase) > 0, and that extension is what
# presses the foot into the ground - so that half-cycle is STANCE, and adhesion
# must be on for it. Defining stance as the other half (cos <= 0) leaves the leg
# extending into the ground with no adhesion, which levers the body upward: the
# fly rears monotonically (pitch climbing past +50 deg) and eventually capsizes
# onto its back. That failure only appears at the low stride frequencies the
# brain actually commands, which is why a fast open-loop test does not catch it.
# Sign is negative because the coxa pitch axis is mirrored left-to-right
# (NeuroMechFly is built with mirror_left2right=True); with +sin the two sides
# push against each other and the fly rotates on the spot instead of walking.
# Verified empirically: +sin gives 0.2 mm of travel per second, -sin gives 4 mm.
# -0.6 rather than a larger stride: swept against travel AND stability, a
# bigger protraction pitches the body enough that the fly eventually capsizes
# once the brain starts modulating the legs asymmetrically. -0.6 travels
# further (6.7 mm/s vs 1.5) and keeps a margin against tipping.
PROTRACTION_RAD = -0.6   # coxa fore-aft amplitude
EXTEND_FEMUR_RAD = 0.6   # femur extension during stance
EXTEND_TIBIA_RAD = 0.8   # tibia extension during stance
EXTEND_TARSUS_RAD = 0.3
# PLACEHOLDER amplitudes: picked by a coarse sweep for net forward travel, not
# fitted to recorded Drosophila kinematics. Real flies walk ~10-20 mm/s; this
# gait manages ~4 mm/s, so it locomotes but is not a quantitative match.

THORAX_SEGMENT = "c_thorax"
FALL_HEIGHT_MM = 0.25   # thorax centre below this = collapsed onto the ground
# A height check alone is blind to the failure that actually happens: the fly
# rolls onto its back while its thorax stays well above FALL_HEIGHT_MM. Check
# orientation too.
FALL_ROLL_DEG = 90.0
FALL_PITCH_DEG = 75.0


class FlyBody:
    def __init__(
        self,
        preset: EnvPreset,
        render: bool = False,
        seed: int = 0,
        actuator_gain: float = 20.0,
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
        self.fly.add_leg_adhesion()
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
        self._wind = np.asarray([float(w) for w in preset.physics["wind"]])

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
    def stance_mask(phase: np.ndarray) -> np.ndarray:
        """True where the foot is planted and pushing (the extension half-cycle)."""
        return np.cos(phase) >= 0.0

    def joint_targets(self, phase: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
        """Turn six leg phases into 42 joint-angle targets."""
        offsets = np.zeros((len(phase), DOFS_PER_LEG))
        extend = np.maximum(0.0, np.cos(phase))  # nonzero only during stance
        offsets[:, COXA_PITCH] = PROTRACTION_RAD * np.sin(phase)
        offsets[:, FEMUR_PITCH] = EXTEND_FEMUR_RAD * extend
        offsets[:, TIBIA_PITCH] = EXTEND_TIBIA_RAD * extend
        offsets[:, TARSUS_PITCH] = EXTEND_TARSUS_RAD * extend
        offsets *= amplitude[:, None]
        return self.neutral_angles + offsets.reshape(-1)

    def step(self, phase: np.ndarray, amplitude: np.ndarray) -> dict:
        self.sim.set_actuator_inputs(
            self.name, self._ActuatorType.POSITION, self.joint_targets(phase, amplitude)
        )
        # Adhesion on during stance, off during swing - otherwise the fly drags
        # its feet and cannot lift them.
        self.sim.set_leg_adhesion_states(self.name, self.stance_mask(phase).astype(float))

        self.sim.step()

        contact_found, contact_forces, *_ = self.sim.get_ground_contact_info(self.name)
        pos = self._thorax_position()
        velocity = (pos - self._prev_pos) / self.timestep if self._prev_pos is not None else np.zeros(3)
        self._prev_pos = pos
        self.last_speed = float(np.linalg.norm(velocity[:2]))

        roll, pitch = self.orientation_deg()
        return {
            "position": pos,
            "speed": self.last_speed,
            "contact_forces": np.linalg.norm(np.asarray(contact_forces), axis=1),
            "contact_found": np.asarray(contact_found) > 0,  # float sensor, not a bool
            "air_speed": float(np.linalg.norm(velocity - self._wind)),
            "joint_angles": np.asarray(self.sim.get_joint_angles(self.name)),
            "roll_deg": roll,
            "pitch_deg": pitch,
            "fell_over": bool(
                pos[2] < FALL_HEIGHT_MM
                or abs(roll) > FALL_ROLL_DEG
                or abs(pitch) > FALL_PITCH_DEG
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
