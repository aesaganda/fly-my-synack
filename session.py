"""Wires brain -> bridge -> body -> bridge -> brain and runs the loop.

Shared by the CLI (`run.py`) and the web service, so both drive an identical
simulation. This is the only module that imports from all four packages; the
packages themselves stay independent.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from body.legs import LEG_ORDER
from body.sim import DEFAULT_CONTROL_EVERY, DEFAULT_TIMESTEP, FlyBody
from body.world import DEFAULT_WORLD
from brain.fetch import get_tables
from brain.lif import LIFNetwork, LIFParams
from brain.subset import DEFAULT_SUBSET
from bridge.decode import MotorDecoder
from bridge.encode import SensoryEncoder
from body.cpg import TripodCPG
from env.loader import EnvPreset, load_preset

# ---- live spike raster ---------------------------------------------------
#
# Purely a readout for the UI: the simulation does not depend on any of it.
# Rows are sampled EVENLY FROM THE POPULATIONS THE DECODER READS rather than at
# random from the whole network, so a viewer can see the two DN pools diverge
# when the fly turns instead of watching anonymous noise.
#
# Each cell is a spike COUNT over its bin, not an instantaneous sample. Sampling
# `spikes` once per bin would show almost nothing: a neuron at 100 Hz fires in
# about 2% of the 0.2 ms brain steps.
RASTER_ROWS_PER_GROUP = 9
RASTER_COLS = 160
# A column is a fixed slice of TIME, not a fixed number of brain steps. The
# brain's own dt follows the control rate, so counting steps turned 5 ms bins
# into 50 ms bins the moment the controller was decimated - and at 50 ms every
# cell saturates and the raster goes solid green.
RASTER_BIN_MS = 5.0

# ---- when to fly ---------------------------------------------------------
#
# Only in a world with air space in it (`--world room`). The rule reuses the
# olfactory machinery rather than adding a planner: receptor adaptation already
# decides when the fly is FINISHED with a smell, and leaving is what an animal
# does next. So the fly walks while it has a smell to work on, takes off when
# the smell has faded, cruises, and lands again when it flies into a new plume.
#
# This is a behavioural rule, not a connectome mechanism, and it is listed in
# the README's approximations table alongside the CPG.
ODOUR_INTERESTING = 0.25        # raw concentration worth walking towards
ODOUR_EXHAUSTED = 0.08          # ...and below which there is nothing here
GROUNDED_BEFORE_TAKEOFF_S = 1.5  # do not bounce straight back into the air
FLIGHT_CRUISE_MM_S = 260.0      # forward speed at full decoded drive
# Above the counter (312 mm) and the table (380 mm), below the lamp shade
# (620 mm). Cruising at counter height just means flying into the counter.
FLIGHT_CRUISE_HEIGHT_MM = 460.0
FLIGHT_CLIMB_GAIN = 1.2         # climb speed per mm of height error
FLIGHT_CLIMB_MAX_MM_S = 200.0
# Positive `turn` is a RIGHT turn for the walking gait, and a positive yaw-rate
# command turns the flying fly LEFT - verified from the flown path, not from the
# sign of the torque term. Hence the negation.
FLIGHT_YAW_RATE = 2.2
# Hitting things. Walking, an avoidance reflex measured WORSE than nothing (see
# the README) because a fly pressed against a face cannot turn - its yaw comes
# from stride asymmetry and it is not going anywhere. Flying, yaw comes from the
# attitude controller and works whatever the fly is touching, so here the reflex
# earns its place: without it the fly flew into the counter and hung there for
# the rest of the run.
FLIGHT_AVOID_YAW = 3.0
FLIGHT_AVOID_BACK_MM_S = -60.0
# The reflex has to LATCH. Released the instant contact breaks, it turns into a
# limit cycle: back off, resume cruise, fly straight back into the same face.
# Measured - the fly sat against the counter for 26 s of a 40 s run doing
# exactly that. 0.45 s of held yaw is enough to point it somewhere else.
FLIGHT_AVOID_HOLD_S = 0.45
# Staying off the walls.
#
# The avoidance reflex is a reflex: it fires on contact and it cannot see a
# corner coming. Measured, the fly worked its way into one and spent the last
# 25 s of a 50 s run climbing it. A real fly keeps off the walls with vision -
# optic flow and the looming response - and this model has eyes it does not use.
# So the flight controller is told where the room is and turns back when it gets
# close to the edge. It is a stand-in for the visual loop, in the same sense as
# the CPG standing in for gait generation, and it is why the fly stays in the
# middle of the room where the interesting things are.
# Landing has to LATCH as well. Cruising at 460 mm the fly passes over a plume
# in a fraction of a second, and a rule that only descends while it can smell
# something never gets down: measured, it crossed the food twice in 70 s at
# odour 0.27 and 0.56 and stayed at cruise height both times.
# ...and the latch has to be a TIMER, not a level. Cleared as soon as the smell
# is lost, it clears within a second of crossing the plume at 260 mm/s, long
# before the fly has descended 450 mm. Measured: it committed to a landing twice
# in 90 s and completed neither.
FLIGHT_APPROACH_HOLD_S = 8.0
# Landing cuts the lift dead, so it has to happen with the feet almost touching:
# releasing at 18 mm dropped the fly the last 18 mm in free fall, arriving at
# ~580 mm/s and tipping it over. The fly stands 0.85 mm tall.
FLIGHT_LAND_FROM_MM = 2.5
FLIGHT_APPROACH_MM_S = 90.0     # slow down on the way in
FLIGHT_APPROACH_HEIGHT_MM = 1.0
# Generous: the room is 1.8 x 1.4 m and the skylight is most of the ceiling, so
# keeping the fly in the middle third is both what stops it grinding along a
# wall and what puts the digital rain in the shot.
FLIGHT_WALL_MARGIN_MM = 470.0
FLIGHT_HOME_YAW_GAIN = 2.5
# Arousal while airborne, exactly the problem `submerged_water` documents: with
# no ground contact the only limb load is a little actuator effort, so the leg
# motor pools starve, the decoded forward drive decays to nothing and the fly
# hangs in the air going nowhere - measured, it froze to within 0.1 mm for the
# last 16 s of a 30 s run. Raising the background excitation breaks the loop the
# same way it does underwater. PLACEHOLDER value.
FLIGHT_TONIC_DRIVE = 1.5


@dataclass
class RunMetrics:
    """Locomotion kinematics - what --compare-envs diffs between presets."""

    steps: int = 0
    timestep: float = 1e-4
    path: list = field(default_factory=list)          # (x, y) body positions, mm
    speeds: list = field(default_factory=list)        # mm/s
    gait_regularity: list = field(default_factory=list)
    joint_traces: list = field(default_factory=list)  # sampled joint-angle vectors
    fell_over: bool = False

    def summary(self) -> dict:
        path = np.asarray(self.path) if self.path else np.zeros((1, 2))
        speeds = np.asarray(self.speeds) if self.speeds else np.zeros(1)
        traces = np.asarray(self.joint_traces) if self.joint_traces else np.zeros((1, 1))
        displacement = float(np.linalg.norm(path[-1] - path[0])) if len(path) > 1 else 0.0
        # Straightness is measured on a STRIDE-INDEPENDENT decimation of the
        # path. `path` is sampled every 100 steps, which is ~1/6 of a stride, so
        # measuring path length on it charges the fly for its own body sway and
        # reports ~0.43 for a trajectory that is actually near-straight. Sampling
        # roughly every 3 strides gives ~0.95 for the same run. Path length is
        # not a sampling-rate-free quantity, so the interval has to be pinned.
        # The decimation must KEEP THE LAST POINT, or the ratio below divides a
        # full-length displacement by a short path length and reports a
        # straightness above 1, which is geometrically impossible.
        coarse = np.concatenate([path[::20], path[-1:]]) if len(path) > 40 else path
        travelled = (
            float(np.linalg.norm(np.diff(coarse, axis=0), axis=1).sum())
            if len(coarse) > 1 else 0.0
        )
        coarse_displacement = (
            float(np.linalg.norm(coarse[-1] - coarse[0])) if len(coarse) > 1 else 0.0
        )
        # Net speed is the honest walking speed. mean_speed_mm_s is the mean of
        # instantaneous |velocity| and is inflated by per-step wobble; path
        # length is worse still, since it grows with the sampling rate.
        elapsed = self.steps * self.timestep
        return {
            "steps": self.steps,
            "net_speed_mm_s": float(displacement / elapsed) if elapsed else 0.0,
            "mean_speed_mm_s": float(speeds.mean()),
            # How much of the leg motion fails to become travel. High values
            # mean the fly is stepping briskly and going nowhere - slipping -
            # which is what low tarsal grip looks like.
            "slip_ratio": (
                float(1.0 - (displacement / elapsed) / speeds.mean())
                if elapsed and speeds.mean() > 1e-9 else 0.0
            ),
            "peak_speed_mm_s": float(speeds.max()),
            "net_displacement_mm": displacement,
            # 1.0 = perfectly straight; lower = more curved/wandering path.
            "path_straightness": (
                min(1.0, float(coarse_displacement / travelled)) if travelled > 1e-9 else 0.0
            ),
            "gait_regularity": float(np.mean(self.gait_regularity)) if self.gait_regularity else 0.0,
            # Std of each joint over time, averaged - a blunt "is it still moving
            # its legs" number that separates walking from flailing or freezing.
            "joint_excursion": float(traces.std(axis=0).mean()),
            "fell_over": self.fell_over,
        }


class Session:
    def __init__(
        self,
        preset: str = "dry_land",
        connectome: str = "neuprint",
        subset: str = DEFAULT_SUBSET,
        device: str = "cpu",
        seed: int = 0,
        render: bool = False,
        connectome_dir: str | Path = "/data/connectome",
        checkpoint: str | Path | None = None,
        synthetic_size: int = 20_000,
        world: str = DEFAULT_WORLD,
        camera_res: tuple[int, int] = (360, 480),
        timestep: float = DEFAULT_TIMESTEP,
        control_every: int = DEFAULT_CONTROL_EVERY,
    ) -> None:
        self.preset: EnvPreset = load_preset(preset)
        self.tables = get_tables(connectome, subset, connectome_dir, synthetic_size=synthetic_size)

        # The LIF clock follows the physics clock. At the default 1e-4 the brain
        # runs every second physics step at its usual 0.2 ms; at a coarser
        # timestep it runs every step and its own dt grows to match, so the
        # membrane time constants stay right in real terms.
        self.control_every = max(1, int(control_every))
        # Seconds per control step. Every timer in the flight logic is counted
        # in CONTROL steps, so converting them with the physics timestep makes
        # them `control_every` times too long - which stopped the fly taking off
        # at all.
        self.control_dt = timestep * self.control_every
        brain_every = max(1, round(LIFParams().dt_ms * 1e-3 / timestep))
        brain_every = max(brain_every, self.control_every)
        self.base_params = LIFParams(dt_ms=brain_every * timestep * 1000.0)
        self.net = LIFNetwork(self.tables, self._params_for(self.preset), device=device, seed=seed)
        if checkpoint:
            self.net.load_weights(checkpoint)

        self.decoder = MotorDecoder(self.net)
        self.body = FlyBody(self.preset, render=render, seed=seed, world=world,
                            camera_res=camera_res, timestep=timestep)

        # The LIF runs on a coarser clock than the physics; stepping a spiking
        # net at 1e-4 s buys nothing and costs 10x.
        self.brain_every = max(1, round(self.base_params.dt_ms * 1e-3 / self.body.timestep))
        # Sensory input is only ever READ by `net.step`, which runs every
        # `brain_every` physics steps - so encoding on every physics step threw
        # half the work away. It is encoded on the step immediately before the
        # brain runs, which is the sample the old code happened to use anyway,
        # so the trajectory is unchanged. dt_s follows, or receptor adaptation
        # would silently run at twice its intended time constant.
        brain_dt = self.brain_every * self.body.timestep
        self.encoder = SensoryEncoder(self.net, self.decoder, dt_s=brain_dt)
        self._seed = seed
        self.cpg = self._new_cpg()
        self._init_raster()

        self.step_count = 0
        self.paused = False
        self._samples_due = self._sample_interval()
        self.metrics = RunMetrics(timestep=self.body.timestep)
        self._external = torch.zeros(self.net.n, device=self.net.device)
        self._last_drive = self.decoder.decode()
        self._last_odour = np.zeros(2)
        self._reset_flight_state()

    def _sample_interval(self) -> int:
        """Control steps between metric samples, keeping the ~100-physics-step
        cadence the summary's straightness decimation assumes."""
        return max(1, round(100 / self.control_every))

    def _reset_flight_state(self) -> None:
        self._grounded_steps = 0     # steps since landing, for the takeoff delay
        self._avoid_steps = 0        # remaining steps of a latched avoidance turn
        self._avoid_direction = 1.0
        self._approaching = 0        # remaining steps of a latched landing approach

    # ---------- configuration ----------

    def _new_cpg(self) -> TripodCPG:
        # Swimming rows all six legs together; walking runs them as two tripods.
        kwargs = {}
        if self.preset.stroke_freq_hz is not None:
            kwargs["base_freq_hz"] = self.preset.stroke_freq_hz
        # The oscillator is advanced once per CONTROL step, so its own dt is the
        # control interval - not the physics timestep, or the stride frequency
        # silently divides by control_every.
        return TripodCPG(
            dt_s=self.body.timestep * self.control_every, seed=self._seed,
            synchronous=self.preset.locomotion == "swim",
            **kwargs,
        )

    def _init_raster(self) -> None:
        """Pick the neurons the activity panel shows, once."""
        groups = [("DN_L", self.decoder.dn_left), ("DN_R", self.decoder.dn_right)]
        groups += [(leg, self.decoder.leg_groups[leg]) for leg in LEG_ORDER]

        rows, self.raster_groups = [], []
        for name, idx in groups:
            take = min(RASTER_ROWS_PER_GROUP, len(idx))
            if take:
                # Evenly spaced rather than random: reproducible, and it spreads
                # the sample across the whole pool.
                pick = np.linspace(0, len(idx) - 1, take).round().astype(int)
                rows.extend(int(idx[i]) for i in pick)
            self.raster_groups.append({"name": name, "rows": take, "size": len(idx)})

        self._raster_idx = torch.tensor(rows, dtype=torch.long, device=self.net.device)
        self.raster_rows = len(rows)
        self._raster = np.zeros((self.raster_rows, RASTER_COLS), dtype=np.uint8)
        self._raster_cursor = 0
        self._spike_accum = torch.zeros(self.net.n, device=self.net.device)
        self._raster_bin = 0
        self._raster_bin_steps = max(1, round(RASTER_BIN_MS / self.base_params.dt_ms))

    def _accumulate_spikes(self) -> None:
        """One add per brain step; a device round-trip only once per column."""
        self._spike_accum += self.net.spikes
        self._raster_bin += 1
        if self._raster_bin < self._raster_bin_steps:
            return
        self._raster_bin = 0
        counts = self._spike_accum[self._raster_idx].clamp(max=255).to("cpu").numpy()
        self._raster[:, self._raster_cursor] = counts.astype(np.uint8)
        self._raster_cursor = (self._raster_cursor + 1) % RASTER_COLS
        self._spike_accum.zero_()

    def _params_for(self, preset: EnvPreset) -> LIFParams:
        params = self.base_params.scaled(preset.scaled_taus(self.base_params.base_taus()))
        if preset.tonic_drive is not None:
            params = replace(params, tonic_drive=preset.tonic_drive)
        return params

    def switch_preset(self, name: str) -> None:
        """Change environment mid-run: physics options and neuron taus together."""
        previous = self.preset.locomotion
        self.preset = load_preset(name)
        self._set_arousal(FLIGHT_TONIC_DRIVE if self.body.airborne else None)
        self.body.apply_preset(self.preset)
        if self.preset.locomotion != previous:
            # Switching between walking and swimming changes the inter-leg
            # coordination, so the oscillators have to be rebuilt.
            self.cpg = self._new_cpg()

    def reset(self) -> None:
        self.net.reset()
        self.body.reset()
        self.encoder.reset()
        self.cpg = self._new_cpg()
        self.step_count = 0
        self._samples_due = self._sample_interval()
        self.metrics = RunMetrics(timestep=self.body.timestep)
        self._raster[:] = 0
        self._raster_cursor = 0
        self._raster_bin = 0
        self._spike_accum.zero_()
        self._last_odour = np.zeros(2)
        self._reset_flight_state()
        self.body.land()
        self._set_arousal(None)

    # ---------- the loop ----------

    def step(self) -> None:
        if self.paused:
            return
        if self.step_count % self.control_every:
            # Physics-only substep: the controller keeps its last command.
            self.body.substep()
            self.step_count += 1
            return

        if self.step_count % self.brain_every == 0:
            self.net.step(self._external)
            self._accumulate_spikes()
            self._last_drive = self.decoder.decode()

        drive = self._last_drive
        # Legs hold their neutral pose in the air; a fly does not run while it
        # flies, and a swinging gait would fight the attitude controller.
        phase, amplitude = self.cpg.step(
            forward=drive.forward, turn=drive.turn,
            per_leg_gain=drive.per_leg_gain, stop=drive.stop or self.body.airborne,
        )
        obs = self.body.step(phase, amplitude)

        self._last_odour = obs["odour_lr"]
        if self.body.can_fly:
            self._fly_or_walk(obs)
        # Encode on the control step immediately before the next brain step -
        # that is the sample the network will read. Written as `+ control_every`
        # and not `+ 1` because control steps arrive `control_every` apart: with
        # `+ 1` the condition simply never fired once the controller ran slower
        # than the physics, and the sensory loop hung open with the network
        # seeing nothing but its tonic drive.
        if (self.step_count + self.control_every) % self.brain_every == 0:
            self._external = self.encoder.encode(
                leg_load=obs["leg_load"],
                air_speed=obs["air_speed"],
                mechanosensory_gain=float(self.preset.sensory["mechanosensory_gain"]),
                johnstons_organ_gain=float(self.preset.sensory["johnstons_organ_gain"]),
                odour_lr=obs["odour_lr"] if self.body.has_odour else None,
                olfactory_gain=float(self.preset.sensory.get("olfactory_gain", 1.0)),
            )

        self.step_count += 1
        self._record(obs)

    def _record(self, obs: dict) -> None:
        m = self.metrics
        m.steps = self.step_count
        # Sampling at 100 Hz keeps a 50k-step run's metrics small. Counted down
        # rather than tested with `step_count % 100`, which never fires once the
        # controller runs every few physics steps and the counter arrives in
        # strides that miss the multiples of 100.
        self._samples_due -= 1
        if self._samples_due <= 0:
            self._samples_due = self._sample_interval()
            m.path.append(obs["position"][:2].tolist())
            m.speeds.append(float(obs["speed"]))
            m.gait_regularity.append(self.cpg.gait_regularity())
            m.joint_traces.append(obs["joint_angles"].tolist())
        if obs["fell_over"]:
            m.fell_over = True

    def run(self, steps: int, progress_every: int = 0) -> dict:
        for _ in range(steps):
            self.step()
            if progress_every and self.step_count % progress_every == 0:
                print(f"  step {self.step_count}/{steps}  {self.behaviour_mode()}", flush=True)
        return self.metrics.summary()

    # ---------- readouts ----------

    def _set_arousal(self, tonic_drive: float | None) -> None:
        """Override the preset's background excitation, or restore it."""
        params = self._params_for(self.preset)
        if tonic_drive is not None:
            params = replace(params, tonic_drive=tonic_drive)
        self.net.set_params(params)

    def _fly_or_walk(self, obs: dict) -> None:
        """Take off when a smell is exhausted, land when a new one turns up."""
        odour = float(np.mean(obs["odour_lr"]))
        if not self.body.airborne:
            self._grounded_steps += 1
            settled = self._grounded_steps * self.control_dt > GROUNDED_BEFORE_TAKEOFF_S
            if settled and odour < ODOUR_EXHAUSTED and not self._last_drive.stop:
                if self.body.takeoff():
                    self._set_arousal(FLIGHT_TONIC_DRIVE)
            return

        # Commit to a landing the moment a smell is worth landing on, and stay
        # committed until the feet are down or the smell is lost entirely.
        if odour > ODOUR_INTERESTING:
            self._approaching = int(FLIGHT_APPROACH_HOLD_S / self.control_dt)
        elif self._approaching:
            self._approaching -= 1

        if self._approaching and obs["position"][2] < FLIGHT_LAND_FROM_MM:
            self.body.land()
            self._set_arousal(None)
            self._approaching = 0
            self._grounded_steps = 0
            return

        bump = self.body.bumped_into()
        if bump:
            self._avoid_steps = int(FLIGHT_AVOID_HOLD_S / self.control_dt)
            self._avoid_direction = bump
        if self._avoid_steps > 0:
            self._avoid_steps -= 1
            self.body.set_flight_command(
                climb_mm_s=80.0, forward_mm_s=FLIGHT_AVOID_BACK_MM_S,
                yaw_rate=FLIGHT_AVOID_YAW * self._avoid_direction,
            )
            return

        drive = self._last_drive
        position = obs["position"]
        target = FLIGHT_APPROACH_HEIGHT_MM if self._approaching else FLIGHT_CRUISE_HEIGHT_MM
        cruise = FLIGHT_APPROACH_MM_S if self._approaching else FLIGHT_CRUISE_MM_S
        yaw_rate = -FLIGHT_YAW_RATE * float(drive.turn)

        bounds = getattr(self.body.world, "flight_bounds", None)
        if bounds is not None:
            half_x, half_y, ceiling = bounds
            near_wall = (abs(position[0]) > half_x - FLIGHT_WALL_MARGIN_MM
                         or abs(position[1]) > half_y - FLIGHT_WALL_MARGIN_MM)
            if near_wall:
                yaw_rate = self._yaw_towards(-position[0], -position[1])
            target = min(target, ceiling - FLIGHT_WALL_MARGIN_MM)

        climb = float(np.clip(FLIGHT_CLIMB_GAIN * (target - position[2]),
                              -FLIGHT_CLIMB_MAX_MM_S, FLIGHT_CLIMB_MAX_MM_S))
        self.body.set_flight_command(
            climb_mm_s=climb,
            forward_mm_s=cruise * float(np.clip(drive.forward, 0.0, 1.0)),
            yaw_rate=yaw_rate,
        )

    def _yaw_towards(self, dx: float, dy: float) -> float:
        """Yaw rate that turns the fly towards a direction in the world plane.

        A positive yaw-rate command turns the flying fly LEFT (verified from the
        flown path), so the heading error is used directly.
        """
        rotation = self.body._body_frame()
        heading = np.arctan2(rotation[1, 0], rotation[0, 0])
        error = (np.arctan2(dy, dx) - heading + np.pi) % (2 * np.pi) - np.pi
        return float(np.clip(FLIGHT_HOME_YAW_GAIN * error, -FLIGHT_YAW_RATE, FLIGHT_YAW_RATE))

    def nearest_source(self) -> dict | None:
        """Which smell is closest, for the UI. Not used by the simulation."""
        sources = self.body.world.odour_sources
        if not sources:
            return None
        x, y = self.body._prev_pos[:2]
        best = min(sources, key=lambda s: (s.x - x) ** 2 + (s.y - y) ** 2)
        return {"name": best.name,
                "distance_mm": round(float(np.hypot(best.x - x, best.y - y)), 2)}

    def behaviour_mode(self) -> str:
        """Coarse label for the UI. A threshold on the decoded drive, nothing more."""
        d = self._last_drive
        if self.body.airborne:
            if abs(d.turn) > 0.35:
                return "flying-" + ("right" if d.turn > 0 else "left")
            return "flying"
        if d.stop or d.forward < 0.1:
            return "stopped"
        gait = "swimming" if self.preset.locomotion == "swim" else "walking"
        if abs(d.turn) > 0.35:
            return f"{gait}-" + ("right" if d.turn > 0 else "left")
        return gait

    def neural_frame(self) -> dict:
        """A JSON-safe snapshot of network activity for the UI.

        The raster is a base64 uint8 image of spike counts per (neuron, 5 ms
        bin), oldest column first. Sending it as bytes rather than nested lists
        keeps a 72x160 window at ~15 kB instead of ~90 kB of JSON numbers.
        """
        ordered = np.roll(self._raster, -self._raster_cursor, axis=1)
        return {
            "rows": self.raster_rows,
            "cols": RASTER_COLS,
            "bin_ms": self._raster_bin_steps * self.base_params.dt_ms,
            "groups": self.raster_groups,
            "raster": base64.b64encode(ordered.tobytes()).decode("ascii"),
            "rates_hz": {k: round(v, 2) for k, v in self.decoder.population_rates().items()},
        }

    def live_metrics(self) -> dict:
        d = self._last_drive
        mean_rate_hz = float(self.net.rate.mean()) * 1000.0 / self.base_params.dt_ms
        adapted = self.encoder.last_odour_adapted
        return {
            "step": self.step_count,
            "preset": self.preset.name,
            "world": self.body.world_name,
            "mode": self.behaviour_mode(),
            "odour": round(float(np.mean(self._last_odour)), 4),
            # What actually steers: the adapted left/right difference. Zero
            # means the antennae agree, or the receptors have habituated.
            "odour_lr_diff": round(float(adapted[1] - adapted[0]), 5),
            "nearest_source": self.nearest_source(),
            "airborne": self.body.airborne,
            "height_mm": round(float(self.body._prev_pos[2]), 1),
            "speed_mm_s": float(self.body.last_speed),
            "gait_regularity": self.cpg.gait_regularity(),
            "forward": d.forward,
            "turn": d.turn,
            "mean_rate_hz": mean_rate_hz,
            "n_neurons": self.net.n,
            "connectome": f"{self.tables.source}:{self.tables.dataset}",
            "paused": self.paused,
        }
