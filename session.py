"""Wires brain -> bridge -> body -> bridge -> brain and runs the loop.

Shared by the CLI (`run.py`) and the web service, so both drive an identical
simulation. This is the only module that imports from all four packages; the
packages themselves stay independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from body.sim import FlyBody
from brain.fetch import get_tables
from brain.lif import LIFNetwork, LIFParams
from brain.subset import DEFAULT_SUBSET
from bridge.decode import MotorDecoder
from bridge.encode import SensoryEncoder
from body.cpg import TripodCPG
from env.loader import EnvPreset, load_preset


@dataclass
class RunMetrics:
    """Locomotion kinematics - what --compare-envs diffs between presets."""

    steps: int = 0
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
        elapsed = self.steps * 1e-4
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
    ) -> None:
        self.preset: EnvPreset = load_preset(preset)
        self.tables = get_tables(connectome, subset, connectome_dir, synthetic_size=synthetic_size)

        self.base_params = LIFParams()
        self.net = LIFNetwork(self.tables, self._params_for(self.preset), device=device, seed=seed)
        if checkpoint:
            self.net.load_weights(checkpoint)

        self.decoder = MotorDecoder(self.net)
        self.encoder = SensoryEncoder(self.net, self.decoder)
        self.body = FlyBody(self.preset, render=render, seed=seed)
        self._seed = seed
        self.cpg = self._new_cpg()

        # The LIF runs on a coarser clock than the physics; stepping a spiking
        # net at 1e-4 s buys nothing and costs 10x.
        self.brain_every = max(1, round(self.base_params.dt_ms * 1e-3 / self.body.timestep))

        self.step_count = 0
        self.paused = False
        self.metrics = RunMetrics()
        self._external = torch.zeros(self.net.n, device=self.net.device)
        self._last_drive = self.decoder.decode()

    # ---------- configuration ----------

    def _new_cpg(self) -> TripodCPG:
        # Swimming rows all six legs together; walking runs them as two tripods.
        kwargs = {}
        if self.preset.stroke_freq_hz is not None:
            kwargs["base_freq_hz"] = self.preset.stroke_freq_hz
        return TripodCPG(
            dt_s=self.body.timestep, seed=self._seed,
            synchronous=self.preset.locomotion == "swim",
            **kwargs,
        )

    def _params_for(self, preset: EnvPreset) -> LIFParams:
        params = self.base_params.scaled(preset.scaled_taus(self.base_params.base_taus()))
        if preset.tonic_drive is not None:
            params = replace(params, tonic_drive=preset.tonic_drive)
        return params

    def switch_preset(self, name: str) -> None:
        """Change environment mid-run: physics options and neuron taus together."""
        previous = self.preset.locomotion
        self.preset = load_preset(name)
        self.net.set_params(self._params_for(self.preset))
        self.body.apply_preset(self.preset)
        if self.preset.locomotion != previous:
            # Switching between walking and swimming changes the inter-leg
            # coordination, so the oscillators have to be rebuilt.
            self.cpg = self._new_cpg()

    def reset(self) -> None:
        self.net.reset()
        self.body.reset()
        self.cpg = self._new_cpg()
        self.step_count = 0
        self.metrics = RunMetrics()

    # ---------- the loop ----------

    def step(self) -> None:
        if self.paused:
            return

        if self.step_count % self.brain_every == 0:
            self.net.step(self._external)
            self._last_drive = self.decoder.decode()

        drive = self._last_drive
        phase, amplitude = self.cpg.step(
            forward=drive.forward, turn=drive.turn,
            per_leg_gain=drive.per_leg_gain, stop=drive.stop,
        )
        obs = self.body.step(phase, amplitude)

        self._external = self.encoder.encode(
            leg_load=obs["leg_load"],
            air_speed=obs["air_speed"],
            mechanosensory_gain=float(self.preset.sensory["mechanosensory_gain"]),
            johnstons_organ_gain=float(self.preset.sensory["johnstons_organ_gain"]),
        )

        self.step_count += 1
        self._record(obs)

    def _record(self, obs: dict) -> None:
        m = self.metrics
        m.steps = self.step_count
        # Sampling at 100 Hz keeps a 50k-step run's metrics small.
        if self.step_count % 100 == 0:
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

    def behaviour_mode(self) -> str:
        """Coarse label for the UI. A threshold on the decoded drive, nothing more."""
        d = self._last_drive
        if d.stop or d.forward < 0.1:
            return "stopped"
        gait = "swimming" if self.preset.locomotion == "swim" else "walking"
        if abs(d.turn) > 0.35:
            return f"{gait}-" + ("right" if d.turn > 0 else "left")
        return gait

    def live_metrics(self) -> dict:
        d = self._last_drive
        mean_rate_hz = float(self.net.rate.mean()) * 1000.0 / self.base_params.dt_ms
        return {
            "step": self.step_count,
            "preset": self.preset.name,
            "mode": self.behaviour_mode(),
            "speed_mm_s": float(self.body.last_speed),
            "gait_regularity": self.cpg.gait_regularity(),
            "forward": d.forward,
            "turn": d.turn,
            "mean_rate_hz": mean_rate_hz,
            "n_neurons": self.net.n,
            "connectome": f"{self.tables.source}:{self.tables.dataset}",
            "paused": self.paused,
        }
