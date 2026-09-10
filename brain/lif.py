"""Sparse leaky integrate-and-fire network over connectome edges.

Deliberately plain: exponential synaptic current, exponential membrane leak,
hard threshold, fixed refractory period. Everything is in normalised units
(threshold = 1.0) rather than millivolts, because calibrating this to real
membrane physiology is not something the connectome alone lets you do, and
dressing it up in mV would imply a precision that is not there.

`brain/` never imports MuJoCo or FlyGym.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from brain.tables import ConnectomeTables


@dataclass(frozen=True)
class LIFParams:
    tau_m_ms: float = 20.0      # membrane time constant
    tau_s_ms: float = 5.0       # synaptic current decay
    refractory_ms: float = 2.2  # male-cns has no refractory data; a standard ~2 ms
    v_threshold: float = 1.0    # normalised units, by definition
    v_reset: float = 0.0
    dt_ms: float = 0.2          # brain step; see BRAIN_STEPS_PER_PHYSICS in body/
    # Scales the whole weight matrix so raw neuPrint synapse counts (1..723)
    # produce sane drive. PLACEHOLDER - it is a gain knob, not a measurement.
    target_row_sum: float = 4.0
    noise_std: float = 0.02     # keeps a silent network from being exactly dead
    # Standing in for everything the subset leaves out. With --neuron-subset
    # motor the network is just DNs and MNs, so the ~13k VNC interneurons and
    # the whole brain that would normally drive them are absent and nothing
    # reaches threshold. This is a tonic background current, not a connectome
    # mechanism, and it is listed in the README's approximations table.
    # Threshold is 1.0, so values slightly above it give a live baseline.
    tonic_drive: float = 1.05

    def scaled(self, taus_ms: dict[str, float]) -> "LIFParams":
        """Return a copy with Q10-scaled time constants applied."""
        return replace(
            self,
            tau_m_ms=taus_ms.get("tau_m_ms", self.tau_m_ms),
            tau_s_ms=taus_ms.get("tau_s_ms", self.tau_s_ms),
            refractory_ms=taus_ms.get("refractory_ms", self.refractory_ms),
        )

    def base_taus(self) -> dict[str, float]:
        return {
            "tau_m_ms": self.tau_m_ms,
            "tau_s_ms": self.tau_s_ms,
            "refractory_ms": self.refractory_ms,
        }


class LIFNetwork:
    """Fixed-connectivity spiking network built from a ConnectomeTables."""

    def __init__(
        self,
        tables: ConnectomeTables,
        params: LIFParams | None = None,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.params = params or LIFParams()
        self.device = torch.device(device)
        self.tables = tables
        self._gen = torch.Generator(device=self.device).manual_seed(seed)

        # Contiguous 0..N-1 indices; bodyId is the external name.
        self.body_ids = tables.neurons["bodyId"].to_numpy()
        self.n = len(self.body_ids)
        self._index_of = {int(b): i for i, b in enumerate(self.body_ids)}

        self.W = self._build_weights(tables)
        self._rebuild_csr()
        self._alloc_state()
        self._recompute_decays()

    # ---------- construction ----------

    def _build_weights(self, tables: ConnectomeTables) -> torch.Tensor:
        e = tables.edges
        pre = np.fromiter((self._index_of[int(p)] for p in e["pre"]), dtype=np.int64, count=len(e))
        post = np.fromiter((self._index_of[int(p)] for p in e["post"]), dtype=np.int64, count=len(e))

        sign = tables.neurons["sign"].to_numpy(dtype=np.float32)
        vals = e["weight"].to_numpy(dtype=np.float32) * sign[pre]

        # Normalise so the mean absolute input per postsynaptic neuron hits
        # target_row_sum. Without this, raw synapse counts either do nothing or
        # saturate every neuron on the first step.
        row_abs = np.zeros(self.n, dtype=np.float64)
        np.add.at(row_abs, post, np.abs(vals))
        mean_row = float(row_abs[row_abs > 0].mean()) if (row_abs > 0).any() else 1.0
        vals = vals * (self.params.target_row_sum / max(mean_row, 1e-9))

        # [post, pre] so that W @ spikes gives postsynaptic input.
        idx = torch.from_numpy(np.stack([post, pre])).to(self.device)
        return torch.sparse_coo_tensor(
            idx, torch.from_numpy(vals).to(self.device), (self.n, self.n)
        ).coalesce()

    def _rebuild_csr(self) -> None:
        """Keep a SciPy CSR copy of W for the CPU path. See `_recurrent`."""
        if self.device.type != "cpu":
            self._csr = None
            return
        w = self.W.coalesce()
        rows, cols = w.indices().cpu().numpy()
        self._csr = sp.csr_matrix(
            (w.values().cpu().numpy().astype(np.float32), (rows, cols)),
            shape=(self.n, self.n),
        )

    def _recurrent(self) -> torch.Tensor:
        """W @ spikes.

        torch's CPU sparse matmul is startlingly slow at this size - measured
        575 us for a 2,129 x 2,129 matrix with 60k non-zeros, which is the
        single largest cost in the whole simulation, ahead of `mj_step`. SciPy
        multiplies the identical matrix in 55 us and returns bit-identical
        values (verified: max abs difference 0.0). Even torch's DENSE matvec is
        faster than its sparse one here.

        So the CPU path goes through SciPy and anything else stays on torch.
        Splitting it this way rather than dropping torch keeps the CUDA path
        that `--gpu` documents, and the big subsets (`vnc` at ~17k neurons,
        `all` at ~176k) are where a device backend actually earns its keep.
        """
        if self._csr is None:
            return torch.sparse.mm(self.W, self.spikes.float().unsqueeze(1)).squeeze(1)
        # `spikes` is a CPU bool tensor, so .numpy() is a view, not a copy.
        return torch.from_numpy(self._csr @ self.spikes.numpy().astype(np.float32))

    def _alloc_state(self) -> None:
        z = lambda: torch.zeros(self.n, device=self.device)  # noqa: E731
        self.v = z()
        self.i_syn = z()
        self.refrac = z()
        self.spikes = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.rate = z()  # low-pass spike trace, what the bridge reads

    def _recompute_decays(self) -> None:
        p = self.params
        self._decay_v = math.exp(-p.dt_ms / p.tau_m_ms)
        self._decay_i = math.exp(-p.dt_ms / p.tau_s_ms)
        # Rate trace smoothed over ~50 ms so the bridge sees populations, not
        # individual spikes.
        self._decay_r = math.exp(-p.dt_ms / 50.0)
        self._refrac_steps = max(1, round(p.refractory_ms / p.dt_ms))

    # ---------- runtime ----------

    def set_params(self, params: LIFParams) -> None:
        """Apply new time constants (e.g. after a preset switch) in place."""
        self.params = params
        self._recompute_decays()

    def reset(self) -> None:
        self._alloc_state()

    def step(self, external: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one dt. `external` is an input current per neuron."""
        p = self.params

        self.i_syn = self.i_syn * self._decay_i + self._recurrent()
        drive = self.i_syn + p.tonic_drive
        if external is not None:
            drive = drive + external
        if p.noise_std > 0:
            drive = drive + torch.randn(
                self.n, device=self.device, generator=self._gen
            ) * p.noise_std

        self.v = self.v * self._decay_v + (1.0 - self._decay_v) * drive
        self.v = torch.where(self.refrac > 0, torch.full_like(self.v, p.v_reset), self.v)

        self.spikes = (self.v >= p.v_threshold) & (self.refrac <= 0)
        self.v = torch.where(self.spikes, torch.full_like(self.v, p.v_reset), self.v)
        self.refrac = torch.clamp(self.refrac - 1, min=0)
        self.refrac = torch.where(
            self.spikes, torch.full_like(self.refrac, self._refrac_steps), self.refrac
        )

        self.rate = self.rate * self._decay_r + (1.0 - self._decay_r) * self.spikes.float()
        return self.spikes

    def indices_for(self, body_ids) -> torch.Tensor:
        return torch.tensor(
            [self._index_of[int(b)] for b in body_ids], dtype=torch.long, device=self.device
        )

    # ---------- checkpoints ----------

    def save(self, path: str | Path) -> None:
        """Persist weights + params so --compare-envs can reuse one brain snapshot."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "body_ids": self.body_ids,
                "W_indices": self.W.indices().cpu(),
                "W_values": self.W.values().cpu(),
                "params": asdict(self.params),
                "source": self.tables.source,
                "dataset": self.tables.dataset,
            },
            path,
        )

    def load_weights(self, path: str | Path) -> None:
        """Restore synaptic weights ONLY - deliberately not the parameters.

        The time constants in a checkpoint are Q10-scaled by whichever preset
        happened to save it. Restoring them here silently overwrote every
        preset's temperature with the first one's, which is precisely what
        --compare-envs exists to vary: hot, cold and temperate all ran with the
        same taus and produced identical behaviour. The checkpoint holds the
        brain; the preset owns the temperature.
        """
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if list(ck["body_ids"]) != list(self.body_ids):
            raise ValueError(
                f"checkpoint {path} was built for a different neuron set "
                f"({len(ck['body_ids'])} neurons vs {self.n} here)"
            )
        self.W = torch.sparse_coo_tensor(
            ck["W_indices"].to(self.device), ck["W_values"].to(self.device), (self.n, self.n)
        ).coalesce()
        self._rebuild_csr()
