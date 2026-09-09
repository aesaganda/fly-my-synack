"""The one schema every connectome source must produce.

`brain/` deliberately knows nothing about MuJoCo or FlyGym. It turns a
connectome into neuron/edge tables and integrates a LIF network over them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Neurotransmitter -> synaptic sign.
#
# In Drosophila, glutamate and histamine are INHIBITORY (they gate GluCl and
# HisCl chloride channels), which is the opposite of the vertebrate intuition.
# Getting this backwards silently flips ~30% of the network.
NT_SIGN: dict[str, float] = {
    "acetylcholine": +1.0,
    "dopamine": +1.0,
    "octopamine": +1.0,
    "serotonin": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
}

# ~12k neurons in male-cns:v1.0 have consensusNt of 'unclear' or null. Treating
# them as excitatory follows the base rate (acetylcholine is ~59% of the
# dataset) but it IS an assumption, not a measurement. Listed in the README's
# approximations table.
UNKNOWN_NT_SIGN = +1.0

# Superclass values used for role selection. Verified against male-cns:v1.0 -
# note there is NO `class == 'descending neuron'`; `class` holds sensory
# categories and is populated for only ~26k neurons. `superclass` is the
# authoritative field.
DESCENDING = ("descending_neuron",)
MOTOR = ("vnc_motor", "cb_motor")
INTRINSIC = ("vnc_intrinsic",)

NEURON_COLUMNS = ["bodyId", "type", "superclass", "subclass", "somaSide", "nt", "sign"]
EDGE_COLUMNS = ["pre", "post", "weight"]


@dataclass(frozen=True)
class ConnectomeTables:
    """Neurons and weighted directed edges, in a source-independent form."""

    neurons: pd.DataFrame  # NEURON_COLUMNS
    edges: pd.DataFrame  # EDGE_COLUMNS
    source: str  # "neuprint" | "feather" | "synthetic" - reported in run metadata
    dataset: str  # e.g. "male-cns:v1.0", or "synthetic-v1" for the fake one

    def __post_init__(self) -> None:
        missing_n = [c for c in NEURON_COLUMNS if c not in self.neurons.columns]
        missing_e = [c for c in EDGE_COLUMNS if c not in self.edges.columns]
        if missing_n:
            raise ValueError(f"{self.source}: neurons table missing {missing_n}")
        if missing_e:
            raise ValueError(f"{self.source}: edges table missing {missing_e}")

    def ids_by_superclass(self, superclasses: tuple[str, ...]) -> pd.Index:
        sel = self.neurons["superclass"].isin(superclasses)
        return pd.Index(self.neurons.loc[sel, "bodyId"])

    def summary(self) -> str:
        n = self.neurons
        return (
            f"{self.source}:{self.dataset} "
            f"{len(n)} neurons, {len(self.edges)} edges, "
            f"{len(self.ids_by_superclass(DESCENDING))} DN, "
            f"{len(self.ids_by_superclass(MOTOR))} MN"
        )


def sign_from_nt(nt: pd.Series) -> pd.Series:
    """Map a neurotransmitter column to +1/-1, defaulting unknowns."""
    return nt.str.lower().map(NT_SIGN).fillna(UNKNOWN_NT_SIGN).astype("float32")
