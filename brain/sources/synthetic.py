"""A structurally-plausible FAKE connectome.

This is NOT the fly. It exists so the whole stack - LIF engine, bridge, body,
presets, web UI, tests - runs with no credentials, no 1.1 GB download and no
network. It reproduces the *shape* of male-cns:v1.0 (role counts, leg
subclasses, prose motor-neuron type names, neurotransmitter mix, sparse
log-normal weights) so that code which works here works on real data, but every
edge is random. Any behaviour it produces is meaningless biologically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brain.tables import ConnectomeTables, sign_from_nt

# Role counts and subclass splits copied verbatim from live male-cns:v1.0
# queries, so downstream code sees realistic population sizes and the same
# per-leg group names even in synthetic mode.
DN_SUBCLASSES = {"xn": 441, "ut": 276, "xl": 275, "fl": 115, "nt": 108,
                 "lt": 31, "ht": 26, "it": 18, "ad": 10, "wt": 2, "hl": 2, None: 10}
# Derived, not restated, so the two can never drift out of sync (the real
# dataset has 1314 descending neurons).
N_DESCENDING = sum(DN_SUBCLASSES.values())
VNC_MOTOR_SUBCLASSES = {"ad": 214, "fl": 135, "hl": 130, "ml": 116,
                        "wm": 67, "nm": 24, "hm": 16, "xm": 6}   # total 708
CB_MOTOR_SUBCLASSES = {"pm": 67, "nm": 20, "am": 13, "rm": 7}    # total 107

# Prose MN type names, matching the male-cns convention (NOT MANC's 'MNfl').
_MN_TYPES = [
    "Ti flexor MN", "Ti extensor MN", "Ta depressor MN", "Ta levator MN",
    "Fe reductor MN", "Tr flexor MN", "Tr extensor MN", "Acc. ti flexor MN",
    "Tergotr. MN", "ltm MN",
]

# consensusNt proportions from the real dataset.
_NT_MIX = {
    "acetylcholine": 0.590, "glutamate": 0.167, "gaba": 0.126,
    "unclear": 0.055, "histamine": 0.045, "dopamine": 0.0022,
    "octopamine": 0.0006, "serotonin": 0.0003,
}


def build(n_neurons: int = 20_000, mean_degree: int = 40, seed: int = 0) -> ConnectomeTables:
    """Generate a fake connectome with male-cns-shaped role structure."""
    rng = np.random.default_rng(seed)

    n_motor = sum(VNC_MOTOR_SUBCLASSES.values()) + sum(CB_MOTOR_SUBCLASSES.values())
    n_special = N_DESCENDING + n_motor
    if n_neurons < n_special:
        raise ValueError(f"n_neurons must be >= {n_special} to hold all DNs and MNs")
    n_intrinsic = n_neurons - n_special

    superclass, subclass, types = [], [], []

    superclass += ["descending_neuron"] * N_DESCENDING
    for sub, count in DN_SUBCLASSES.items():
        subclass += [sub] * count
    types += [f"DN{p}{i:03d}" for i, p in zip(range(N_DESCENDING), rng.choice(list("agp"), N_DESCENDING))]

    for sc, table in (("vnc_motor", VNC_MOTOR_SUBCLASSES), ("cb_motor", CB_MOTOR_SUBCLASSES)):
        for sub, count in table.items():
            superclass += [sc] * count
            subclass += [sub] * count
            types += list(rng.choice(_MN_TYPES, count))

    superclass += ["vnc_intrinsic"] * n_intrinsic
    subclass += [None] * n_intrinsic
    types += [f"IN{i:06d}" for i in range(n_intrinsic)]

    # Normalised rather than hand-balanced: the proportions are rounded
    # observed frequencies and do not sum to exactly 1.
    probs = np.fromiter(_NT_MIX.values(), float)
    nts = rng.choice(list(_NT_MIX), size=n_neurons, p=probs / probs.sum())
    neurons = pd.DataFrame(
        {
            "bodyId": np.arange(1, n_neurons + 1, dtype=np.int64),
            "type": types,
            "superclass": superclass,
            "subclass": subclass,
            "somaSide": rng.choice(["L", "R"], size=n_neurons),
            "nt": nts,
        }
    )
    neurons["sign"] = sign_from_nt(neurons["nt"])

    # Sparse random edges. Real connectomes are far from uniform-random, but the
    # point here is exercising the sparse path, not modelling structure.
    n_edges = n_neurons * mean_degree
    edges = pd.DataFrame(
        {
            "pre": rng.integers(1, n_neurons + 1, n_edges, dtype=np.int64),
            "post": rng.integers(1, n_neurons + 1, n_edges, dtype=np.int64),
            # Synaptic counts in neuPrint are heavy-tailed; log-normal is a
            # reasonable stand-in. Rounded to integers like real weights.
            "weight": np.maximum(1, rng.lognormal(1.0, 1.0, n_edges).astype(np.int32)),
        }
    )
    edges = edges[edges["pre"] != edges["post"]].reset_index(drop=True)  # drop autapses

    return ConnectomeTables(neurons=neurons, edges=edges, source="synthetic", dataset="synthetic-v1")
