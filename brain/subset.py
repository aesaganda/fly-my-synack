"""Neuron-subset selection, applied identically to every connectome source.

The full male-cns network is ~176k neurons / ~25.6M edges, which is not what you
want by default on a laptop or a 12 GB GPU. Subsets are defined by `superclass`,
which is the authoritative role field in male-cns:v1.0.
"""

from __future__ import annotations

import pandas as pd

from brain.tables import DESCENDING, INTRINSIC, MOTOR, ConnectomeTables

# name -> superclasses kept ( None = keep everything )
SUBSETS: dict[str, tuple[str, ...] | None] = {
    # Default. Just the input and output layers the bridge actually reads:
    # ~1.3k descending + ~0.8k motor neurons. Runs on CPU in seconds.
    "motor": DESCENDING + MOTOR,
    # Adds the VNC interneurons that sit between them (~13k more).
    "vnc": DESCENDING + MOTOR + INTRINSIC + ("ascending_neuron",),
    # Everything.
    "all": None,
}

DEFAULT_SUBSET = "motor"


def apply_subset(tables: ConnectomeTables, subset: str) -> ConnectomeTables:
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}, expected one of {sorted(SUBSETS)}")
    keep = SUBSETS[subset]
    if keep is None:
        return tables

    neurons = tables.neurons[tables.neurons["superclass"].isin(keep)].reset_index(drop=True)
    kept_ids = pd.Index(neurons["bodyId"])
    edges = tables.edges[
        tables.edges["pre"].isin(kept_ids) & tables.edges["post"].isin(kept_ids)
    ].reset_index(drop=True)
    return ConnectomeTables(
        neurons=neurons, edges=edges, source=tables.source, dataset=tables.dataset
    )
