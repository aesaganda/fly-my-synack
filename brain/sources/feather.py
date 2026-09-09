"""Read the public male-cns bulk Feather exports from a local directory.

Files come from https://male-cns.janelia.org/download/ (CC-BY 4.0):

  connectome-weights-male-cns-v1.0-minconf-0.5.feather   1.1 GB  adjacency
  body-annotations-male-cns-v1.0-minconf-0.5.feather      13 MB  roles
  body-neurotransmitters-male-cns-v1.0.feather            42 MB  NT

This is the only path that can carry the full 176k-neuron network. Automated
download may be blocked on your network (see README); the files can simply be
dropped into the mounted /data/connectome directory instead.

The exact column names inside these files were NOT verifiable while writing
this (storage.googleapis.com is proxy-blocked here), so every column is
resolved through the alias tables below rather than assumed. Run
`python -m brain.sources.feather --schema <dir>` to print what a file actually
contains and extend the aliases if needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from brain.tables import ConnectomeTables, sign_from_nt

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

# canonical name -> candidate column names, first match wins.
_ALIASES: dict[str, tuple[str, ...]] = {
    "bodyId": ("bodyId", "body_id", "bodyid", "id"),
    "type": ("type", "cell_type", "celltype"),
    "superclass": ("superclass", "super_class"),
    "subclass": ("subclass", "sub_class"),
    "somaSide": ("somaSide", "soma_side", "side"),
    "nt": ("consensusNt", "consensus_nt", "predictedNt", "predicted_nt", "nt", "neurotransmitter"),
    "pre": ("bodyId_pre", "body_id_pre", "pre", "pre_id", "bodyid_pre"),
    "post": ("bodyId_post", "body_id_post", "post", "post_id", "bodyid_post"),
    "weight": ("weight", "count", "n_syn", "synapses"),
}


def _pick(df: pd.DataFrame, canonical: str, required: bool = True) -> str | None:
    for candidate in _ALIASES[canonical]:
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(
            f"could not find a column for {canonical!r}; file has {list(df.columns)}. "
            f"Add the real name to _ALIASES in {__file__}."
        )
    return None


def _rename(df: pd.DataFrame, canonicals: list[str], required: bool = True) -> pd.DataFrame:
    mapping = {}
    for c in canonicals:
        found = _pick(df, c, required=required)
        if found:
            mapping[found] = c
    out = df[list(mapping)].rename(columns=mapping)
    for c in canonicals:
        if c not in out.columns:
            out[c] = None
    return out


def load(directory: str | Path, dataset: str = "male-cns:v1.0") -> ConnectomeTables:
    d = Path(directory)
    missing = [f for f in (ANNOTATIONS, WEIGHTS) if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {d}. Download from {BASE_URL}/<file> and place them there "
            "(see README: 'Full-network data')."
        )

    neurons = _rename(
        pd.read_feather(d / ANNOTATIONS), ["bodyId", "type", "superclass", "subclass", "somaSide"]
    )

    nt_path = d / NEUROTRANSMITTERS
    if nt_path.exists():
        nt = _rename(pd.read_feather(nt_path), ["bodyId", "nt"])
        neurons = neurons.merge(nt, on="bodyId", how="left")
    else:
        neurons["nt"] = None
    neurons["nt"] = neurons["nt"].fillna("unclear").astype(str)
    neurons["sign"] = sign_from_nt(neurons["nt"])

    edges = _rename(pd.read_feather(d / WEIGHTS), ["pre", "post", "weight"])
    edges = edges[edges["pre"] != edges["post"]].reset_index(drop=True)

    return ConnectomeTables(neurons=neurons, edges=edges, source="feather", dataset=dataset)


def _print_schema(directory: str) -> None:
    """Resolve the 'unknown column names' question against real files."""
    d = Path(directory)
    for f in (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS):
        p = d / f
        if not p.exists():
            print(f"{f}: ABSENT")
            continue
        head = pd.read_feather(p)
        print(f"\n{f}: {len(head)} rows")
        print(f"  columns: {list(head.columns)}")
        print(head.head(3).to_string(max_colwidth=24))


if __name__ == "__main__":
    _print_schema(sys.argv[1] if len(sys.argv) > 1 else "/data/connectome")
