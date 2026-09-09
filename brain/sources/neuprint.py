"""Fetch male-cns:v1.0 tables from the neuPrint Cypher endpoint.

Deliberately uses plain HTTP rather than `neuprint-python`: the Client class
hard-requires a token, while this endpoint answers anonymously, and the whole
point of the default path is that it works with no credentials. A token is sent
when one is configured, which raises the rate limits.

For `--neuron-subset all` use the Feather source instead - 25.6M edges is not a
reasonable thing to pull over this API.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

from brain.tables import ConnectomeTables, sign_from_nt

DEFAULT_SERVER = os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org")
DEFAULT_DATASET = os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0")

# Batch size for the edge query, in source bodyIds per request.
_EDGE_BATCH = 500
_TIMEOUT = 180


def _token() -> str | None:
    # NEUPRINT_TOKEN is the docker-compose-facing name; the env var
    # neuprint-python itself uses is NEUPRINT_APPLICATION_CREDENTIALS.
    return os.environ.get("NEUPRINT_TOKEN") or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")


def cypher(query: str, server: str = DEFAULT_SERVER, dataset: str = DEFAULT_DATASET,
           retries: int = 3) -> pd.DataFrame:
    headers = {"Content-Type": "application/json"}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{server}/api/custom/custom",
                json={"cypher": query, "dataset": dataset},
                headers=headers,
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            payload = r.json()
            return pd.DataFrame(payload["data"], columns=payload["columns"])
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            time.sleep(2**attempt)
    hint = ""
    if isinstance(last, requests.exceptions.SSLError):
        hint = (
            "\nTLS verification failed. Behind a TLS-inspecting corporate proxy, "
            "Python does not trust the proxy's CA even when curl does. Point "
            "REQUESTS_CA_BUNDLE at a bundle that includes it, e.g. on macOS:\n"
            "  security find-certificate -a -p "
            "/System/Library/Keychains/SystemRootCertificates.keychain > ca.pem\n"
            "  export REQUESTS_CA_BUNDLE=$PWD/ca.pem\n"
            "Or run with --connectome synthetic, which needs no network."
        )
    raise RuntimeError(
        f"neuPrint query failed after {retries} attempts against {server}: {last}{hint}"
    ) from last


def fetch(superclasses: tuple[str, ...] | None, server: str = DEFAULT_SERVER,
          dataset: str = DEFAULT_DATASET) -> ConnectomeTables:
    """Fetch neurons of the given superclasses and the edges among them."""
    if superclasses is None:
        raise ValueError(
            "refusing to pull the full ~176k-neuron / 25.6M-edge network over the "
            "neuPrint API. Use --connectome feather for --neuron-subset all "
            "(see README: 'Full-network data')."
        )

    want = list(superclasses)
    neurons = cypher(
        "MATCH (n:Neuron) WHERE n.superclass IN $sc "
        "RETURN n.bodyId AS bodyId, n.type AS type, n.superclass AS superclass, "
        "n.subclass AS subclass, n.somaSide AS somaSide, "
        "coalesce(n.consensusNt, n.predictedNt, 'unclear') AS nt".replace("$sc", repr(want)),
        server, dataset,
    )
    if neurons.empty:
        raise RuntimeError(f"no neurons returned for superclasses {want} from {dataset}")
    neurons["sign"] = sign_from_nt(neurons["nt"])

    ids = neurons["bodyId"].tolist()
    id_set = repr(ids)
    frames = []
    for i in range(0, len(ids), _EDGE_BATCH):
        batch = repr(ids[i : i + _EDGE_BATCH])
        frames.append(
            cypher(
                f"MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron) "
                f"WHERE a.bodyId IN {batch} AND b.bodyId IN {id_set} "
                f"RETURN a.bodyId AS pre, b.bodyId AS post, w.weight AS weight",
                server, dataset,
            )
        )
    edges = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["pre", "post", "weight"])
    edges = edges[edges["pre"] != edges["post"]].reset_index(drop=True)

    return ConnectomeTables(neurons=neurons, edges=edges, source="neuprint", dataset=dataset)
