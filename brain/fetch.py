"""Get connectome tables from whichever source is available, with a parquet cache.

Source selection:
  neuprint  - Cypher API. Works without a token. Default for the 'motor'/'vnc'
              subsets; refuses 'all' (25.6M edges is not an API job).
  feather   - local bulk files in the cache dir. The only route to the full net.
  synthetic - generated, no network, no credentials. Used by the tests.

Also runnable as a module to prefetch during a Docker build:
    python -m brain.fetch --subset motor --out /data/connectome
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from brain.subset import DEFAULT_SUBSET, SUBSETS, apply_subset
from brain.tables import ConnectomeTables

DEFAULT_CACHE = Path("/data/connectome")
SOURCES = ("neuprint", "feather", "synthetic")


def _cache_paths(cache_dir: Path, source: str, subset: str) -> tuple[Path, Path, Path]:
    stem = f"{source}-{subset}"
    return (
        cache_dir / f"{stem}-neurons.parquet",
        cache_dir / f"{stem}-edges.parquet",
        cache_dir / f"{stem}-meta.json",
    )


def _read_cache(cache_dir: Path, source: str, subset: str) -> ConnectomeTables | None:
    n_p, e_p, m_p = _cache_paths(cache_dir, source, subset)
    if not (n_p.exists() and e_p.exists() and m_p.exists()):
        return None
    meta = json.loads(m_p.read_text())
    return ConnectomeTables(
        neurons=pd.read_parquet(n_p),
        edges=pd.read_parquet(e_p),
        source=meta.get("source", source),
        dataset=meta.get("dataset", "unknown"),
    )


def _write_cache(tables: ConnectomeTables, cache_dir: Path, subset: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_p, e_p, m_p = _cache_paths(cache_dir, tables.source, subset)
    tables.neurons.to_parquet(n_p, index=False)
    tables.edges.to_parquet(e_p, index=False)
    m_p.write_text(
        json.dumps(
            {
                "source": tables.source,
                "dataset": tables.dataset,
                "subset": subset,
                "n_neurons": int(len(tables.neurons)),
                "n_edges": int(len(tables.edges)),
            },
            indent=2,
        )
    )


def _fetch(source: str, subset: str, cache_dir: Path, synthetic_size: int) -> ConnectomeTables:
    superclasses = SUBSETS[subset]

    if source == "synthetic":
        from brain.sources import synthetic

        return apply_subset(synthetic.build(n_neurons=synthetic_size), subset)

    if source == "feather":
        from brain.sources import feather

        return apply_subset(feather.load(cache_dir), subset)

    from brain.sources import neuprint

    return neuprint.fetch(superclasses)


def get_tables(
    source: str = "neuprint",
    subset: str = DEFAULT_SUBSET,
    cache_dir: Path | str = DEFAULT_CACHE,
    refresh: bool = False,
    synthetic_size: int = 20_000,
) -> ConnectomeTables:
    if source not in SOURCES:
        raise ValueError(f"unknown connectome source {source!r}, expected one of {SOURCES}")
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}, expected one of {sorted(SUBSETS)}")

    cache_dir = Path(cache_dir)
    if not refresh and source != "synthetic":
        cached = _read_cache(cache_dir, source, subset)
        if cached is not None:
            return cached

    tables = _fetch(source, subset, cache_dir, synthetic_size)
    if source != "synthetic":
        try:
            _write_cache(tables, cache_dir, subset)
        except OSError as exc:  # read-only mount during a build, etc.
            print(f"warning: could not cache connectome to {cache_dir}: {exc}", file=sys.stderr)
    return tables


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prefetch connectome tables into a parquet cache.")
    ap.add_argument("--source", default="neuprint", choices=SOURCES)
    ap.add_argument("--subset", default=DEFAULT_SUBSET, choices=sorted(SUBSETS))
    ap.add_argument("--out", default=str(DEFAULT_CACHE))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    tables = get_tables(args.source, args.subset, args.out, refresh=args.refresh)
    print(tables.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
