import argparse
import json
import os
import sys
from pathlib import Path

def _normalize_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(key)
        if value is None:
            os.environ[key] = "1"
            continue
        try:
            if int(value) <= 0:
                os.environ[key] = "1"
        except ValueError:
            os.environ[key] = "1"


_normalize_thread_env()

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agree.mixed_data import DATASET_SPECS, build_mixed_graph_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument(
        "--raw_data_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "mixed_raw"),
        help="Root directory containing the original mixed-type datasets.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "graph_data"),
        help="Root directory where exported graph datasets will be written.",
    )
    parser.add_argument("--construct_name", type=str, default="IPD", choices=["IPD", "IP", "I"])
    parser.add_argument("--adj_name", type=str, default="dis", choices=["dis", "graph"])
    args = parser.parse_args()

    features, adjacency, labels = build_mixed_graph_dataset(
        args.dataset,
        raw_data_root=args.raw_data_root,
        construct_name=args.construct_name,
        adj_name=args.adj_name,
    )

    output_dir = Path(args.output_root) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{args.dataset}_feat.npy", features)
    np.save(output_dir / f"{args.dataset}_adj.npy", adjacency)
    np.save(output_dir / f"{args.dataset}_label.npy", labels)

    metadata = {
        "dataset": args.dataset,
        "construct_name": args.construct_name,
        "adj_name": args.adj_name,
        "num_nodes": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_classes": int(len(np.unique(labels))),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
