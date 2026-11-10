import argparse
import json
from pathlib import Path

import numpy as np


ALIGNMENT_FIELDS = ("candidate_ids", "labels", "image_ids", "scanners")


def combine_caches(paths):
    if len(paths) < 2:
        raise ValueError("At least two embedding caches are required")
    caches = [np.load(path) for path in paths]
    reference = caches[0]
    for path, cache in zip(paths[1:], caches[1:]):
        mismatches = [
            field for field in ALIGNMENT_FIELDS if not np.array_equal(reference[field], cache[field])
        ]
        if str(reference["model_id"][0]) != str(cache["model_id"][0]):
            mismatches.append("model_id")
        if str(reference["pooling"][0]) != str(cache["pooling"][0]):
            mismatches.append("pooling")
        if mismatches:
            raise ValueError(f"Cache {path} does not align on: {', '.join(mismatches)}")

    features = [cache["features"].astype(np.float32) for cache in caches]
    if any(array.ndim != 2 for array in features):
        raise ValueError("Every feature array must have shape [candidates, features]")
    scales = [str(cache["scale"][0]) for cache in caches]
    return {
        "features": np.concatenate(features, axis=1),
        **{field: reference[field] for field in ALIGNMENT_FIELDS},
        "scale": np.asarray(["+".join(scales)]),
        "model_id": reference["model_id"],
        "model_key": reference["model_key"],
        "pooling": np.asarray([f"concatenated_{reference['pooling'][0]}"]),
    }, scales, [array.shape[1] for array in features]


def main():
    parser = argparse.ArgumentParser(description="Concatenate aligned multi-scale embedding caches.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    combined, scales, dimensions = combine_caches(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **combined)
    metadata = {
        "model_id": str(combined["model_id"][0]),
        "scales": scales,
        "source_caches": [str(path) for path in args.inputs],
        "source_dimensions": dimensions,
        "feature_dimension": int(combined["features"].shape[1]),
        "candidates": int(combined["features"].shape[0]),
        "fusion": "feature_concatenation",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
