#!/usr/bin/env python
"""Build or inspect the immutable three-family V3 index manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import (
    build_manifest,
    validate_expected_counts,
    write_manifest_atomic,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = V3Config.from_yaml(args.config)
    manifest = build_manifest(config.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": config.data.expected_train_records,
            "validation": config.data.expected_validation_records,
        },
    )
    summary = {
        "allowed_medium_types": manifest.allowed_medium_types,
        "excluded_medium_types": manifest.excluded_medium_types,
        "counts_after": manifest.counts_after,
        "digest": manifest.digest,
    }
    print(json.dumps(summary, sort_keys=True))
    if not args.dry_run:
        destination = args.output or config.data.manifest_json
        print(write_manifest_atomic(manifest, destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

