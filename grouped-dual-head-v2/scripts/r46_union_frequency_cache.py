#!/usr/bin/env python3
"""Verified union adapter for R40 fit-cache components with distinct manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


UNION_SCHEMA = "r46_frequency_cache_union_identity_v1"


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class UnionFrequencyCacheCollection:
    """Expose multiple audited R40 fit collections as one read-only collection."""

    def __init__(
        self,
        r40,
        components: Sequence[Sequence[Path]],
        *,
        expected_subset: str = "fit",
    ):
        if expected_subset != "fit":
            raise ValueError("R46 union is restricted to fit caches")
        if len(components) < 2 or any(not paths for paths in components):
            raise ValueError("R46 union requires at least two nonempty components")
        self.expected_subset = expected_subset
        self.components = []
        self.paths = ()
        self.handles = []
        self.records = []
        self.cache_evidence = {}
        try:
            self.components = [
                r40.FrequencyCacheCollection(paths, expected_subset="fit")
                for paths in components
            ]
            reference = self.components[0]
            for component in self.components[1:]:
                if not np.array_equal(
                    component.frequency_indices, reference.frequency_indices
                ):
                    raise RuntimeError("R46 component frequency indices disagree")
                if not np.allclose(
                    component.frequency_hz, reference.frequency_hz, atol=1.0e-6
                ):
                    raise RuntimeError("R46 component frequencies disagree")
                if int(component.retained) != int(reference.retained):
                    raise RuntimeError("R46 component DCT geometry disagrees")
                if abs(
                    float(component.stored_dt_s) - float(reference.stored_dt_s)
                ) > 1.0e-12:
                    raise RuntimeError("R46 component time axes disagree")

            paths = []
            handles = []
            records = []
            sample_ids = []
            group_ids = []
            families = []
            identity_records = []
            handle_offset = 0
            component_identity = []
            for component_index, component in enumerate(self.components):
                paths.extend(component.paths)
                handles.extend(component.handles)
                component_identity.append(
                    {
                        "component_index": component_index,
                        "selection_sha256": component.selection_sha256,
                        "cache_basenames": [path.name for path in component.paths],
                    }
                )
                for record_position, (file_index, local_row) in enumerate(
                    component.records
                ):
                    global_file_index = handle_offset + int(file_index)
                    records.append((global_file_index, int(local_row)))
                    sample_id = str(component.sample_ids[record_position])
                    group_id = str(component.group_ids[record_position])
                    family = str(component.families[record_position])
                    sample_ids.append(sample_id)
                    group_ids.append(group_id)
                    families.append(family)
                    identity_records.append(
                        {
                            "cache_basename": component.paths[file_index].name,
                            "local_row": int(local_row),
                            "sample_id": sample_id,
                            "group_id": group_id,
                            "family": family,
                        }
                    )
                handle_offset += len(component.handles)

            if len(sample_ids) != len(set(sample_ids)):
                raise RuntimeError("R46 union contains duplicate sample IDs")
            component_group_sets = [set(component.group_ids) for component in self.components]
            for left in range(len(component_group_sets)):
                for right in range(left + 1, len(component_group_sets)):
                    overlap = component_group_sets[left] & component_group_sets[right]
                    if overlap:
                        raise RuntimeError(
                            f"R46 component group overlap: {sorted(overlap)[:3]}"
                        )

            self.paths = tuple(paths)
            self.handles = handles
            self.records = records
            self.sample_ids = tuple(sample_ids)
            self.group_ids = tuple(group_ids)
            self.families = tuple(families)
            self.frequency_indices = reference.frequency_indices.copy()
            self.frequency_hz = reference.frequency_hz.copy()
            self.retained = int(reference.retained)
            self.stored_dt_s = float(reference.stored_dt_s)
            for component in self.components:
                self.cache_evidence.update(component.cache_evidence)
            self.union_identity = {
                "schema": UNION_SCHEMA,
                "components": component_identity,
                "records": identity_records,
            }
            self.selection_sha256 = canonical_sha(self.union_identity)
            self.component_selection_sha256 = tuple(
                str(component.selection_sha256) for component in self.components
            )
        except Exception:
            self.close()
            raise

    @property
    def frequency_count(self) -> int:
        return int(len(self.frequency_indices))

    def close(self) -> None:
        for component in getattr(self, "components", []):
            component.close()
        self.components = []
        self.handles = []

    def __del__(self):
        self.close()
