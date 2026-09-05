#!/usr/bin/env python3
"""Read one real R40 v2 fit cache through the training data path."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import train_r40_frequency_residual_operator as r40


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    collection = r40.FrequencyCacheCollection([args.cache], expected_subset="fit")
    try:
        dataset = r40.FitFrequencyDataset(collection)
        assert len(dataset) == collection.frequency_count * len(collection.records)
        batch = dataset[len(dataset) // 2]
        (
            base,
            residual,
            static,
            static_scale,
            frequency,
            f0,
            t0,
            frequency_scale,
            target_total,
            frequency_weight,
            family_weight,
        ) = batch
        features = r40.make_features(
            base[None],
            static[None],
            static_scale[None],
            frequency_hz=frequency[None],
            frequency_scale=frequency_scale[None],
            source_f0_hz=f0[None],
            source_t0_s=t0[None],
        )
        values = [
            base,
            residual,
            static,
            static_scale,
            frequency,
            f0,
            t0,
            frequency_scale,
            target_total,
            frequency_weight,
            family_weight,
            features,
        ]
        assert all(torch.isfinite(value).all() for value in values)
        assert features.shape == (
            1,
            r40.INPUT_CHANNELS,
            collection.retained,
            collection.retained,
        )
        handle = collection.handles[0]
        residual_all = np.asarray(handle["residual_dct_norm"][0], dtype=np.float64)
        scales_all = np.asarray(handle["frequency_scale"][0], dtype=np.float64)
        weights_all = r40.rfft_weights(r40.TIME_COUNT)[collection.frequency_indices]
        retained_error = float(
            np.sum(
                weights_all
                * scales_all**2
                * np.sum(residual_all**2, axis=(1, 2, 3), dtype=np.float64)
            )
        )
        unselected_error = float(handle["base_error_square_unselected"][0])
        target_square = float(handle["target_square_total"][0])
        retained_proxy_rel_l2 = math.sqrt(
            (unselected_error + retained_error) / target_square
        )
        print(
            {
                "records": len(collection.records),
                "frequency_count": collection.frequency_count,
                "dataset_length": len(dataset),
                "features": tuple(features.shape),
                "frequency_hz": float(frequency),
                "static_scale_min": float(static_scale.min()),
                "static_scale_max": float(static_scale.max()),
                "target_total": float(target_total),
                "retained_proxy_rel_l2": retained_proxy_rel_l2,
                "cache_base_rel_l2": float(
                    handle.attrs["base_record_rel_l2_mean"]
                ),
            }
        )
    finally:
        collection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
