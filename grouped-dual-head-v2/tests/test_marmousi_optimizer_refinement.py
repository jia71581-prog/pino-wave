from pathlib import Path
import sys

from fno_acoustic.config import load_config


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from select_marmousi_optimizer_candidate import select_candidate


BASELINE = "artifacts/elastic_vti_pino_marmousi/checkpoints/best.pt"
SPLIT = "artifacts/elastic_vti_pino_marmousi/splits.json"
STATS = "artifacts/elastic_vti_pino_marmousi/normalization_stats.json"


def test_optimizer_candidate_configs_control_all_non_lr_variables() -> None:
    candidates = {
        "lr2e5": 2.0e-5,
        "lr5e5": 5.0e-5,
        "lr1e4": 1.0e-4,
    }
    signatures = []
    for tag, learning_rate in candidates.items():
        cfg = load_config(ROOT / f"configs/pino_elastic_vti_marmousi_opt_{tag}.yaml")
        assert cfg["train"]["init_checkpoint"] == BASELINE
        assert cfg["data"]["split_manifest"] == SPLIT
        assert cfg["normalization"]["stats_path"] == STATS
        assert cfg["normalization"]["reuse_stats"] is True
        assert cfg["train"]["learning_rate"] == learning_rate
        assert cfg["train"]["weight_decay"] == 1.0e-5
        assert cfg["train"]["grad_clip"] == 0.5
        assert cfg["train"]["epochs"] == 3
        assert cfg["train"]["max_train_batches"] == 64
        assert cfg["train"]["max_val_batches"] == 8
        assert cfg["train"]["batch_size"] == 1
        assert cfg["train"]["scheduler"] == "cosine"
        signatures.append(
            (
                cfg["model"],
                cfg["loss"],
                cfg["sampling"],
                cfg["data"]["path_glob"],
            )
        )

    assert signatures[1:] == signatures[:-1]


def test_selector_rejects_all_candidates_above_threshold() -> None:
    rows = [
        {
            "tag": "lr2e5",
            "learning_rate": 2.0e-5,
            "validation_relative_l2": 0.105,
        }
    ]

    result = select_candidate(rows, threshold=0.1045, tie_tolerance=1.0e-4)

    assert result["accepted"] is False
    assert result["selected"] is None


def test_selector_uses_lower_lr_inside_tie_tolerance() -> None:
    rows = [
        {
            "tag": "lr2e5",
            "learning_rate": 2.0e-5,
            "validation_relative_l2": 0.10420,
        },
        {
            "tag": "lr5e5",
            "learning_rate": 5.0e-5,
            "validation_relative_l2": 0.10415,
        },
        {
            "tag": "lr1e4",
            "learning_rate": 1.0e-4,
            "validation_relative_l2": 0.10410,
        },
    ]

    result = select_candidate(rows, threshold=0.1045, tie_tolerance=1.0e-4)

    assert result["accepted"] is True
    assert result["selected"]["tag"] == "lr2e5"


def test_selector_requires_finite_metrics() -> None:
    rows = [
        {
            "tag": "lr2e5",
            "learning_rate": 2.0e-5,
            "validation_relative_l2": float("nan"),
        }
    ]

    result = select_candidate(rows, threshold=0.1045, tie_tolerance=1.0e-4)

    assert result["accepted"] is False
