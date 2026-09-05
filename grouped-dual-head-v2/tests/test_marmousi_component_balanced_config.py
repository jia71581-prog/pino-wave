from pathlib import Path

from fno_acoustic.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_marmousi_component_balanced_short_config_contract() -> None:
    cfg = load_config(
        ROOT / "configs/pino_elastic_vti_marmousi_component_balanced_short.yaml"
    )

    assert cfg["train"]["init_checkpoint"] == (
        "artifacts/elastic_vti_pino_marmousi/checkpoints/best.pt"
    )
    assert cfg["data"]["split_manifest"] == (
        "artifacts/elastic_vti_pino_marmousi/splits.json"
    )
    assert cfg["normalization"]["stats_path"] == (
        "artifacts/elastic_vti_pino_marmousi/normalization_stats.json"
    )
    assert cfg["normalization"]["reuse_stats"] is True
    assert cfg["loss"]["component_relative_l2_weights"] == [1.0, 1.0]
    assert cfg["train"]["epochs"] == 5
    assert cfg["train"]["max_train_batches"] == 64
    assert cfg["train"]["max_val_batches"] == 8
    assert cfg["train"]["batch_size"] == 1
    assert cfg["train"]["learning_rate"] == 2.0e-5
    assert cfg["train"]["weight_decay"] == 1.0e-5
    assert cfg["train"]["grad_clip"] == 0.5
    assert cfg["train"]["checkpoint_dir"] == (
        "artifacts/elastic_vti_pino_marmousi_component_balanced_short/checkpoints"
    )
