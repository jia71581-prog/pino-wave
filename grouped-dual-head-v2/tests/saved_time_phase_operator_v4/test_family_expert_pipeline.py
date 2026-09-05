from pathlib import Path


PIPELINE = Path("scripts/run_remote_family_expert_pipeline.sh")


def test_pipeline_has_durable_terminal_and_nonzero_gate_handling():
    text = PIPELINE.read_text()

    assert "trap finalize EXIT" in text
    assert "flock -n 9" in text
    assert "pipeline_terminal.superseded" in text
    assert "set +e" in text and "gate_rc=$?" in text
    assert "pipeline_terminal.json" in text
    assert "gate_rc -eq 2" in text


def test_pipeline_uses_fresh_process_for_each_descending_memory_probe():
    text = PIPELINE.read_text()

    assert "TRAINING_TIME_BLOCK=${TRAINING_TIME_BLOCK:-2}" in text
    assert "ADAMW_IMPLEMENTATION=${ADAMW_IMPLEMENTATION:-fused}" in text
    assert 'optimizer["training_time_block"] = time_block' in text
    assert 'optimizer["adamw_implementation"] = implementation' in text
    assert "for microbatch in 12 6 4 3" in text
    assert "torchrun" in text.lower()
    assert "--smoke-updates 2" in text
    assert 'family_experts["stage_epoch_offset"] = max(1, dense_epoch - 1)' in text
    assert 'probe_dir}.superseded.' in text
    assert "select_saved_time_physical_microbatch.py" in text


def test_pipeline_runs_overfit_then_pilot_and_only_promotes_a_passing_gate():
    text = PIPELINE.read_text()

    assert "--overfit-updates 30" in text
    assert "--microbatch-records 1" in text
    assert "--pilot" in text
    assert "gate_saved_time_family_expert_candidate.py" in text
    assert "prepare_saved_time_long_continuation.py" in text
    assert text.index("gate_saved_time_family_expert_candidate.py") < text.index(
        "prepare_saved_time_long_continuation.py"
    )
    assert "--epochs 40" in text


def test_pipeline_requires_fresh_predecessor_terminal():
    text = PIPELINE.read_text()

    assert "PREDECESSOR_FRESHNESS" in text
    assert "-nt \"$PREDECESSOR_FRESHNESS\"" in text
    assert "PREDECESSOR_GATE" in text
    assert "PREDECESSOR_GLOBAL_GATE" in text
    assert "skipped_predecessor_promoted" in text
