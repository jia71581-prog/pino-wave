import pytest

import json

from scripts.evaluate_saved_time_v4_full_support import (
    evaluation_checkpoint_identity,
    final_time_blocks,
)


def test_final_time_blocks_cover_all_401_indices_once():
    blocks = final_time_blocks(stored_time_count=401, block_size=16)

    assert blocks[0] == (0, 16)
    assert blocks[-1] == (400, 401)
    assert tuple(index for start, stop in blocks for index in range(start, stop)) == tuple(
        range(401)
    )


@pytest.mark.parametrize("count,block", [(0, 16), (401, 0)])
def test_final_time_blocks_reject_nonpositive_counts(count, block):
    with pytest.raises(ValueError, match="positive"):
        final_time_blocks(stored_time_count=count, block_size=block)


def test_evaluator_can_bind_a_refinement_checkpoint_identity(tmp_path):
    default = tmp_path / "parent" / "run" / "run_identity.json"
    default.parent.mkdir(parents=True)
    default.write_text(json.dumps({"run_digest": "parent"}))
    refinement = tmp_path / "asam" / "run_identity.json"
    refinement.parent.mkdir(parents=True)
    refinement.write_text(json.dumps({"run_digest": "refined"}))
    config = {"artifact_dir": str(tmp_path / "parent")}

    path, identity = evaluation_checkpoint_identity(
        config, checkpoint_identity=refinement
    )

    assert path == refinement.resolve()
    assert identity["run_digest"] == "refined"
