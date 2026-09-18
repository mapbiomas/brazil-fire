"""Testes do runner ee_batch_tiled com EE fake (M7, sem rede)."""

import tempfile

from mapbiomas_cog.engine.runners.ee_batch import EeBatchTiledRunner
from tests.fakes import install_fake_ee, make_task


def test_run_submits_all_tiles():
    task = make_task()
    with install_fake_ee() as ee:
        runner = EeBatchTiledRunner(state_dir=tempfile.mkdtemp())
        result = runner.run(task)
        assert result.ok
        assert result.submitted == task.grid.n_tiles
        assert len(ee.states) == task.grid.n_tiles


def test_run_is_idempotent():
    task = make_task()
    with install_fake_ee():
        runner = EeBatchTiledRunner(state_dir=tempfile.mkdtemp())
        runner.run(task)
        again = runner.run(task)
        assert again.submitted == 0


def test_sync_and_retry_only_failed():
    task = make_task()
    with install_fake_ee() as ee:
        runner = EeBatchTiledRunner(state_dir=tempfile.mkdtemp())
        runner.run(task)
        ids = list(ee.states)
        ee.set_states({i: "COMPLETED" for i in ids})
        ee.set_states({ids[0]: "FAILED"})

        synced = runner.sync(task)
        assert synced.failed == 1
        assert not synced.ok

        retried = runner.retry(task)
        assert retried.submitted == 1
