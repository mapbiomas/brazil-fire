"""Testes do runner hv_local: backoff e execucao com fakes (M6/M7)."""

import tempfile

from mapbiomas_cog.engine.runners.hv_local import HvLocalRunner, is_retryable, with_backoff
from tests.fakes import FakeFS, install_fake_ee, make_task


def test_is_retryable():
    assert is_retryable(Exception("HTTP 429 Too Many Requests"))
    assert is_retryable(Exception("quota exceeded"))
    assert not is_retryable(Exception("invalid argument"))


def test_with_backoff_retries_then_ok():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("429 rate limit")
        return "ok"

    delays = []
    assert with_backoff(fn, retries=5, base_delay=1.0, sleep=delays.append) == "ok"
    assert calls["n"] == 3
    assert delays == [1.0, 2.0]


def test_with_backoff_gives_up():
    def fn():
        raise Exception("429")

    try:
        with_backoff(fn, retries=2, base_delay=1.0, sleep=lambda _d: None)
        raise AssertionError("deveria ter falhado")
    except Exception as exc:  # noqa: BLE001
        assert "429" in str(exc)


def test_hv_run_writes_tiles_and_is_idempotent():
    task = make_task()
    with install_fake_ee():
        fs = FakeFS()
        runner = HvLocalRunner(state_dir=tempfile.mkdtemp(), fs=fs, max_workers=2)
        result = runner.run(task)
        assert result.done == task.grid.n_tiles
        assert len([p for p in fs.files if p.endswith(".tif")]) == task.grid.n_tiles

        again = runner.run(task)
        assert again.submitted == 0


def test_hv_sync_reports_state():
    task = make_task()
    with install_fake_ee():
        runner = HvLocalRunner(state_dir=tempfile.mkdtemp(), fs=FakeFS())
        runner.run(task)
        result = runner.sync(task)
        assert result.done == task.grid.n_tiles
        assert result.ok
