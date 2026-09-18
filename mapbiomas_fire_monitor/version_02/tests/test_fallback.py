"""Testes do fallback de runner no Engine (M6)."""

from mapbiomas_cog.engine.base import Engine, RunResult, RunnerUnavailable, register_runner
from tests.fakes import make_task


@register_runner("fail_x")
class _FailRunner:
    def __init__(self, **_kwargs):
        pass

    def run(self, task):
        raise RunnerUnavailable("computePixels indisponivel")


@register_runner("ok_x")
class _OkRunner:
    def __init__(self, **_kwargs):
        pass

    def run(self, task):
        return RunResult(unit=task.unit_key, detail="fallback-ok")


def test_engine_falls_back():
    engine = Engine(runner="fail_x", fallback="ok_x")
    result = engine.run(make_task())
    assert result.detail == "fallback-ok"
    assert engine.runner_name == "ok_x"


def test_engine_without_fallback_raises():
    engine = Engine(runner="fail_x")
    try:
        engine.run(make_task())
        raise AssertionError("deveria levantar RunnerUnavailable")
    except RunnerUnavailable:
        pass
