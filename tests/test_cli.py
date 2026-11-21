import pytest

from model_evaluator import cli


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("evaluate", "compare", "batch", "chat", "logs", "models"):
        assert command in out


def test_models_command(capsys, no_openrouter_key):
    assert cli.main(["models", "--providers", "openai"]) == 0
    out = capsys.readouterr().out
    assert "gpt-6-luna" in out and "claude" not in out


def test_batch_command_passes_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "evaluate_batch", lambda cases, **kw: seen.update(cases=cases, **kw))
    cli.main(["batch", "datasets/example.json", "--models", "a", "b", "--scorer", "none", "-y"])
    assert seen["cases"] == "datasets/example.json"
    assert seen["models"] == ["a", "b"]
    assert seen["scorer"] is None
    assert seen["confirm"] is False
