import pytest

from confocal.sources import EnvVarTemplateMixin


@pytest.fixture
def render():
    return EnvVarTemplateMixin._render_env_vars.__get__(object.__new__(EnvVarTemplateMixin))


def test_dollar_brace_set(render, monkeypatch):
    monkeypatch.setenv("CF_VAR", "abc")
    assert render("k: ${CF_VAR}") == "k: abc"


def test_dollar_brace_unset_raises(render, monkeypatch):
    monkeypatch.delenv("CF_VAR", raising=False)
    with pytest.raises(ValueError, match="CF_VAR"):
        render("k: ${CF_VAR}")


def test_dollar_brace_default(render, monkeypatch):
    monkeypatch.delenv("CF_VAR", raising=False)
    assert render("k: ${CF_VAR:-fallback}") == "k: fallback"
    assert render("k: ${CF_VAR:-}") == "k: "


def test_dollar_brace_default_ignored_when_set(render, monkeypatch):
    monkeypatch.setenv("CF_VAR", "real")
    assert render("k: ${CF_VAR:-fallback}") == "k: real"


def test_escape(render, monkeypatch):
    monkeypatch.delenv("CF_VAR", raising=False)
    assert render("k: $${CF_VAR}") == "k: ${CF_VAR}"


def test_commented_line_skipped(render, monkeypatch):
    monkeypatch.delenv("CF_VAR", raising=False)
    assert render("# k: ${CF_VAR}") == "# k: ${CF_VAR}"


def test_jinja_still_works_and_mixes(render, monkeypatch):
    monkeypatch.setenv("CF_A", "a")
    monkeypatch.setenv("CF_B", "b")
    assert render("k: \"{{ env_var('CF_A') }}-${CF_B}\"") == 'k: "a-b"'


def test_non_matching_dollar_left_alone(render):
    assert render("k: $FOO ${1BAD} $5") == "k: $FOO ${1BAD} $5"
