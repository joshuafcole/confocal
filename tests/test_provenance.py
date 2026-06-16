"""Tests for provenance rendering (show_provenance)."""
from typing import Dict

from pydantic import BaseModel, Field, SecretStr

from confocal.config import show_provenance, _abbreviate_home


class TestDictRecursion:
    """A dict of models (e.g. connections) is recursed into, per-field, with secrets masked."""

    def test_dict_of_models_attributed_per_field_with_masked_secret(self, capsys):
        class _Conn(BaseModel):
            account: str = ""
            password: SecretStr = SecretStr("")

        class _CfgConns(BaseModel):
            connections: Dict[str, _Conn] = {}

        cfg = _CfgConns(connections={"sf": _Conn(account="ACC", password=SecretStr("hunter2"))})
        prov = {
            "connections.sf.account": [("/home/.snowflake/config.toml", "ACC")],
            "connections.sf.password": [("/home/.snowflake/config.toml", "hunter2")],
        }

        show_provenance(cfg, prov, verbose=True)
        out = capsys.readouterr().out

        # account is attributed to its file (not a single "connections = {...} default" blob)
        assert "ACC" in out
        assert "/home/.snowflake/config.toml" in out
        # password is masked — the plaintext from provenance must not leak
        assert "hunter2" not in out
        assert "********" in out


class TestAbbreviateHome:
    def test_collapses_home_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _abbreviate_home(str(tmp_path / ".snowflake" / "config.toml")) == "~/.snowflake/config.toml"

    def test_leaves_non_home_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _abbreviate_home("/etc/raiconfig.yaml") == "/etc/raiconfig.yaml"

    def test_does_not_mangle_sibling_sharing_home_prefix(self, monkeypatch):
        monkeypatch.setenv("HOME", "/Users/me")
        # /Users/meadow shares the "/Users/me" prefix but is not under HOME.
        assert _abbreviate_home("/Users/meadow/raiconfig.yaml") == "/Users/meadow/raiconfig.yaml"
        assert _abbreviate_home("/Users/me") == "~"
        assert _abbreviate_home("/Users/me/x.yaml") == "~/x.yaml"


class _Conn(BaseModel):
    account: str = ""
    warehouse: str = ""


class _Cfg(BaseModel):
    name: str = ""
    conn: _Conn = _Conn()


class TestShowProvenanceNested:
    def test_nested_field_attributed_to_source_not_default(self, capsys):
        """A nested leaf must be looked up by its dotted full path, not its leaf name."""
        cfg = _Cfg(name="proj", conn=_Conn(account="ACC", warehouse="WH"))
        prov = {
            "name": [("/proj/raiconfig.yaml", "proj")],
            "conn.account": [("/home/.snowflake/config.toml", "ACC")],
            "conn.warehouse": [("/proj/raiconfig.yaml", "WH")],
        }

        show_provenance(cfg, prov, verbose=True)
        out = capsys.readouterr().out

        # The nested account field is attributed to the snowflake file, not "default".
        assert "/home/.snowflake/config.toml" in out
        assert "default" not in out

    def test_top_level_field_still_attributed(self, capsys):
        cfg = _Cfg(name="proj")
        prov = {"name": [("/proj/raiconfig.yaml", "proj")]}

        show_provenance(cfg, prov, verbose=True)
        out = capsys.readouterr().out

        assert "/proj/raiconfig.yaml" in out

    def test_aliased_field_attributed_via_alias_key(self, capsys):
        """A field with an alias (e.g. `schema` for `schema_`) is keyed in provenance by
        the alias, so lookup must fall back to the alias when the field name misses."""

        class _Deploy(BaseModel):
            schema_: str = Field(default="", alias="schema")

        class _Cfg2(BaseModel):
            deployment: _Deploy = _Deploy()

        cfg = _Cfg2(deployment=_Deploy(schema="DB.S"))
        prov = {"deployment.schema": [("/proj/raiconfig.yaml", "DB.S")]}  # keyed by alias

        show_provenance(cfg, prov, verbose=True)
        out = capsys.readouterr().out

        assert "/proj/raiconfig.yaml" in out
        assert "default" not in out


class TestSourceAnnotation:
    """Default view: every field shows its source (primary dimmed, others highlighted)."""

    def test_every_field_shows_its_source(self, capsys):
        cfg = _Cfg(name="proj", conn=_Conn(account="ACC"))
        prov = {
            "name": [("/proj/raiconfig.yaml", "proj")],            # from primary
            "conn.account": [("/home/.snowflake/config.toml", "ACC")],  # from elsewhere
        }

        show_provenance(cfg, prov, verbose=False, primary_source="/proj/raiconfig.yaml")
        out = capsys.readouterr().out

        # Both sources are shown — the primary is just de-emphasized, not hidden.
        assert "/home/.snowflake/config.toml" in out
        assert "/proj/raiconfig.yaml" in out

    def test_verbose_shows_full_chain_including_primary(self, capsys):
        cfg = _Cfg(name="proj")
        prov = {"name": [("/proj/raiconfig.yaml", "proj")]}

        show_provenance(cfg, prov, verbose=True, primary_source="/proj/raiconfig.yaml")
        out = capsys.readouterr().out

        assert "/proj/raiconfig.yaml" in out
