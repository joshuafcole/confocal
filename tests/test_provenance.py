"""Tests for provenance rendering (show_provenance)."""
from pydantic import BaseModel

from confocal.config import show_provenance, _abbreviate_home


class TestAbbreviateHome:
    def test_collapses_home_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _abbreviate_home(str(tmp_path / ".snowflake" / "config.toml")) == "~/.snowflake/config.toml"

    def test_leaves_non_home_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _abbreviate_home("/etc/raiconfig.yaml") == "/etc/raiconfig.yaml"


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


class TestOptionARendering:
    """Default view: annotate only fields whose source differs from the primary file."""

    def test_primary_source_not_annotated_others_are(self, capsys):
        cfg = _Cfg(name="proj", conn=_Conn(account="ACC"))
        prov = {
            "name": [("/proj/raiconfig.yaml", "proj")],            # from primary
            "conn.account": [("/home/.snowflake/config.toml", "ACC")],  # from elsewhere
        }

        show_provenance(cfg, prov, verbose=False, primary_source="/proj/raiconfig.yaml")
        out = capsys.readouterr().out

        # The non-primary source is annotated; the primary path is suppressed.
        assert "/home/.snowflake/config.toml" in out
        assert "/proj/raiconfig.yaml" not in out

    def test_verbose_shows_full_chain_including_primary(self, capsys):
        cfg = _Cfg(name="proj")
        prov = {"name": [("/proj/raiconfig.yaml", "proj")]}

        show_provenance(cfg, prov, verbose=True, primary_source="/proj/raiconfig.yaml")
        out = capsys.readouterr().out

        assert "/proj/raiconfig.yaml" in out
