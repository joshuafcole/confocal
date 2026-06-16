"""Tests for the hierarchical_merge model_config option on the YAML source."""
from pathlib import Path
from typing import Any, Optional

from pydantic import Field
from confocal import BaseConfig, ConfocalSettingsConfigDict


FILENAME = "hier_config.yaml"


class HierConfig(BaseConfig):
    """Merges every hier_config.yaml from cwd up to the filesystem root."""

    model_config = ConfocalSettingsConfigDict(
        yaml_file=FILENAME,
        hierarchical_merge=True,
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    name: Optional[str] = None
    values: dict[str, Any] = Field(default_factory=dict)


class SingleFileConfig(BaseConfig):
    """Same files, but hierarchical_merge off — only the nearest file is read."""

    model_config = ConfocalSettingsConfigDict(
        yaml_file=FILENAME,
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    name: Optional[str] = None
    values: dict[str, Any] = Field(default_factory=dict)


class EnvOverrideConfig(BaseConfig):
    """hierarchical_merge on, but an explicit override path must win with no merge."""

    model_config = ConfocalSettingsConfigDict(
        yaml_file=FILENAME,
        hierarchical_merge=True,
        env_file_override="HIER_CONFIG_PATH",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    name: Optional[str] = None
    values: dict[str, Any] = Field(default_factory=dict)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _two_level_tree(tmp_path):
    """root has {a,b}; child (cwd) overrides b and adds c. Returns (workdir, root, child)."""
    root = _write(
        tmp_path / FILENAME,
        "name: root\nvalues:\n  a: 1\n  b: 1\n",
    )
    workdir = tmp_path / "proj"
    child = _write(
        workdir / FILENAME,
        "name: child\nvalues:\n  b: 2\n  c: 3\n",
    )
    return workdir, root, child


class TestHierarchicalMerge:
    def test_nearest_wins_field_level_merge(self, tmp_path, monkeypatch):
        workdir, _root, child = _two_level_tree(tmp_path)
        monkeypatch.chdir(workdir)

        cfg = HierConfig()

        assert cfg.name == "child"  # nearest file wins on scalar conflict
        # Field-level deep merge: a from root, b overridden by child, c from child.
        assert cfg.values == {"a": 1, "b": 2, "c": 3}

    def test_resolved_path_is_nearest_file(self, tmp_path, monkeypatch):
        workdir, _root, child = _two_level_tree(tmp_path)
        monkeypatch.chdir(workdir)

        cfg = HierConfig()

        assert getattr(cfg, "_resolved_config_file", None) == str(child)

    def test_single_ancestor_only(self, tmp_path, monkeypatch):
        _write(tmp_path / FILENAME, "name: root\nvalues:\n  a: 1\n")
        workdir = tmp_path / "proj" / "deep"
        workdir.mkdir(parents=True)
        monkeypatch.chdir(workdir)

        cfg = HierConfig()

        assert cfg.name == "root"
        assert cfg.values == {"a": 1}


class TestHierarchicalProvenance:
    def test_per_file_provenance_recorded(self, tmp_path, monkeypatch):
        workdir, root, child = _two_level_tree(tmp_path)
        monkeypatch.chdir(workdir)

        cfg = HierConfig()
        prov = cfg.config_provenance

        # Nearest file is listed first (the winner) for a conflicting scalar.
        assert prov["name"][0] == (str(child), "child")
        # A field only the root file sets is attributed to the root file.
        assert prov["values.a"] == [(str(root), 1)]
        # A field both set: child wins, root retained in the chain.
        assert prov["values.b"][0] == (str(child), 2)
        assert (str(root), 1) in prov["values.b"]


class TestSingleFileRegression:
    def test_only_nearest_read_when_flag_off(self, tmp_path, monkeypatch):
        workdir, _root, _child = _two_level_tree(tmp_path)
        monkeypatch.chdir(workdir)

        cfg = SingleFileConfig()

        # Flag off: only the nearest file — no 'a' from the root file.
        assert cfg.name == "child"
        assert cfg.values == {"b": 2, "c": 3}


class TestProfileOverlayPerFile:
    """A profile overrides only within its own file; nearest file still wins overall."""

    def test_ancestor_profile_does_not_override_nearer_file(self, tmp_path, monkeypatch):
        # Ancestor file: active profile sets `name`. Nearest file: sets `name` at top level.
        _write(
            tmp_path / FILENAME,
            "active_profile: dev\nprofile:\n  dev:\n    name: from_ancestor_profile\n",
        )
        workdir = tmp_path / "proj"
        _write(workdir / FILENAME, "name: from_child\n")
        monkeypatch.chdir(workdir)

        cfg = HierConfig()

        # Nearest file wins; the ancestor's profile does not reach across files.
        assert cfg.name == "from_child"
        assert cfg.active_profile == "dev"


class TestDirectoryNamedLikeConfig:
    """A directory whose name matches the config file must be skipped, not read."""

    def test_directory_match_does_not_crash(self, tmp_path, monkeypatch):
        # find_in matches on existence, so an ancestor *directory* named like the config
        # would be returned; reading it must be skipped rather than raising IsADirectoryError.
        (tmp_path / FILENAME).mkdir()
        workdir = tmp_path / "proj"
        _write(workdir / FILENAME, "name: from_child\n")
        monkeypatch.chdir(workdir)

        cfg = HierConfig()  # must not raise

        assert cfg.name == "from_child"


class TestMissingAbsolutePath:
    """A configured-but-missing absolute file path contributes nothing, never crashes."""

    def test_missing_absolute_file_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        class MissingFileConfig(BaseConfig):
            model_config = ConfocalSettingsConfigDict(
                yaml_file=str(tmp_path / "does_not_exist.yaml"),
                extra="ignore",
                nested_model_default_partial_update=True,
            )
            name: Optional[str] = None

        cfg = MissingFileConfig()  # must not raise FileNotFoundError
        assert cfg.name is None


class TestEnvOverrideDisablesMerge:
    def test_override_path_wins_no_merge(self, tmp_path, monkeypatch):
        workdir, _root, child = _two_level_tree(tmp_path)
        monkeypatch.chdir(workdir)
        monkeypatch.setenv("HIER_CONFIG_PATH", str(child))

        cfg = EnvOverrideConfig()

        # Only the override file is loaded; the root file is not merged in.
        assert cfg.name == "child"
        assert cfg.values == {"b": 2, "c": 3}
