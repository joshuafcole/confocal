"""Tests for find_upwards / find_all_upwards directory walking."""
from pathlib import Path

from confocal import find_upwards, find_all_upwards


NEEDLE = Path("raiconfig.yaml")


def _touch(path: Path) -> Path:
    """Create an empty file (and any missing parents) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


class TestFindAllUpwards:
    def test_no_matches_returns_empty_list(self, tmp_path, monkeypatch):
        workdir = tmp_path / "a" / "b" / "c"
        workdir.mkdir(parents=True)
        monkeypatch.chdir(workdir)

        assert find_all_upwards(NEEDLE) == []

    def test_single_match_in_cwd(self, tmp_path, monkeypatch):
        workdir = tmp_path / "proj"
        workdir.mkdir()
        f = _touch(workdir / NEEDLE)
        monkeypatch.chdir(workdir)

        assert find_all_upwards(NEEDLE) == [f]

    def test_single_match_in_ancestor(self, tmp_path, monkeypatch):
        f = _touch(tmp_path / NEEDLE)
        workdir = tmp_path / "proj" / "nested"
        workdir.mkdir(parents=True)
        monkeypatch.chdir(workdir)

        assert find_all_upwards(NEEDLE) == [f]

    def test_multiple_matches_ordered_nearest_first(self, tmp_path, monkeypatch):
        # Files at three levels: tmp_path, tmp_path/proj, tmp_path/proj/nested (cwd).
        root_f = _touch(tmp_path / NEEDLE)
        proj_f = _touch(tmp_path / "proj" / NEEDLE)
        workdir = tmp_path / "proj" / "nested"
        nested_f = _touch(workdir / NEEDLE)
        monkeypatch.chdir(workdir)

        result = find_all_upwards(NEEDLE)

        # Nearest-first: cwd, then parent, then grandparent.
        nearest_three = result[:3]
        assert nearest_three == [nested_f, proj_f, root_f]

    def test_absolute_needle_returned_as_is(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        absolute = tmp_path / "somewhere" / NEEDLE

        # Mirrors find_upwards: absolute paths are returned without an existence check.
        assert find_all_upwards(absolute) == [absolute]


class TestFindUpwardsRegression:
    """find_upwards must still return only the nearest match (unchanged behavior)."""

    def test_returns_nearest_only(self, tmp_path, monkeypatch):
        _touch(tmp_path / NEEDLE)
        workdir = tmp_path / "proj"
        nearest = _touch(workdir / NEEDLE)
        monkeypatch.chdir(workdir)

        assert find_upwards(NEEDLE) == nearest
