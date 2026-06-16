#!/usr/bin/env python3

from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import Any

# tomllib is only available in Python 3.11+, use tomli for older versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "tomli is required for Python < 3.11. "
            "Install it with: pip install tomli"
        )

from pydantic_settings.sources import DEFAULT_PATH, PathType
from pydantic_settings import BaseSettings
from pydantic_settings.sources import (
    TomlConfigSettingsSource,
    YamlConfigSettingsSource,
)
import yaml

from .utils import find_upwards, overlay_profile


# Maps settings class → the resolved config file path found during source evaluation.
# Consumed by BaseConfig.model_post_init to populate _resolved_config_file.
_resolved_paths: dict[type, Path] = {}


# ------------------------------------------------------------------------------
# Config Source Mixins
# ------------------------------------------------------------------------------


class AncestorConfigMixin:
    """Mixin for finding config files in ancestor directories."""

    def __init__(self, *args, case_sensitive: bool = False, **kwargs):
        self._case_sensitive = case_sensitive
        super().__init__(*args, **kwargs)

    def _read_files(self, files: PathType | None, deep_merge: bool = False) -> dict[str, Any]:
        """Read config files, searching upwards if not found directly.

        When ``self._hierarchical`` is set, each (non-absolute) file pattern is matched
        against *every* ancestor directory and the results are deep-merged with the file
        nearest the current directory winning. Otherwise only the nearest match is read,
        which is the original single-file behavior.
        """
        if files is None:
            return {}
        if isinstance(files, (str, os.PathLike)):
            files = [files]

        from .utils import deep_merge as merge_dicts, find_all_upwards

        hierarchical = getattr(self, "_hierarchical", False)

        vars: dict[str, Any] = {}
        # (path_str, data) for every ancestor file read in hierarchical mode, nearest-first.
        # Used to attach per-file provenance so config:explain can show which file set
        # each field.
        hier_per_file: list[tuple[str, dict[str, Any]]] = []
        for file in files:
            file_path = Path(file).expanduser()

            if hierarchical and not file_path.is_absolute():
                # All ancestor matches, nearest-first. Keep only files — find_in matches on
                # existence and could return a directory. Apply farthest-first so the nearest
                # file wins on conflict; record the nearest as the resolved path.
                matches = [m for m in find_all_upwards(file_path, self._case_sensitive) if m.is_file()]
                if matches and self.settings_cls not in _resolved_paths:  # type: ignore[attr-defined]
                    _resolved_paths[self.settings_cls] = matches[0]  # type: ignore[attr-defined]
                per_file: list[tuple[str, dict[str, Any]]] = []
                for match in matches:
                    data = self._read_file(match)  # type: ignore[attr-defined]
                    if data:
                        per_file.append((str(match), data))
                for _path, data in reversed(per_file):
                    vars = merge_dicts(vars, data)
                hier_per_file.extend(per_file)
                continue

            # (continues below for the non-hierarchical / single-file case)

            # First source to resolve a file owns the displayed "resolved path" for this
            # settings class; later sources don't overwrite it.
            first_resolved = self.settings_cls not in _resolved_paths  # type: ignore[attr-defined]
            file_data = None
            if file_path.is_file():
                if first_resolved:
                    _resolved_paths[self.settings_cls] = file_path  # type: ignore[attr-defined]
                file_data = self._read_file(file_path)  # type: ignore[attr-defined]
            else:
                found_path = find_upwards(file_path, self._case_sensitive)
                # find_upwards returns an absolute path as-is without an existence check,
                # so guard with is_file() — a configured-but-missing path contributes
                # nothing rather than raising FileNotFoundError on read.
                if found_path and found_path.is_file():
                    if first_resolved:
                        _resolved_paths[self.settings_cls] = found_path  # type: ignore[attr-defined]
                    file_data = self._read_file(found_path)  # type: ignore[attr-defined]

            if file_data:
                if deep_merge:
                    vars = merge_dicts(vars, file_data)
                else:
                    vars.update(file_data)

        # Stash the raw per-file dicts (nearest-first) so __call__ can apply per-file
        # profile overlay (which needs current_state, only available at call time) before
        # merging the hierarchy. Provenance is built later in _hierarchical_overlay_merge.
        self._hier_per_file = hier_per_file  # type: ignore[attr-defined]

        return vars


class EnvVarTemplateMixin:
    """Mixin for environment variable templating in yaml and toml config files."""

    def __init__(self, *args, enable_env_vars: bool = True, **kwargs):
        self._enable_env_vars = enable_env_vars
        super().__init__(*args, **kwargs)

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        with open(file_path, encoding=getattr(self, 'yaml_file_encoding', None) or getattr(self, 'toml_file_encoding', None)) as f:
            content = f.read()

        if self._enable_env_vars:
            content = self._render_env_vars(content)

        return self._parse_content(content, file_path)

    def _parse_content(self, content: str, file_path: Path) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _parse_content() "
            "to parse file content into a dict"
        )

    def _render_env_vars(self, content: str) -> str:
        """
        Process {{ env_var('VAR') }} and {{ env_var('VAR', 'default') }} patterns.
        Supports both single and double quotes.
        """
        def replace_env_var(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.lastindex and match.lastindex > 1 else None

            value = os.environ.get(var_name)
            if value is None:
                if default is not None:
                    return default
                raise ValueError(
                    f"Environment variable '{var_name}' not found and no default provided. "
                    f"Use: {{{{ env_var('{var_name}', 'default_value') }}}}"
                )
            return value

        # Match {{ env_var('VAR') }} or {{ env_var('VAR', 'default') }}
        pattern = r"\{\{\s*env_var\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*['\"]([^'\"]+)['\"])?\s*\)\s*\}\}"
        return re.sub(pattern, replace_env_var, content)


# ------------------------------------------------------------------------------
# Ancestor TOML Config Settings Source
# ------------------------------------------------------------------------------


class AncestorTomlConfigSettingsSource(AncestorConfigMixin, EnvVarTemplateMixin, TomlConfigSettingsSource):
    """
    Read config from the nearest matching toml file in this or a containing folder.
    Supports environment variable templating with {{ env_var('VAR') }} syntax (via EnvVarTemplateMixin).
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        toml_file: PathType | None = DEFAULT_PATH,
        *,
        case_sensitive: bool = False,
        enable_env_vars: bool = True,
    ):
        self.settings_cls = settings_cls  # set early — base __init__ calls _read_files before setting this
        self._case_sensitive = case_sensitive
        self._enable_env_vars = enable_env_vars

        super().__init__(settings_cls=settings_cls, toml_file=toml_file)

    def _parse_content(self, content: str, file_path: Path) -> dict[str, Any]:
        return tomllib.loads(content)

    @overlay_profile
    def __call__(self):
        return super().__call__()


# ------------------------------------------------------------------------------
# Ancestor YAML Config Settings Source
# ------------------------------------------------------------------------------


class AncestorYamlConfigSettingsSource(AncestorConfigMixin, EnvVarTemplateMixin, YamlConfigSettingsSource):
    """
    Read config from the nearest matching YAML file in this or a containing folder.
    Supports environment variable templating with {{ env_var('VAR') }} syntax (via EnvVarTemplateMixin).
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_file: PathType | None = None,
        *,
        case_sensitive: bool = False,
        enable_env_vars: bool = True,
        hierarchical: bool = False,
    ):
        self.settings_cls = settings_cls  # set early — base __init__ calls _read_files before setting this
        self._case_sensitive = case_sensitive
        self._enable_env_vars = enable_env_vars
        self._hierarchical = hierarchical

        super().__init__(settings_cls=settings_cls, yaml_file=yaml_file)

    def _parse_content(self, content: str, file_path: Path) -> dict[str, Any]:
        return yaml.safe_load(content) or {}

    @overlay_profile
    def _single_file_call(self):
        return super().__call__()

    def __call__(self):
        if getattr(self, "_hierarchical", False):
            return self._hierarchical_overlay_merge()
        return self._single_file_call()

    def _hierarchical_overlay_merge(self) -> dict[str, Any]:
        """Resolve each file's own active profile, then merge nearest-wins.

        A profile overrides values *within its own file* only; cross-file precedence
        (nearest directory wins) is applied afterwards. So a profile in a far ancestor
        cannot override a value set by a nearer file.
        """
        from .config import pivot_config_sources
        from .utils import deep_merge, resolve_active_profile_name, overlay_one

        per_file = getattr(self, "_hier_per_file", [])  # (path, raw_dict), nearest-first
        if not per_file:
            return {}

        # Active-profile name: env/init first, else the nearest file that sets it.
        merged_raw: dict[str, Any] = {}
        for _path, data in reversed(per_file):  # farthest-first => nearest wins
            merged_raw = deep_merge(merged_raw, data)
        name = resolve_active_profile_name(self.settings_cls, self.current_state, merged_raw)

        # Overlay each file's own profile block, then merge nearest-wins.
        effective = [(path, overlay_one(data, name)) for path, data in per_file]  # nearest-first
        result: dict[str, Any] = {}
        for _path, eff in reversed(effective):  # farthest-first
            result = deep_merge(result, eff)

        if "config_provenance" in self.settings_cls.model_fields:
            result["_config_provenance"] = pivot_config_sources(dict(effective))
        return result
