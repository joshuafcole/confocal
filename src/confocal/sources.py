#!/usr/bin/env python3

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any

from pydantic_settings.sources import DEFAULT_PATH, PathType
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings
from pydantic_settings.sources import (
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)
import yaml

from .utils import find_upwards


# ------------------------------------------------------------------------------
# Config Source Mixins
# ------------------------------------------------------------------------------


class AncestorConfigMixin:
    """Mixin for finding config files in ancestor directories."""

    def __init__(self, *args, case_sensitive: bool = False, **kwargs):
        self._case_sensitive = case_sensitive
        super().__init__(*args, **kwargs)

    def _read_files(self, files: PathType | None) -> dict[str, Any]:
        """Read config files, searching upwards if not found directly."""
        if files is None:
            return {}
        if isinstance(files, (str, os.PathLike)):
            files = [files]

        vars: dict[str, Any] = {}
        for file in files:
            file_path = Path(file).expanduser()
            if file_path.is_file():
                vars.update(self._read_file(file_path))
            else:
                found_path = find_upwards(file_path, self._case_sensitive)
                if found_path:
                    vars.update(self._read_file(found_path))
        return vars


class EnvVarTemplateMixin:
    """Mixin for environment variable templating in config files."""

    def __init__(self, *args, enable_env_vars: bool = True, **kwargs):
        self._enable_env_vars = enable_env_vars
        super().__init__(*args, **kwargs)

    def _render_env_vars(self, content: str) -> str:
        """
        Process {{ env_var('VAR') }} and {{ env_var('VAR', 'default') }} patterns.

        Supports both single and double quotes.
        Raises ValueError if env var not found and no default provided.
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


class AncestorTomlConfigSettingsSource(AncestorConfigMixin, TomlConfigSettingsSource):
    """
    Read config from the nearest matching toml file in this or a containing folder.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        toml_file: PathType | None = DEFAULT_PATH,
        *,
        case_sensitive=False,
    ):
        # Call mixin init which will call super().__init__
        super().__init__(case_sensitive=case_sensitive, settings_cls=settings_cls, toml_file=toml_file)


# ------------------------------------------------------------------------------
# Ancestor YAML Config Settings Source
# ------------------------------------------------------------------------------


class AncestorYamlConfigSettingsSource(AncestorConfigMixin, EnvVarTemplateMixin, PydanticBaseSettingsSource):
    """
    Read config from the nearest matching YAML file in this or a containing folder.
    Supports environment variable templating with {{ env_var('VAR') }} syntax.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_file: str | Path | None = None,
        *,
        case_sensitive: bool = False,
        enable_env_vars: bool = True,
    ):
        self._yaml_file = yaml_file
        # Call mixin inits which will call PydanticBaseSettingsSource.__init__
        super().__init__(
            case_sensitive=case_sensitive,
            enable_env_vars=enable_env_vars,
            settings_cls=settings_cls
        )

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Not used since we override __call__ directly
        return None, "", False

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        """Read and parse a YAML file, applying env var templating if enabled."""
        with open(file_path) as f:
            content = f.read()

        if self._enable_env_vars:
            content = self._render_env_vars(content)

        return yaml.safe_load(content) or {}

    def __call__(self) -> dict[str, Any]:
        return self._read_files(self._yaml_file)
