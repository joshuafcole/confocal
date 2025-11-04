#!/usr/bin/env python3

#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict
import os
import re
from pathlib import Path
from typing import Any, NamedTuple
from pydantic_settings.sources import DEFAULT_PATH, PathType
from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
    DefaultSettingsSource,
)
from rich.tree import Tree
import rich
import yaml

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


def deep_merge(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dicts
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def find_in(root: Path, needle: Path, case_sensitive=False) -> Path | None:
    """Enables finding subpaths case-insensitively on case-sensitive platforms."""
    if (root / needle).exists():
        return root / needle  # If it exists, we found it exactly.
    if case_sensitive:
        return None  # Otherwise, there is no case sensitive match.

    cur = root
    for part in needle.parts:
        if (cur / part).exists():
            cur = cur / part
        else:
            for entry in cur.iterdir():
                if entry.name.lower() == part.lower():
                    cur = cur / entry.name  # Found it with alternate casing
                    break
            else:
                return None  # No case insensitive match for the current part either

    return cur


def find_upwards(needle: Path, case_sensitive=False) -> Path | None:
    """
    Find the nearest ancestor which contains the given path, if any, and returns that qualified path.
    """
    # If absolute, return as-is
    if needle.is_absolute():
        return needle

    cur = Path.cwd()

    # Keep going up until we hit the root
    while True:
        found = find_in(cur, needle, case_sensitive)
        if found:
            return found

        if cur == cur.parent:
            return None  # Stop if we're at root directory
        else:
            cur = cur.parent


def pivot_config_sources(
    source_values: dict[str, dict[str, Any]]
) -> dict[str, list[tuple[str, Any]]]:
    """
    Pivot a dict of sources providing partial nested configs into
    a dict of properties containing a list of contributing sources.
    """
    result = defaultdict(list)

    for source_name, values in source_values.items():
        if "_config_provenance" in values:
            # Handle composite sources
            for key, sources in values["_config_provenance"].items():
                result[key].extend(sources)
        else:
            # Handle regular sources
            flat_values = flatten_dict(values)
            for key, value in flat_values.items():
                result[key].append((source_name, value))

    return dict(result)


def flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """
    Given an arbitrarily nested dict structure, flatten it to a single level dict with dotted keys.
    """
    result = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, new_key))
        else:
            result[new_key] = value
    return result


# ------------------------------------------------------------------------------
# Provenance Tracking
# ------------------------------------------------------------------------------

__og_default_call = DefaultSettingsSource.__call__


def inject_provenance(self: DefaultSettingsSource):
    """
    Captures which sources provided which values to help explain the final state of the config.
    """
    final = __og_default_call(self)

    if "config_provenance" in self.settings_cls.model_fields:
        final_sources = {**self.settings_sources_data, "DefaultSettingsSource": final}
        final["config_provenance"] = pivot_config_sources(final_sources)

    return final


DefaultSettingsSource.__call__ = inject_provenance


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


# ------------------------------------------------------------------------------
# Provenance Explanation
# ------------------------------------------------------------------------------


class FieldStyle(NamedTuple):
    fg_val: str
    bg_val: str
    fg_source: str
    bg_source: str


active_style = FieldStyle("white", "gray39", "white", "gray23")
inactive_style = FieldStyle("gray", "grey23", "dim", "gray15")

DEFAULT_SKIP_FIELDS = {"config_provenance", "profiles", "active_profile"}

SOURCE_LABELS = {
    "InitSettingsSource": "passed kwarg",
    "EnvSettingsSource": "environment variable",
    "ProfileMixin": "active profile",
    "AncestorYamlConfigSettingsSource": "config.yaml",
    "AncestorTomlConfigSettingsSource": "config.toml",
    "DefaultSettingsSource": "default",
}


def show_provenance_node(
    parent: Tree, name: str, sources: list[tuple[str, Any]], verbose: bool
) -> None:
    if not sources:
        return

    parts = [f"{name} ="]
    for ix, (current_source, current_value) in enumerate(sources):
        (fg_val, bg_val, fg_source, bg_source) = (
            active_style if ix == 0 else inactive_style
        )
        parts.append(
            f"[{fg_val} on {bg_val}] {current_value} [/]"
            + f"[{fg_source} on {bg_source}] {SOURCE_LABELS[current_source]} [/]"
        )
        if not verbose:
            break

    parent.add(" ".join(parts))


def show_provenance(
    config: BaseModel,
    provenance: dict[str, list[tuple[str, Any]]],
    verbose=False,
    skip_fields: set[str] | None = None,
    tree: Tree | None = None,
    path: str = "",
) -> None:
    """Pretty print config showing value sources and overrides."""
    if skip_fields is None:
        skip_fields = DEFAULT_SKIP_FIELDS

    is_root = tree is None
    if tree is None:
        tree = Tree("Configuration")

    for field_name in config.model_fields:
        full_path = f"{path}.{field_name}" if path else field_name
        if full_path in skip_fields:
            continue

        value = getattr(config, field_name)
        if hasattr(value, "model_fields"):
            # Nested config
            subtree = tree.add(field_name)
            show_provenance(value, provenance, verbose, skip_fields, subtree, full_path)
        else:
            # leaf property
            sources = provenance.get(field_name) or [("DefaultSettingsSource", value)]
            show_provenance_node(tree, field_name, sources, verbose)

    if is_root:
        rich.print(tree)


# ------------------------------------------------------------------------------
# Base Config
# ------------------------------------------------------------------------------


class BaseConfig(BaseSettings):
    """
    Shared Config + Base class for specialized configs
    """

    model_config = SettingsConfigDict(
        toml_file=None,
        yaml_file=None,
        extra="ignore",
        nested_model_default_partial_update=True,
        env_nested_delimiter="__",
    )

    _config_title: str | None = None
    config_provenance: dict[str, list[tuple[str, Any]]] = Field(
        default_factory=dict, exclude=True, repr=False
    )

    def explain(self, verbose=False):
        tree = Tree(self._config_title or self.__class__.__name__)
        show_provenance(self, self.config_provenance, verbose, None, tree)
        rich.print(tree)

    @classmethod
    def load(cls, *args, **kwargs):
        """
        Attempt to create an instance of `cls` ignoring required fields.
        If all required fields are not provided by a config file, env var etc.
        this will throw an error at runtime.
        """
        return cls(*args, **kwargs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources = [init_settings, env_settings, dotenv_settings]

        yaml_file = settings_cls.model_config.get("yaml_file")
        toml_file = settings_cls.model_config.get("toml_file")

        if yaml_file and toml_file:
            raise ValueError(
                "Cannot specify both 'yaml_file' and 'toml_file' in model_config. "
                "Please use only one config file format."
            )

        if yaml_file:
            sources.append(AncestorYamlConfigSettingsSource(settings_cls, yaml_file))
        elif toml_file:
            sources.append(AncestorTomlConfigSettingsSource(settings_cls))

        sources.append(file_secret_settings)

        return tuple(sources)
