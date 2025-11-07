#!/usr/bin/env python3

#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict
from typing import Any, NamedTuple
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (
    PydanticBaseSettingsSource,
    DefaultSettingsSource,
)
from rich.tree import Tree
import rich

from .sources import AncestorTomlConfigSettingsSource, AncestorYamlConfigSettingsSource

# ------------------------------------------------------------------------------
# Profile Config Settings Source
# ------------------------------------------------------------------------------


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


DefaultSettingsSource.__call__ = inject_provenance # type: ignore[method-assign]


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
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict, alias="profile")
    active_profile: str | None = Field(default=None)

    def explain(self, verbose=False):
        tree = Tree(self._config_title or self.__class__.__name__)
        profiles = getattr(self, "profiles", None)
        active = getattr(self, "active_profile", None)
        if profiles:
            profiles_node = tree.add("Available Profiles")
            for profile in profiles.keys():
                profiles_node.add(
                    f"{profile} (active)"
                    if active == profile
                    else f"[dim]{profile}[/dim]"
                )

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
            sources.append(AncestorTomlConfigSettingsSource(settings_cls, toml_file))

        sources.append(file_secret_settings)

        return tuple(sources)
