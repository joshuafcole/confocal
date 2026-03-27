#!/usr/bin/env python3

#!/usr/bin/env python3
from __future__ import annotations
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (
    PydanticBaseSettingsSource,
    DefaultSettingsSource,
)
from rich.tree import Tree
import rich

from .sources import AncestorTomlConfigSettingsSource, AncestorYamlConfigSettingsSource, _resolved_paths


# ------------------------------------------------------------------------------
# Extended SettingsConfigDict
# ------------------------------------------------------------------------------


class ConfocalSettingsConfigDict(SettingsConfigDict, total=False):
    """Extension of pydantic-settings' ``SettingsConfigDict`` with confocal-specific options.

    Use this instead of ``SettingsConfigDict`` when you need ``env_file_override``.

    Extra fields
    ------------
    env_file_override : str, optional
        Name of an environment variable whose value, when set, overrides the
        ``yaml_file`` / ``toml_file`` path configured on the class.  The env var
        value must be a path to a file of the same format as the class's configured
        file type (YAML or TOML).  If the env var is not set the configured
        ``yaml_file`` / ``toml_file`` is used unchanged.
    """

    env_file_override: str

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

# Module-level store: maps settings class → computed provenance dict.
# Populated by inject_provenance (called during source evaluation) and consumed
# by BaseConfig.model_post_init.  Using the class as key is safe because only
# one instance is being built at a time per class within a single thread.
_pending_provenance: dict[type, dict] = {}

__og_default_call = DefaultSettingsSource.__call__


def inject_provenance(self: DefaultSettingsSource):
    """
    Captures which sources provided which values to help explain the final state of the config.

    The provenance dict cannot be injected directly into the returned ``final`` dict because
    pydantic-settings strips any value that equals its field default before passing the merged
    state to the model constructor (``config_provenance`` defaults to ``{}``, so it would always
    be stripped).  Instead we stash it in ``_pending_provenance`` keyed by the settings class
    and let ``BaseConfig.model_post_init`` pick it up after the model is built.
    """
    final = __og_default_call(self)

    if "config_provenance" in self.settings_cls.model_fields:
        final_sources = {**self.settings_sources_data, "DefaultSettingsSource": final}
        _pending_provenance[self.settings_cls] = pivot_config_sources(final_sources)

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
    parent: Tree,
    name: str,
    sources: list[tuple[str, Any]],
    verbose: bool,
    source_labels: dict[str, str] | None = None,
) -> None:
    if not sources:
        return

    labels = source_labels if source_labels is not None else SOURCE_LABELS
    parts = [f"{name} ="]
    for ix, (current_source, current_value) in enumerate(sources):
        (fg_val, bg_val, fg_source, bg_source) = (
            active_style if ix == 0 else inactive_style
        )
        label = labels.get(current_source, current_source)
        parts.append(
            f"[{fg_val} on {bg_val}] {current_value} [/]"
            + f"[{fg_source} on {bg_source}] {label} [/]"
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
    source_labels: dict[str, str] | None = None,
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
            show_provenance(value, provenance, verbose, skip_fields, subtree, full_path, source_labels)
        else:
            # leaf property
            sources = provenance.get(field_name) or [("DefaultSettingsSource", value)]
            show_provenance_node(tree, field_name, sources, verbose, source_labels)

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

    def model_post_init(self, __context: Any) -> None:
        """Attach provenance data and resolved config file path computed during settings-source evaluation."""
        prov = _pending_provenance.pop(type(self), None)
        if prov is not None:
            object.__setattr__(self, "config_provenance", prov)

        resolved_path = _resolved_paths.pop(type(self), None)
        if resolved_path is not None:
            object.__setattr__(self, "_resolved_config_file", str(resolved_path))

    def explain(self, verbose=False):
        # Show the resolved config file path as a header line
        file_path = getattr(self, "_resolved_config_file", None)
        if file_path:
            env_override_key = self.model_config.get("env_file_override")
            if env_override_key and os.environ.get(str(env_override_key)):
                rich.print(f"[bold]Config file:[/bold] [cyan]{file_path}[/cyan] [dim](via {env_override_key})[/dim]")
            else:
                rich.print(f"[bold]Config file:[/bold] [cyan]{file_path}[/cyan]")

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

        # Build source labels — show the actual file path when env_file_override is active
        source_labels = dict(SOURCE_LABELS)
        env_override_key = self.model_config.get("env_file_override")
        if env_override_key:
            env_path = os.environ.get(str(env_override_key))
            if env_path:
                label = f"{env_path} (via {env_override_key})"
                source_labels["AncestorYamlConfigSettingsSource"] = label
                source_labels["AncestorTomlConfigSettingsSource"] = label

        show_provenance(self, self.config_provenance, verbose, None, tree, source_labels=source_labels)
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

        # If the class opts in via env_file_override, check the named env var and
        # replace yaml_file / toml_file with the env var path when it is set.
        # Format is determined from the env var path's extension, regardless of
        # what yaml_file / toml_file was configured on the class (it may be None
        # if the default file was not found at import time).
        env_override_key = settings_cls.model_config.get("env_file_override")
        if env_override_key:
            env_path = os.environ.get(str(env_override_key))
            if env_path:
                ext = Path(env_path).suffix.lower()
                if ext in (".yaml", ".yml"):
                    yaml_file = [env_path]
                    toml_file = None
                elif ext == ".toml":
                    toml_file = env_path
                    yaml_file = None
                else:
                    raise ValueError(
                        f"{env_override_key}={env_path!r} has unsupported extension {ext!r}. "
                        "Expected .yaml, .yml, or .toml."
                    )

        if yaml_file:
            sources.append(AncestorYamlConfigSettingsSource(settings_cls, yaml_file))
        elif toml_file:
            sources.append(AncestorTomlConfigSettingsSource(settings_cls, toml_file))

        sources.append(file_secret_settings)

        return tuple(sources)
