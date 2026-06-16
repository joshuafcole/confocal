#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
from pydantic_settings.sources import PydanticBaseSettingsSource


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


def profile_candidate_keys(settings_cls) -> list[str]:
    """Keys that may carry the active-profile selector: the field name plus any
    validation aliases (pydantic-settings keys alias-matched env vars by the alias)."""
    keys = ["active_profile"]
    field_info = settings_cls.model_fields.get("active_profile")
    if field_info is not None and field_info.validation_alias is not None:
        from pydantic import AliasChoices
        va = field_info.validation_alias
        if isinstance(va, AliasChoices):
            keys += [c for c in va.choices if isinstance(c, str)]
        elif isinstance(va, str):
            keys.append(va)
    return keys


def resolve_active_profile_name(settings_cls, current_state: dict, fallback_state: dict):
    """Resolve the active-profile name: prefer higher-priority sources (env vars, init)
    in ``current_state``, then fall back to the merged config ``fallback_state``."""
    keys = profile_candidate_keys(settings_cls)
    return (
        next((current_state[k] for k in keys if k in current_state), None)
        or next((fallback_state[k] for k in keys if k in fallback_state), None)
    )


def overlay_one(data: dict, profile_name: str | None) -> dict:
    """Overlay ``data``'s own ``profile[profile_name]`` block onto ``data`` — the profile
    overrides values *within this same dict*. Returns ``data`` unchanged when the profile
    is absent. Used for per-file profile resolution before cross-file merging."""
    if not profile_name:
        return data
    profiles = data.get("profiles") or data.get("profile") or {}
    block = profiles.get(profile_name)
    if isinstance(block, dict):
        return deep_merge(data, block)
    return data


def overlay_profile(fn):
    """
    Decorator for `PydanticBaseSettingsSource` descendant's `__call__` method
    which adds a virtual "profile" source with higher precedence, but keeping
    access to this sources data. Merges all keys under profile.<active_profile>
    into the base config.
    """

    def wrapper(self: PydanticBaseSettingsSource):
        # Import here to avoid circular dependency (at call time, not decorator time)
        from .config import pivot_config_sources

        source_state = fn(self)
        cur_state = deep_merge(source_state, self.current_state)

        active_profile_name = resolve_active_profile_name(
            self.settings_cls, self.current_state, cur_state
        )

        # Check both singular and plural forms for profiles
        profiles = cur_state.get("profiles") or cur_state.get("profile", {})
        active_profile = profiles.get(active_profile_name, {})

        if not active_profile or not isinstance(active_profile, dict):
            return source_state

        merged = deep_merge(source_state, active_profile)
        if "config_provenance" in self.settings_cls.model_fields:
            combined_sources = {
                "ProfileMixin": active_profile,
                self.__class__.__name__: source_state,
            }
            merged["_config_provenance"] = pivot_config_sources(combined_sources)

        return merged

    return wrapper


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


def find_all_upwards(needle: Path, case_sensitive=False) -> list[Path]:
    """
    Find every ancestor (including the current directory) that contains the given
    path, walking from the current working directory up to the filesystem root.

    Returns the qualified matches ordered nearest-first: index 0 is the match closest
    to the current directory, and the last element is the one closest to the filesystem
    root. Returns an empty list if no ancestor contains the path.

    This is the multi-match counterpart of :func:`find_upwards`, which returns only the
    nearest match. The walk stops at the filesystem root, matching ``find_upwards``.

    For an absolute ``needle`` the path is returned as-is in a single-element list,
    mirroring :func:`find_upwards`.
    """
    if needle.is_absolute():
        return [needle]

    matches: list[Path] = []
    cur = Path.cwd()

    while True:
        found = find_in(cur, needle, case_sensitive)
        if found:
            matches.append(found)

        if cur == cur.parent:
            break  # Reached the filesystem root
        cur = cur.parent

    return matches
