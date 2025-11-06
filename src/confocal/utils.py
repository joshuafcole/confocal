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

        # Find the profile selector field
        profile_field = "active_profile"

        active_profile_name = cur_state.get(profile_field)

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
