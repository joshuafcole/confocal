"""Profile and provenance internals must never appear in a config dump."""
from __future__ import annotations

from pydantic import SecretStr

from confocal import BaseConfig


class _Cfg(BaseConfig):
    name: str = "base"
    token: SecretStr = SecretStr("")


def _build() -> _Cfg:
    return _Cfg(
        profiles={"dev": {"token": "profile-secret", "name": "dev"}},
        active_profile="dev",
    )


def test_model_dump_omits_profiles_and_provenance():
    dumped = _build().model_dump()
    assert "profiles" not in dumped
    assert "config_provenance" not in dumped
    assert "profile-secret" not in str(dumped)


def test_json_dump_and_repr_omit_profile_values():
    cfg = _build()
    assert "profile-secret" not in cfg.model_dump_json()
    assert "profile-secret" not in repr(cfg)


def test_profiles_still_readable_for_explain():
    assert _build().profiles["dev"]["name"] == "dev"
