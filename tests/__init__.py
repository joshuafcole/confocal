from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from confocal import BaseConfig


class YamlTestConfig(BaseConfig):
    model_config = SettingsConfigDict(
        yaml_file="tests/fixtures/test_config.yaml",
        extra="ignore",
        nested_model_default_partial_update=True,
    )
    
    database_url: str
    api_key: str | None = None
    debug: bool = False
    timeout: int = 30
    max_connections: int | None = None


class TomlTestConfig(BaseConfig):
    model_config = SettingsConfigDict(
        toml_file="tests/fixtures/test_config.toml",
        extra="ignore",
        nested_model_default_partial_update=True,
    )
    
    database_url: str
    api_key: str | None = None
    debug: bool = False
    timeout: int = 30
    max_connections: int | None = None
