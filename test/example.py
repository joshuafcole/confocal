#!/usr/bin/env python3


from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict
from confocal import BaseConfig


class CompilerConfig(BaseModel):
    experimental: bool = False
    optimization_level: int = 0


class Config(BaseConfig):
    model_config = SettingsConfigDict(
        toml_file="raiconfig.toml",
        env_prefix="rai_",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    compiler: CompilerConfig

    # optional, override validation aliases
    active_profile: str | None = Field(default=None, validation_alias="rai_profile")

    # Engine
    engine: str | None = None
    engine_size: str = "S"
    engine_auto_suspend: int = 60

    # Flags
    use_graph_index: bool = True
    use_package_manager: bool = False
    ensure_change_tracking: bool = False


class AzureConfig(Config):
    platform: Literal["azure"]
    host: str = "azure.relationalai.com"
    port: int = 443
    region: str = "us-east"
    scheme: str = "https"
    client_credentials_url: str = "https://login.relationalai.com/oauth/token"
    client_id: str
    client_secret: str


class SnowflakeConfig(Config):
    platform: Literal["snowflake"] = "snowflake"
    user: str
    account: str
    role: str = "PUBLIC"
    warehouse: str
    rai_app_name: str

    # Override default engine size
    engine_size: str = "HIGHMEM_X64_S"


# ------------------------------------------------------------------------------
# Example 2: YAML Config
# ------------------------------------------------------------------------------

class YamlConfig(BaseConfig):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
    )

    database_url: str
    debug: bool = False


# ------------------------------------------------------------------------------
# Run Examples
# ------------------------------------------------------------------------------

cfg = Config(foo="bye", compiler={"optimization_level": 5}, alpha="h")
cfg.explain()
