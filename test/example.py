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
    engine_size = "S"
    engine_auto_suspend: int = 60

    # Flags
    use_graph_index = True
    use_package_manager = False
    ensure_change_tracking = False


class AzureConfig(Config):
    platform: Literal["azure"]
    host = "azure.relationalai.com"
    port = 443
    region = "us-east"
    scheme = "https"
    client_credentials_url = "https://login.relationalai.com/oauth/token"
    client_id: str
    client_secret: str


class SnowflakeConfig(Config):
    platform: Literal["snowflake"] = "snowflake"
    user: str
    account: str
    role = "PUBLIC"
    warehouse: str
    rai_app_name: str

    # Override default engine size
    engine_size = "HIGHMEM_X64_S"


cfg = Config(foo="bye", compiler={"optimization_level": 5}, alpha="h")
cfg.explain(True)

cfg.explain()

Config.load(foo="bye", compiler={"optimization_level": 5}, alpha="h")
