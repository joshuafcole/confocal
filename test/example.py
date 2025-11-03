#!/usr/bin/env python3


from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict
from confocal import BaseConfig, Profile


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
# Example 2: YAML Config with Profiles
# ------------------------------------------------------------------------------

class MyProfile(Profile):
    """Custom profile extending the base Profile class"""
    database_url: str


class YamlConfig(BaseConfig):
    """
    Config that supports both file loading and direct config passing.

    - YamlConfig() - loads from config.yaml file
    - YamlConfig(profiles={...}) - skips file loading, uses passed data
    """
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
    )

    profiles: dict[str, MyProfile]
    database_url: str
    debug: bool = False


# ------------------------------------------------------------------------------
# Run Examples
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Example 1: TOML Config")
    print("=" * 70)
    cfg = Config(foo="bye", compiler={"optimization_level": 5}, alpha="h")
    cfg.explain(True)
    print()
    cfg.explain()

    print("\n" + "=" * 70)
    print("Example 2: YAML Config - Load from File")
    print("=" * 70)
    print("# Load from config.yaml:")
    print("config = YamlConfig()")
    print("# or")
    print("config = YamlConfig.load()")
    print()

    print("=" * 70)
    print("Example 3: YAML Config - Direct Config Passing (Skip File Loading)")
    print("=" * 70)
    print("# Same config class, but pass profiles directly:")
    print()

    # Create profiles in code
    dev_profile = MyProfile(database_url='postgresql://localhost:5432')
    prod_profile = MyProfile(database_url='postgresql://prod:5432')

    # Pass directly to the SAME config class, skipping file loading
    direct_config = YamlConfig(
        database_url='my_direct_db',
        debug=True,
        profiles={'dev': dev_profile, 'prod': prod_profile}
    )

    print("config = YamlConfig(")
    print("    database_url='my_direct_db',")
    print("    debug=True,")
    print("    profiles={'dev': dev_profile, 'prod': prod_profile}")
    print(")")
    print()
    print("✅ File loading was skipped!")
    print(f"   database_url: {direct_config.database_url}")
    print(f"   debug: {direct_config.debug}")
    print(f"   profiles: {list(direct_config.profiles.keys())}")
    print()

    print("=" * 70)
    print("Key Features:")
    print("=" * 70)
    print("✓ TOML config loading")
    print("✓ YAML config loading with profiles")
    print("✓ Custom Profile types with extra fields")
    print("✓ Direct config passing (skips file loading)")
    print("✓ Provenance tracking with explain()")
    print()
