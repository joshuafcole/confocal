"""Tests for YAML configuration loading with env var templating."""
import os
from typing import Optional
import pytest
from pydantic import AliasChoices, Field
from pydantic_settings import SettingsConfigDict
from confocal import BaseConfig
from tests import YamlTestConfig


class ProfileSingularYamlConfig(BaseConfig):
    """Config loading a fixture that uses the singular 'profile:' key."""

    model_config = SettingsConfigDict(
        yaml_file="tests/fixtures/test_config_profile_singular.yaml",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    database_url: str
    api_key: Optional[str] = None
    debug: bool = False
    timeout: int = 30
    max_connections: Optional[int] = None


class AliasedProfileYamlConfig(BaseConfig):
    """Config with a validation alias on active_profile, to test overlay_profile alias resolution."""

    model_config = SettingsConfigDict(
        yaml_file="tests/fixtures/test_config.yaml",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    active_profile: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CUSTOM_ACTIVE_PROFILE", "ACTIVE_PROFILE", "active_profile"),
    )
    database_url: str
    api_key: Optional[str] = None
    debug: bool = False
    timeout: int = 30
    max_connections: Optional[int] = None


class TestYamlConfigLoading:
    """Test suite for YAML configuration loading."""
    
    @pytest.fixture
    def clean_env(self):
        """Clean environment variables before each test."""
        env_vars = ['TEST_DB_URL', 'TEST_API_KEY', 'PROD_DB_URL', 'PROD_API_KEY']
        original_values = {}
        
        # Save and clear
        for var in env_vars:
            original_values[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
        
        yield
        
        # Restore
        for var, value in original_values.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_yaml_loads_with_default_env_vars(self, clean_env):
        config = YamlTestConfig(active_profile='dev')
        
        assert config.database_url == "postgresql://localhost:5432/dev"
        assert config.api_key == "dev_key_123"
        assert config.debug is True
        assert config.max_connections == 10
        assert config.timeout == 30  # From global defaults
    
    def test_yaml_loads_with_custom_env_vars(self, clean_env):
        # Set custom env vars
        os.environ['TEST_DB_URL'] = 'postgresql://custom:5432/test'
        os.environ['TEST_API_KEY'] = 'custom_test_key'
        
        config = YamlTestConfig(active_profile='dev')
        
        assert config.database_url == "postgresql://custom:5432/test"
        assert config.api_key == "custom_test_key"
        assert config.debug is True
        assert config.max_connections == 10
    
    def test_yaml_prod_profile_with_env_vars(self, clean_env):
        os.environ['PROD_DB_URL'] = 'postgresql://production:5432/prod'
        os.environ['PROD_API_KEY'] = 'super_secret_prod_key'
        
        config = YamlTestConfig(active_profile='prod')
        
        assert config.database_url == "postgresql://production:5432/prod"
        assert config.api_key == "super_secret_prod_key"
        assert config.debug is False  # prod profile setting
        assert config.max_connections == 100  # prod profile setting
    
    def test_yaml_global_defaults_without_profile(self, clean_env):
        config = YamlTestConfig()
        
        assert config.database_url == "postgresql://localhost:5432/dev"  # Uses dev profile from file's default
        assert config.debug is True
        assert config.timeout == 30
    
    def test_yaml_missing_env_var_uses_default(self, clean_env):
        # Only set one env var
        os.environ['TEST_DB_URL'] = 'postgresql://partial:5432/test'
        
        config = YamlTestConfig(active_profile='dev')
        
        assert config.database_url == "postgresql://partial:5432/test"
        assert config.api_key == "dev_key_123"  # Uses default


class TestActiveProfileAliasResolution:
    """Test that overlay_profile respects validation_alias on active_profile.

    When active_profile has AliasChoices, an env var matching a non-field-name
    alias (e.g. CUSTOM_ACTIVE_PROFILE) should still trigger the profile overlay
    and take priority over the YAML's own active_profile value.
    """

    @pytest.fixture(autouse=True)
    def clean_env(self):
        keys = ["CUSTOM_ACTIVE_PROFILE", "ACTIVE_PROFILE"]
        saved = {k: os.environ.pop(k, None) for k in keys}
        yield
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    def test_alias_env_var_selects_profile(self):
        os.environ["CUSTOM_ACTIVE_PROFILE"] = "prod"
        config = AliasedProfileYamlConfig()
        assert config.active_profile == "prod"
        assert config.debug is False
        assert config.max_connections == 100

    def test_alias_env_var_overrides_yaml_active_profile(self):
        # YAML has active_profile: dev — env alias should win
        os.environ["CUSTOM_ACTIVE_PROFILE"] = "prod"
        config = AliasedProfileYamlConfig()
        assert config.active_profile == "prod"
        assert config.max_connections == 100  # prod value, not dev's 10

    def test_field_name_env_var_still_works(self):
        os.environ["ACTIVE_PROFILE"] = "prod"
        config = AliasedProfileYamlConfig()
        assert config.debug is False
        assert config.max_connections == 100

    def test_yaml_active_profile_resolves_when_aliases_reorder_key(self):
        """YAML uses key 'active_profile', but AliasChoices puts a non-field-name
        alias first. pydantic-settings' YAML source rewrites the returned key to
        the first AliasChoices entry, so the overlay must scan all candidate keys
        in cur_state — not just 'active_profile'."""
        # No env vars set — rely on YAML's own active_profile: dev
        config = AliasedProfileYamlConfig()
        assert config.active_profile == "dev"
        # Overlay applied (dev profile has debug=True, max_connections=10)
        assert config.debug is True
        assert config.max_connections == 10


class TestProfilesKeyAlias:
    """Test that both 'profile:' and 'profiles:' YAML keys populate config.profiles.

    Before the fix, only 'profile:' populated config.profiles (pydantic alias).
    'profiles:' worked for overlay but left config.profiles empty.
    After the fix, both keys are accepted via AliasChoices.
    """

    def test_profiles_plural_key_populates_config_profiles(self):
        # test_config.yaml uses 'profiles:' (plural)
        config = YamlTestConfig(active_profile="dev")
        assert "dev" in config.profiles
        assert "prod" in config.profiles

    def test_profile_singular_key_populates_config_profiles(self):
        # test_config_profile_singular.yaml uses 'profile:' (singular)
        config = ProfileSingularYamlConfig(active_profile="dev")
        assert "dev" in config.profiles
        assert "prod" in config.profiles

    def test_profiles_plural_overlay_applies_correctly(self):
        # Overlay still applies with 'profiles:' key
        config = YamlTestConfig(active_profile="prod")
        assert config.debug is False
        assert config.max_connections == 100

    def test_profile_singular_overlay_applies_correctly(self):
        # Overlay still applies with 'profile:' key
        config = ProfileSingularYamlConfig(active_profile="prod")
        assert config.debug is False
        assert config.max_connections == 100
