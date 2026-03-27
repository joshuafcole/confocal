"""Tests for env_file_override — loading config from a path set by an environment variable."""
import os
import pytest
from typing import Optional
from pydantic_settings import SettingsConfigDict
from confocal import BaseConfig, ConfocalSettingsConfigDict


# ---------------------------------------------------------------------------
# Test config classes
# ---------------------------------------------------------------------------

class YamlConfigWithOverride(BaseConfig):
    """YAML config that opts in to env_file_override."""
    model_config = ConfocalSettingsConfigDict(
        yaml_file="tests/fixtures/test_config.yaml",
        env_file_override="TEST_CONFIG_FILE",
        extra="ignore",
        nested_model_default_partial_update=True,
    )
    database_url: str
    api_key: Optional[str] = None
    debug: bool = False
    timeout: int = 30
    max_connections: Optional[int] = None


class TomlConfigWithOverride(BaseConfig):
    """TOML config that opts in to env_file_override."""
    model_config = ConfocalSettingsConfigDict(
        toml_file="tests/fixtures/test_config.toml",
        env_file_override="TEST_CONFIG_FILE",
        extra="ignore",
        nested_model_default_partial_update=True,
    )
    database_url: str
    api_key: Optional[str] = None
    debug: bool = False
    timeout: int = 30
    max_connections: Optional[int] = None


class YamlConfigWithoutOverride(BaseConfig):
    """YAML config that does NOT opt in — should be unaffected by env var."""
    model_config = SettingsConfigDict(
        yaml_file="tests/fixtures/test_config.yaml",
        extra="ignore",
        nested_model_default_partial_update=True,
    )
    database_url: str
    api_key: Optional[str] = None
    debug: bool = False
    timeout: int = 30
    max_connections: Optional[int] = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_override_env():
    """Ensure TEST_CONFIG_FILE is unset before and after each test."""
    original = os.environ.pop("TEST_CONFIG_FILE", None)
    yield
    if original is not None:
        os.environ["TEST_CONFIG_FILE"] = original
    else:
        os.environ.pop("TEST_CONFIG_FILE", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnvFileOverride:

    def test_yaml_override_loads_from_env_var_path(self, tmp_path):
        """When env var points to an absolute YAML path, that file is loaded."""
        alt_config = tmp_path / "alt_config.yaml"
        alt_config.write_text(
            "database_url: postgresql://alt:5432/alt\n"
            "api_key: alt_key\n"
            "debug: true\n"
        )

        os.environ["TEST_CONFIG_FILE"] = str(alt_config)
        config = YamlConfigWithOverride()

        assert config.database_url == "postgresql://alt:5432/alt"
        assert config.api_key == "alt_key"
        assert config.debug is True

    def test_toml_override_loads_from_env_var_path(self, tmp_path):
        """When env var points to an absolute TOML path, that file is loaded."""
        alt_config = tmp_path / "alt_config.toml"
        alt_config.write_text(
            'database_url = "postgresql://alt-toml:5432/alt"\n'
            'api_key = "alt_toml_key"\n'
            "debug = true\n"
        )

        os.environ["TEST_CONFIG_FILE"] = str(alt_config)
        config = TomlConfigWithOverride()

        assert config.database_url == "postgresql://alt-toml:5432/alt"
        assert config.api_key == "alt_toml_key"
        assert config.debug is True

    def test_env_var_not_set_uses_default_yaml_file(self):
        """When env var is absent, the class's default yaml_file is used."""
        # TEST_CONFIG_FILE not set — loads from tests/fixtures/test_config.yaml
        config = YamlConfigWithOverride()
        assert config.timeout == 30  # value from the default fixture file

    def test_env_var_not_set_uses_default_toml_file(self):
        """When env var is absent, the class's default toml_file is used."""
        config = TomlConfigWithOverride()
        assert config.timeout == 30

    def test_class_without_opt_in_unaffected_by_env_var(self, tmp_path):
        """A class without env_file_override is not affected even if the env var is set."""
        alt_config = tmp_path / "alt_config.yaml"
        alt_config.write_text("database_url: postgresql://should-not-load:5432/x\n")

        os.environ["TEST_CONFIG_FILE"] = str(alt_config)
        # YamlConfigWithoutOverride has no env_file_override — loads its own fixture
        config = YamlConfigWithoutOverride()

        assert config.database_url != "postgresql://should-not-load:5432/x"
        assert config.timeout == 30  # from the real fixture

    def test_unsupported_extension_raises_error(self, tmp_path):
        """An unsupported file extension raises a clear ValueError."""
        bad_file = tmp_path / "config.json"
        bad_file.write_text("{}")

        os.environ["TEST_CONFIG_FILE"] = str(bad_file)

        with pytest.raises(ValueError, match="unsupported extension"):
            YamlConfigWithOverride()

    def test_yaml_override_with_profiles(self, tmp_path):
        """env_file_override works correctly with profile overlays."""
        alt_config = tmp_path / "alt_with_profiles.yaml"
        alt_config.write_text(
            "database_url: base_url\n"
            "active_profile: staging\n"
            "profile:\n"
            "  staging:\n"
            "    database_url: postgresql://staging:5432/db\n"
            "    debug: true\n"
        )

        os.environ["TEST_CONFIG_FILE"] = str(alt_config)
        config = YamlConfigWithOverride()

        assert config.database_url == "postgresql://staging:5432/db"
        assert config.debug is True
