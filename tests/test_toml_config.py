"""Tests for TOML configuration loading with env var templating."""
import os
import pytest
from tests import TomlTestConfig, YamlTestConfig


class TestTomlConfigLoading:
    """Test suite for TOML configuration loading."""
    
    @pytest.fixture
    def clean_env(self):
        """Clean environment variables before each test."""
        env_vars = ['TEST_DB_URL', 'TEST_API_KEY', 'PROD_DB_URL', 'PROD_API_KEY']
        original_values = {}
        
        for var in env_vars:
            original_values[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
        
        yield
        
        for var, value in original_values.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_toml_loads_with_default_env_vars(self, clean_env):
        config = TomlTestConfig(active_profile='dev')
        
        assert config.database_url == "postgresql://localhost:5432/dev"
        assert config.api_key == "dev_key_123"
        assert config.debug is True
        assert config.max_connections == 10
    
    def test_toml_loads_with_custom_env_vars(self, clean_env):
        os.environ['TEST_DB_URL'] = 'postgresql://toml-custom:5432/test'
        os.environ['TEST_API_KEY'] = 'toml_custom_key'
        
        config = TomlTestConfig(active_profile='dev')
        
        assert config.database_url == "postgresql://toml-custom:5432/test"
        assert config.api_key == "toml_custom_key"
    
    def test_toml_prod_profile(self, clean_env):
        config = TomlTestConfig(active_profile='prod')
        
        # Uses defaults since env vars not set
        assert config.database_url == "postgresql://prod:5432/prod"
        assert config.api_key == "prod_key_456"
        assert config.debug is False
        assert config.max_connections == 100
    
    def test_toml_yaml_consistency(self, clean_env):
        os.environ['TEST_DB_URL'] = 'postgresql://consistency:5432/test'
        os.environ['TEST_API_KEY'] = 'consistency_key'
        
        toml_config = TomlTestConfig(active_profile='dev')
        yaml_config = YamlTestConfig(active_profile='dev')
        
        assert toml_config.database_url == yaml_config.database_url
        assert toml_config.api_key == yaml_config.api_key
        assert toml_config.debug == yaml_config.debug
        assert toml_config.max_connections == yaml_config.max_connections