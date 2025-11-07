"""Tests for YAML configuration loading with env var templating."""
import os
import pytest
from tests import YamlTestConfig


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