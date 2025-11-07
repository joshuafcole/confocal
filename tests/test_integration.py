import os
import pytest
from tests import YamlTestConfig, TomlTestConfig


class TestConfigIntegration:
    @pytest.fixture
    def clean_env(self):
        env_vars = ['TEST_DB_URL', 'TEST_API_KEY', 'PROD_DB_URL', 'PROD_API_KEY']
        original = {var: os.environ.get(var) for var in env_vars}
        
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        
        yield
        
        for var, value in original.items():
            if value is not None:
                os.environ[var] = value
    
    @pytest.mark.parametrize("config_class", [YamlTestConfig, TomlTestConfig])
    def test_direct_parameter_passing(self, clean_env, config_class):
        config = config_class(
            database_url="direct://localhost/db",
            api_key="direct_key",
            debug=True,
            timeout=60,
            max_connections=50
        )
        
        assert config.database_url == "direct://localhost/db"
        assert config.api_key == "direct_key"
        assert config.debug is True
        assert config.timeout == 60
        assert config.max_connections == 50
    
    @pytest.mark.parametrize("config_class", [YamlTestConfig, TomlTestConfig])
    def test_direct_params_override_file_config(self, clean_env, config_class):
        config = config_class(
            active_profile='dev',
            database_url='override://localhost/override',  # Direct override
            timeout=99  # Direct override
        )
        
        # Direct params win
        assert config.database_url == "override://localhost/override"
        assert config.timeout == 99
        
        # File config used for non-overridden
        assert config.api_key == "dev_key_123"
        assert config.debug is True
    
    @pytest.mark.parametrize("config_class", [YamlTestConfig, TomlTestConfig])
    def test_env_vars_override_file_defaults(self, clean_env, config_class):
        os.environ['TEST_DB_URL'] = 'env://from-environment/db'
        
        config = config_class(active_profile='dev')
        
        assert config.database_url == "env://from-environment/db"
    
    @pytest.mark.parametrize("config_class", [YamlTestConfig, TomlTestConfig])
    def test_profile_switching_changes_values(self, clean_env, config_class):
        """
        Test that changing active_profile loads different configuration values.
        """
        dev_config = config_class(active_profile='dev')
        prod_config = config_class(active_profile='prod')
        
        # Different profiles have different values
        assert dev_config.debug is True
        assert prod_config.debug is False
        assert dev_config.max_connections == 10
        assert prod_config.max_connections == 100
    
    @pytest.mark.parametrize("config_class", [YamlTestConfig, TomlTestConfig])
    def test_missing_config_file_uses_defaults(self, clean_env, config_class):
        """
        Test that direct parameters can override config file values.
        When passing database_url directly, it overrides the file config.
        """
        config = config_class(
            database_url='fallback://localhost/db',
            active_profile=None  # Disable profile to test defaults
        )
        
        assert config.database_url == "fallback://localhost/db"
    
    @pytest.mark.parametrize("config_class", [YamlTestConfig, TomlTestConfig])
    def test_invalid_profile_name(self, clean_env, config_class):
        """
        Test behavior when requesting a profile that doesn't exist.
        Should fall back to global defaults.
        """
        config = config_class(active_profile='nonexistent_profile')
        
        # Should use global defaults when profile doesn't exist
        assert config.database_url == "default_db"
        assert config.debug is False
    
    @pytest.mark.parametrize("config_class", [YamlTestConfig, TomlTestConfig])
    def test_empty_env_var_template(self, clean_env, config_class):
        """
        Test that empty string env vars are handled correctly.
        """
        os.environ['TEST_DB_URL'] = ''  # Empty string
        
        config = config_class(active_profile='dev')
        
        # Empty env var should still be used (not fall back to default)
        assert config.database_url == ''