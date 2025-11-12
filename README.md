# Confocal

A multi-layer configuration management library built on pydantic with source tracking and profile support.

## Features

- Hierarchical config file discovery (searches parent directories)
- Profile support for different environments
- Full source tracking to explain where each config value came from
- Built on pydantic for robust validation and type safety
- Rich terminal output for config inspection

## Installation

```bash
pip install confocal
```

## Quick Start

```python
from confocal import BaseConfig
from pydantic import Field

class MyConfig(BaseConfig):
    database_url: str
    name: str = Field(default="Anonymous")
    debug: bool = False

# Load config from raiconfig.toml, environment variables, etc.
config = MyConfig.load()

# Show where config values came from
config.explain()
```

## Config Sources (in order of precedence)

1. Initialization arguments
2. Environment variables
3. Active profile from config file
4. Config files (YAML or TOML)
5. Default values

## Using Config Files

Confocal supports both TOML and YAML config files. Specify which format to use in your config class:

### Using TOML (default)
```python
class MyConfig(BaseConfig):
    model_config = SettingsConfigDict(
        toml_file="config.toml",
    )
```

### Using YAML
```python
class MyConfig(BaseConfig):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
    )
```

## Using Profiles

Profile support works with both TOML and YAML files.

**TOML example** (`raiconfig.toml`):
```toml
database_url = "postgresql://prod-db:5432"

[profile.dev]
database_url = "postgresql://localhost:5432"
debug = true

[profile.test]
database_url = "postgresql://test-db:5432"
```

**YAML example** (`config.yaml`):
```yaml
database_url: "postgresql://prod-db:5432"

profile:
  dev:
    database_url: "postgresql://localhost:5432"
    debug: true
  test:
    database_url: "postgresql://test-db:5432"
```

Activate a profile:
```bash
export ACTIVE_PROFILE=dev
```

### YAML Environment Variables

YAML configs support environment variable substitution:

```yaml
database_url: "{{ env_var('DB_URL', 'postgresql://localhost:5432') }}"
api_key: "{{ env_var('API_KEY') }}"  # Required, will error if not set
```

## Advanced Usage

Show full config inheritance chain:
```python
config.explain(verbose=True)
```

Manually find nearest config file in parent directories:
```python
from confocal import find_upwards
config_path = find_upwards("config.toml")
```

## License

MIT
