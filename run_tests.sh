#!/bin/bash
# Simple test runner script

echo "🧪 Running confocal tests..."

# Install in editable mode with test dependencies
pip install -e ".[test]"

# Run tests with coverage
pytest tests/ -v --cov=confocal --cov-report=term-missing --cov-report=html

echo "✅ Tests complete! Coverage report in htmlcov/index.html"