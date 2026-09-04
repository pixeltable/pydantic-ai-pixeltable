# Contributing

Thanks for your interest in contributing to `pydantic-ai-pixeltable`!

## Development Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

Tests use a real Pixeltable catalog and require no API keys.

## Code Style

```bash
ruff check .
ruff format .
```

## Pull Requests

1. Fork the repo and create a branch from `main`.
2. Add tests for any new functionality.
3. Run `ruff check . && ruff format --check .` and fix any issues.
4. Run `pytest tests/ -v` and ensure all tests pass.
5. Open a PR with a clear description.

## Releasing

Releases are published to PyPI automatically when a GitHub Release is created.

1. Bump version in `pydantic_ai_pixeltable/__init__.py` and `pyproject.toml`.
2. Commit and push to `main`.
3. Create a GitHub Release with a `v*` tag (e.g. `v0.1.0`).
