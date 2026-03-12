# Agents Guidelines

## Python

- Always use `uv` for Python package management and virtual environments.
- Use `uv venv` to create virtual environments.
- Use `uv pip install` to install packages.
- Use `uv run` to execute Python scripts within the venv.
- Never use bare `pip`, `pip3`, or `python -m venv`. Always go through `uv`.
