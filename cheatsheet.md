# Cheatsheet of Useful Commands

Use this markdown file to find useful terminal commands for each segment of the MLOps pipeline.

## Table of Contents
- [Styling](#styling)

## Virtual Environment

If a developer add a package to their uv environment using `uv add`, `pyproject.toml` should be updated automatically and then pushed to GitHub by that dev. Then when another dev does `git pull` they will receive the updated `.toml` file and they can do:

```bash
uv sync
```
to make their local uv environment match what the `.toml` is describing.

## Styling

Use the following to check for styling errors and fix them automatically.

```bash
uv run ruff check . #--fix
uv run ruff format .
```

## DVC

```bash
```