# Cheatsheet of Useful Commands

Use this markdown file to find useful terminal commands for each segment of the MLOps pipeline.

## Table of Contents
- [Virtual Environment](#virtual-environment)
- [Version History and Information]()
- [Styling](#styling)
- [DVC](#dvc)

## Version History and Information

To switch to a version listed below:

```bash
git checkout v0.0
uv run dvc checkout
```

To switch back to master branch after done inspecting previous version:
```
git checkout master
```

To see all tags or branches:
```bash
git tags
git branches
```

To push a new tag to remote:
```bash
git push origin <v0.0> # single tag
git push origin --tags # all tags
```

### v0.0
Configured everything up to the end of M8 properly. DVC is also linked and set up with a test image to start with. Further versions will include the full dataset we plan to use.

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

When using DVC for the first time, you'll have to set your credentials. To find them go to the dagshub repository for the project (https://dagshub.com/Dave25639/mlopsproj) and click the green "Remote" dropdown -> Data -> DVC -> Setup credentials. Remember to click the eye icon so it actually shows you your token then copy and run like below:

```bash
dvc remote modify origin --local access_key_id your_token
dvc remote modify origin --local secret_access_key your_token
```

Now you should be able to pull the dataset with:
```bash
uv run dvc pull
```

If you have added or modified files in a dataset, you should run

```bash
uv run dvc add data/raw
```

for example to update the dvc metadata file with the new changes. Then make sure to push the new metadata file to github and run

```bash
uv run dvc push
```

to push the changes and new data files to dagshub.
