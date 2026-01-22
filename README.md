# Project Checklist

> **Note:** All lists are _exhaustive_, meaning you are **not expected to complete every single point** for the exam.
> The module related to each item is shown in parentheses.

---

## Week 1

- [x] Create a git repository (M5)
- [x] Make sure that all team members have write access to the GitHub repository (M5)
- [x] Create a dedicated environment for your project to keep track of packages (M2)
- [x] Create the initial file structure using **cookiecutter** with an appropriate template (M6)
- [x] Fill out `data.py` to download and preprocess required data (if necessary) (M6)
- [x] Add a model to `model.py` and a training procedure to `train.py`, and get it running (M6)
- [x] Keep dependencies up to date in:
  - [x] `requirements.txt` / `requirements_dev.txt`, **or**
  - [x] `pyproject.toml` / `uv.lock` (M2 + M6)
- [x] Comply with good coding practices (PEP8) (M7)
- [x] Document essential parts of the code (docstrings/comments) (M7)
- [x] Set up version control for all or part of the data (M8)
- [x] Add command-line interfaces (CLI) and project commands where appropriate (M9)
- [x] Construct one or more Dockerfiles (M10)
- [x] Build Docker images locally and verify they work (M10)
- [x] Write one or more configuration files for experiments (M11)
- [x] Use **Hydra** for configuration loading and hyperparameter management (M11)
- [x] Use profiling to optimize performance-critical code (M12)
- [x] Use logging to log important events (M14)
- [x] Use **Weights & Biases** to log training progress and artifacts (M14)
- [x] Consider running a hyperparameter sweep (M14)
- [x] Use **PyTorch Lightning** (if applicable) to reduce boilerplate (M15)

---

## Week 2

- [x] Write unit tests for the data pipeline (M16)
- [x] Write unit tests for model construction and/or training (M16)
- [x] Calculate code coverage (M16)
- [x] Set up continuous integration (CI) on GitHub (M17)
- [ ] Add caching and multi-OS / multi-Python / multi-PyTorch testing to CI (M17)
- [x] Add linting to the CI pipeline (M17)
- [x] Add **pre-commit hooks** (M18)
- [ ] Add a workflow that triggers when data changes (M19)
- [ ] Add a workflow that triggers when the model registry changes (M19)
- [ ] Create a GCP Bucket for data storage and link it to data version control (M21)
- [ ] Create a workflow to automatically build Docker images (M21)
- [ ] Run model training on GCP (Engine or Vertex AI) (M21)
- [ ] Create a **FastAPI** application for model inference (M22)
- [ ] Deploy the model using **GCP Functions** or **Cloud Run** (M23)
- [ ] Write API tests and integrate them into CI (M24)
- [ ] Load test the deployed application (M24)
- [ ] Create a specialized ML deployment API using **ONNX**, **BentoML**, or both (M25)
- [ ] Create a frontend for the API (M26)

---

## Week 3

- [ ] Evaluate robustness to data drift (M27)
- [ ] Collect input-output data from the deployed application (M27)
- [ ] Deploy a drift detection API to the cloud (M27)
- [ ] Instrument the API with system metrics (M28)
- [ ] Set up cloud monitoring for the instrumented application (M28)
- [ ] Create alerting systems in GCP for incorrect behavior (M28)
- [ ] Optimize data loading using distributed data loading (if applicable) (M29)
- [ ] Optimize training using distributed training (if applicable) (M30)
- [ ] Experiment with quantization, compilation, and pruning to speed up inference (M31)

---

## Extra

- [ ] Write documentation for the application (M32)
- [ ] Publish documentation using **GitHub Pages** (M32)
- [ ] Revisit the initial project description — did it turn out as expected?
- [ ] Create an architectural diagram of the MLOps pipeline
- [ ] Ensure all group members understand all parts of the project
- [ ] Upload all code to GitHub

````markdown
# mlopsproj

nicolai

## Project structure

The directory structure of the project looks like this:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps)..
````
