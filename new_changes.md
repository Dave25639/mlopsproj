Main Config File Changes
File: configs/config.yaml
Added:
yamldefaults:
  - data: data
  - model: model
  - train: train
  - logging: logging
  - _self_              # Added for proper composition order

2. Train Script Changes
Fixed Hydra Decorator
File: src/mlopsproj/train.py (Line 160)
Changed from:
python@hydra.main(version_base="1.3", config_path="configs", config_name="config")
Changed to:
python@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
Reason: When running as a module (python -m mlopsproj.train), the path must be relative to the module location (src/mlopsproj/).
Fixed Config Access Paths
File: src/mlopsproj/train.py
Changed multiple lines from:
pythoncfg.model.output_dir          # ❌ Wrong
cfg.model.model_name          # ❌ Wrong
cfg.model.learning_rate       # ❌ Wrong
Changed to:
pythoncfg.output_dir                           # ✅ Correct (top-level)
cfg.model.architecture.model_name        # ✅ Correct
cfg.model.architecture.learning_rate     # ✅ Correct
Reason: Config hierarchy must match the YAML structure.

3. Configuration Value Fixes
Model Config
File: configs/model/model.yaml
Changed:
yamlarchitecture:
  freeze: true              # ❌ Wrong key name
  warmup_steps: 1          # ❌ Too low
  max_steps: 1             # ❌ Too low
  
checkpointing:
  dirpath: ""              # ❌ Empty
To:
yamlarchitecture:
  freeze_backbone: true    # ✅ Matches code expectation
  warmup_steps: 500        # ✅ Reasonable value
  max_steps: null          # ✅ Calculated automatically
  
checkpointing:
  dirpath: "checkpoints"   # ✅ Valid path
Train Config
File: configs/train/train.yaml
Changed:
yamlaccumulate_grad_batches: true    # ❌ Should be integer
fast_dev_run: True               # ⚠️  Quick test mode
To:
yamlaccumulate_grad_batches: 1       # ✅ Valid integer
fast_dev_run: False              # ✅ For real training

4. Docker Configuration
Fixed Dockerfile
File: dockerfiles/train.dockerfile
Changed ENTRYPOINT from:
dockerfileENTRYPOINT ["uv", "run", "src/mlopsproj/train.py"]
To:
dockerfileENTRYPOINT ["uv", "run", "python", "-m", "mlopsproj.train"]
Added:
dockerfile# Copy configs directory (needed for Hydra)
COPY configs configs/
Reason: Running as a module enables relative imports and proper Hydra config discovery.

5. Data Pipeline
Created Data Organization Script
File: src/mlopsproj/organize_data_script.py
Modified to create proper structure:
pythonprocessed_dir = base / "processed" / "images"  # Added /images/ level
Result: Data organized as:
data/
├── raw/
│   └── images/
│       ├── apple_pie/
│       └── ...
└── processed/
    └── images/
        ├── train/
        │   ├── apple_pie/
        │   └── ...
        └── test/
            ├── apple_pie/
            └── ...
Fixed Image Transforms
File: src/mlopsproj/data.py
Added missing transforms:
pythondef build_transforms(train: bool = False):
    transforms.Compose([
        transforms.Resize(256),         # ✅ Added
        transforms.CenterCrop(224),     # ✅ Added
        # ... rest of transforms
    ])
Reason: All images must be same size (224x224) for ViT model.

6. Additional Fixes
Mac-Specific Issue
Problem: MPS backend incompatibility with multi-worker DataLoader
Solution:
bashuv run python -m mlopsproj.train num_workers=0
Or permanently set in configs/config.yaml:
yamlnum_workers: 0

How to Use
Run with Default Config:
bashuv run python -m mlopsproj.train
Override Config Values:
bashuv run python -m mlopsproj.train \
  num_workers=0 \
  train.fast_dev_run=False \
  train.max_epochs=10 \
  batch_size=64 \
  model.architecture.learning_rate=0.001
Run in Docker:
bashdocker build -t mlops_train -f dockerfiles/train.dockerfile .
docker run --rm -v $(pwd)/data:/data mlops_train data.data_dir=/data
View Config:
bashuv run python -m mlopsproj.train --print-config

Key Learnings

Config paths are relative to the script location when using module execution
Config hierarchy in code must match YAML structure (e.g., cfg.model.architecture.x)
Docker requires module execution (-m) for relative imports to work
Hydra allows runtime config overrides from command line
All images must be resized to consistent dimensions before batching
Mac MPS requires num_workers=0 in DataLoader for stability
