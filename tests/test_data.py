import pytest
import torch
from mlopsproj.data import Food101DataModule
from torch.utils.data import Dataset
from tests import _PATH_DATA

@pytest.fixture(scope="session")
def setup_food101_datamodule():
    dm = Food101DataModule(
        _PATH_DATA,
        batch_size=16,
        num_workers=2,
        img_size=224,
        data_fraction=0.1
    )
    dm.prepare_data()
    dm.setup()
    return dm

def test_food101_setup_creates_datasets(setup_food101_datamodule):
    dm = setup_food101_datamodule
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None

    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0
    assert len(dm.test_dataset) > 0

def test_dataloaders_use_correct_datasets(setup_food101_datamodule):
    dm = setup_food101_datamodule

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    assert train_loader.dataset is dm.train_dataset
    assert val_loader.dataset is dm.val_dataset
    assert test_loader.dataset is dm.test_dataset

def test_train_batch_structure(setup_food101_datamodule):
    dm = setup_food101_datamodule

    images, labels = next(iter(dm.train_dataloader()))

    assert images.ndim == 4
    assert labels.ndim == 1

    assert images.shape[0] <= dm.batch_size
    assert images.shape[1] == 3
    assert images.shape[2] == dm.img_size
    assert images.shape[3] == dm.img_size

    assert labels.dtype in (torch.int64, torch.long)
    assert labels.min().item() >= 0
    assert labels.max().item() < dm.num_classes

def test_images_are_normalized(setup_food101_datamodule):
    dm = setup_food101_datamodule

    images, _ = next(iter(dm.train_dataloader()))

    assert images.dtype == torch.float32
    assert images.max() <= 1.0 or images.max() <= 3.0
    assert images.min() >= -3.0

def test_data_fraction(setup_food101_datamodule):
    fraction = 0.1
    dm_frac = setup_food101_datamodule

    dm_full = Food101DataModule(_PATH_DATA)
    dm_full.setup()

    assert len(dm_frac.train_dataset) - int(len(dm_full.train_dataset) * fraction) < 2
    assert len(dm_frac.val_dataset) - int(len(dm_full.val_dataset) * fraction) < 2
    assert len(dm_frac.test_dataset) - int(len(dm_full.test_dataset) * fraction) < 2

def test_dataloader_works_in_loop(setup_food101_datamodule):
    dm = setup_food101_datamodule

    for batch in dm.train_dataloader():
        images, labels = batch
        break
