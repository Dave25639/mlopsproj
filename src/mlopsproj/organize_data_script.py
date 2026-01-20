import os
import shutil
from pathlib import Path

def split_data(base_dir="/workspaces/mlopsproj/data"):
    base = Path(base_dir)
    src_images = base / "raw" / "images"
    meta_dir = base / "meta"
    processed_dir = base / "processed"

    for split in ["train", "test"]:
        split_file = meta_dir / f"{split}.txt"
        dest_root = processed_dir / split

        with open(split_file, "r") as f:
            samples = [line.strip() for line in f.readlines() if line.strip()]

        for sample in samples:
            label, file_id = sample.split("/")
            src_path = src_images / f"{sample}.jpg"
            dest_dir = dest_root / label
            dest_path = dest_dir / f"{file_id}.jpg"

            dest_dir.mkdir(parents=True, exist_ok=True)
            if src_path.exists():
                shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    split_data()
