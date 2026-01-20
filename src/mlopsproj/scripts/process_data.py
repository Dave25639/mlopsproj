import shutil
from pathlib import Path

from PIL import Image

IMG_SIZE = 224

def process_image(
    src_path: Path,
    dest_path: Path,
    overwrite: bool = False,
):
    if dest_path.exists() and not overwrite:
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.convert("RGB")

        w, h = img.size
        short = min(w, h)

        # center crop to square
        left = (w - short) // 2
        top = (h - short) // 2
        img = img.crop((left, top, left + short, top + short))

        # resize to 224x224
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        img.save(dest_path, format="JPEG", quality=95)


def preprocess_data(
    base_dir="/workspaces/mlopsproj/data",
    overwrite=False,
):
    base = Path(base_dir)

    src_images = base / "raw" / "images"
    meta_dir = base / "meta"
    out_images = base / "processed" / "images"

    for split in ["train", "test"]:
        split_file = meta_dir / f"{split}.txt"

        with open(split_file) as f:
            samples = [line.strip() for line in f if line.strip()]

        for sample in samples:
            label, file_id = sample.split("/")

            src_path = src_images / label / f"{file_id}.jpg"
            dest_path = out_images / label / f"{file_id}.jpg"

            if not src_path.exists():
                print(f"Missing: {src_path}")
                continue

            process_image(
                src_path,
                dest_path,
                overwrite=overwrite,
            )


if __name__ == "__main__":
    preprocess_data(overwrite=False)
