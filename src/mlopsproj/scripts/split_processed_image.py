from pathlib import Path
import shutil


def split_processed_images(
    base_dir="data",
    overwrite=False,
):
    base = Path(base_dir)

    images_root = base / "processed" / "images"
    meta_dir = base / "meta"

    for split in ["train", "test"]:
        split_file = meta_dir / f"{split}.txt"
        split_root = images_root / split

        with open(split_file) as f:
            samples = [line.strip() for line in f if line.strip()]

        for sample in samples:
            cls, file_id = sample.split("/")

            src = images_root / cls / f"{file_id}.jpg"
            dst = split_root / cls / f"{file_id}.jpg"

            if not src.exists():
                print(f"[WARN] Missing source: {src}")
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                if overwrite:
                    dst.unlink()
                else:
                    continue

            shutil.move(src, dst)

    # OPTIONAL: clean up empty class dirs
    for cls_dir in images_root.iterdir():
        if cls_dir.is_dir() and cls_dir.name not in {"train", "test"}:
            try:
                cls_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    split_processed_images(
        base_dir="data",
        overwrite=False,
    )
