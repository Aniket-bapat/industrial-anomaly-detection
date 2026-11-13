
from pathlib import Path

base = Path("dataset/bottle")

for split in ["train", "test", "ground_truth"]:
    print(f"\n{split.upper()}")

    split_path = base / split

    for category in sorted(split_path.iterdir()):
        if category.is_dir():
            images = list(category.glob("*.png"))
            print(f"{category.name}: {len(images)} images")
