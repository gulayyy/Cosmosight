import os
import shutil
import random
from pathlib import Path
import json  # <-- added

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)

# Ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# (Optional) Cap Galaxy class (undersample) — reduces imbalance
CLASS_CAP = {"Galaxy": 8000, "Nebula": None, "Star": None}  # <-- added

# Valid extensions
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Paths relative to project root (safe based on this file's location)
ROOT = Path(__file__).resolve().parents[1]
raw_folder = ROOT / "data" / "processed_images"
base_dir   = ROOT / "data" / "split_dataset"

train_dir = base_dir / "train"
val_dir   = base_dir / "val"
test_dir  = base_dir / "test"

# Reset target folders (clean start)
if base_dir.exists():
    shutil.rmtree(base_dir)
for p in (train_dir, val_dir, test_dir):
    p.mkdir(parents=True, exist_ok=True)

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VALID_EXTS

summary = {}

for category in sorted(os.listdir(raw_folder)):
    category_path = raw_folder / category
    if not category_path.is_dir():
        continue

    # Get only images
    images = [p.name for p in map(Path, os.listdir(category_path))
              if is_image_file(category_path / p)]

    # Skip empty class
    if not images:
        print(f"⚠️  Skipped empty class: {category}")
        continue

    random.shuffle(images)

    # (Optional) Apply cap (random truncation after shuffle)
    cap = CLASS_CAP.get(category, None)  # <-- added
    if cap is not None and len(images) > cap:
        images = images[:cap]  # <-- added

    n = len(images)
    n_train = int(TRAIN_RATIO * n)
    n_val   = int(VAL_RATIO * n)
    n_test  = n - n_train - n_val

    # Ensure at least 1 sample for very small classes
    if n_val == 0 and n >= 2:
        n_val, n_test = 1, max(0, n_test - 1)
    if n_test == 0 and n >= 2:
        n_test, n_val = 1, max(0, n_val - 1)

    splits = [
        (images[:n_train], train_dir / category),
        (images[n_train:n_train+n_val], val_dir / category),
        (images[n_train+n_val:], test_dir / category),
    ]

    for files, dst_root in splits:
        dst_root.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src = category_path / fname
            dst = dst_root / fname
            shutil.copy2(src, dst)  # preserves metadata too

    summary[category] = {
        "total": n,
        "train": n_train,
        "val": len(splits[1][0]),
        "test": len(splits[2][0]),
    }

# Print summary
print("✅ Images successfully split into train/val/test folders.\n")
print("Class distribution:")
for cls, stats in summary.items():
    print(f" - {cls:10s}  total={stats['total']:6d} | "
          f"train={stats['train']:6d} | val={stats['val']:5d} | test={stats['test']:5d}")

print(f"\nFolder: {base_dir}")
print(f"Seed: {SEED}  (reproducible split)")

# Write class order from train folder (for evaluate/train consistency)  <-- added
class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
models_dir = ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)
class_names_path = models_dir / "CLASS_NAMES.json"
with open(class_names_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print("\nCLASS_NAMES written:", class_names_path)
