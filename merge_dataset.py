import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

# Define target output path
OUTPUT_DIR = Path("./merged_dataset")

# Define class mapping: map original class names/IDs from each dataset to 0 (dry) or 1 (wet)
# Example: map original names to target class ID (0: dry, 1: wet)
CLASS_MAPPING = {
    # Dataset 1 & 2 class mapping
    "plastic": 0, "paper": 0, "metal": 0, "glass": 0, "cardboard": 0, "dry": 0,
    "food": 1, "organic": 1, "wet": 1, "bio": 1
}

def process_dataset(dataset_path, dataset_yaml_path):
    with open(dataset_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    orig_names = data_cfg.get('names', {})
    if isinstance(orig_names, list):
        orig_names = {i: name for i, name in enumerate(orig_names)}

    for split in ['train', 'val', 'test']:
        img_dir = Path(dataset_path) / split / 'images'
        label_dir = Path(dataset_path) / split / 'labels'
        
        if not img_dir.exists():
            continue

        out_img_dir = OUTPUT_DIR / split / 'images'
        out_lbl_dir = OUTPUT_DIR / split / 'labels'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_file in tqdm(list(img_dir.glob('*.*')), desc=f"Processing {dataset_path} - {split}"):
            lbl_file = label_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue

            # Copy image with unique prefix to avoid filename collisions
            new_stem = f"{Path(dataset_path).stem}_{img_file.stem}"
            shutil.copy(img_file, out_img_dir / f"{new_stem}{img_file.suffix}")

            # Read and remap labels
            new_lines = []
            with open(lbl_file, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    orig_name = orig_names.get(cls_id)

                    if orig_name in CLASS_MAPPING:
                        new_cls_id = CLASS_MAPPING[orig_name]
                        new_lines.append(f"{new_cls_id} {' '.join(parts[1:])}\n")

            with open(out_lbl_dir / f"{new_stem}.txt", 'w') as out_lf:
                out_lf.writelines(new_lines)

# Process each source dataset folder
process_dataset("./dataset1", "./dataset1/data.yaml")
process_dataset("./dataset2", "./dataset2/data.yaml")

# Generate unified data.yaml
yaml_content = {
    'path': str(OUTPUT_DIR.resolve()),
    'train': 'train/images',
    'val': 'val/images',
    'names': {0: 'dry', 1: 'wet'}
}
with open(OUTPUT_DIR / 'data.yaml', 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print("Datasets successfully merged!")
