import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

def make_family_split(category, data_root, manifest_dir, val_fraction=0.2, seed=42):
    """Train/val manifests where an MVTec normal image and its MIRAGE images stay in the same split."""
    data_root, manifest_dir = Path(data_root), Path(manifest_dir)

    # finding the folders with the normal MVTec images and the MIRAGE images,
    # and for each MIRAGE image finding the MVTec image it was generated from (its family)
    normal_dir = data_root / "mvtec" / category / "train" / "good"
    mirage_dir = data_root / "mirage" / category

    tracking = pd.read_csv(data_root / "mirage" / "generation_tracking.csv")
    category_tracking = tracking[tracking["object"] == category]
    mirage_to_source = {Path(img).stem: Path(base).stem for img, base in zip(category_tracking["img_path"], category_tracking["base_img_path"])}

    # adding one row for each normal MVTec training image
    # a normal image is its own family (family = image name, e.g. "132")
    samples, empty_masks = [], []
    for path in sorted(normal_dir.glob("*.png")):
        if not path.name.startswith("._"): # skipping macOS metadata files, they are not images
            samples.append({"sample_id": f"mvtec_{path.stem}", "source": "mvtec",
                            "image_path": str(path.relative_to(data_root)), "family": path.stem})

    # adding one row for each MIRAGE image, in the family of its source MVTec image
    for path in sorted((mirage_dir / "images").glob("*.png")):
        if path.name.startswith("._"):
            continue
        # skipping images with an empty mask. The defect is not marked, so the image would teach a wrong label
        mask = np.array(Image.open(mirage_dir / "masks" / f"{path.stem}_mask.png").convert("L")) > 127
        if not mask.any():
            empty_masks.append(path.stem)
            continue
        samples.append({"sample_id": f"mirage_{path.stem}", "source": "mirage",
                        "image_path": str(path.relative_to(data_root)), "family": mirage_to_source[path.stem]})
    samples_table = pd.DataFrame(samples)

    # Splitting by family, not by image. Sorting the families (so the same seed always
    # gives the same split), shuffling them, and putting 20% of the families in validation.
    # Every image goes to the split of its family, so related images never cross the split
    families = sorted(samples_table["family"].unique())
    random.Random(seed).shuffle(families)
    val_families = set(families[: round(len(families) * val_fraction)])
    samples_table["split"] = np.where(samples_table["family"].isin(val_families), "val", "train")

    # safety check (no family in both splits), then save one manifest per split
    families_in_both = (samples_table.groupby("family")["split"].nunique() > 1).sum()
    assert families_in_both == 0

    output_dir = manifest_dir / category
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        split_table = samples_table[samples_table["split"] == split].drop(columns="split")
        split_table.to_csv(output_dir / f"{split}_manifest_family.csv", index=False)
        print(f"{category} {split}: {split_table['family'].nunique()} families, {len(split_table)} images "
              f"({(split_table.source == 'mvtec').sum()} MVTec + {(split_table.source == 'mirage').sum()} MIRAGE)")
    print("Excluded empty masks:", empty_masks)
    print("Families in both splits:", families_in_both)
    return samples_table


    # training dataset
class SupervisedSegmentationDataset(Dataset):
    """Real normal MVTec images and MIRAGE synthetic anomalies, each with a pixel mask."""

    def __init__(self, manifest_path, data_root, img_size=(256, 256)):
        self.samples = pd.read_csv(manifest_path)
        self.data_root = Path(data_root)
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    # loading the image
    def __getitem__(self, index):
        row = self.samples.iloc[index]
        image_path = self.data_root / row["image_path"]
        image = TF.to_tensor(Image.open(image_path).convert("RGB"))

        # making the mask (MIRAGE images have a defect mask, normal MVTec images get an empty mask with no defect)
        if row["source"] == "mirage":
            mask_path = image_path.parent.parent / "masks" / f"{image_path.stem}_mask.png"
            mask = np.array(Image.open(mask_path).convert("L")) > 127
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2])

        # resizing the image and the mask to the same size
        image = TF.resize(image, list(self.img_size), antialias=True)
        mask = TF.resize(mask, list(self.img_size), interpolation=InterpolationMode.NEAREST)
        return image, mask.long()


