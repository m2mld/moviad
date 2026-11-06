import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.configurations import Split


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


def pick_per_defect_type(anomalous, n_pick, rng):
    """Choose n_pick rows from the anomalous images, with each defect type represented in proportion to its size."""
    counts = anomalous["label"].value_counts().sort_index()
    exact = counts / counts.sum() * n_pick
    quota = np.floor(exact).astype(int)
    # the images left after rounding down go to the defect types with the largest remainders
    for defect_type in (exact - quota).sort_values(ascending=False).index[: n_pick - quota.sum()]:
        quota[defect_type] += 1

    chosen = []
    for defect_type, n_type in quota.items():
        rows = anomalous.index[anomalous["label"] == defect_type]
        chosen.extend(rng.choice(rows, size=n_type, replace=False))
    return sorted(chosen)


def make_real_anomaly_split(category, data_root, manifest_dir, n_test_anomalous=50, val_fraction=0.2,
                            min_train_anomalous=30, seed=42):
    """Scenario 2 split of the real MVTec AD test images of one category.

    n_test_anomalous anomalous images stay in the test set (chosen per defect type); the other anomalous
    images go to training, and val_fraction of them to validation. Normal test images are subsampled so that
    the reduced test set keeps the original normal/anomalous ratio. Returns None if fewer than
    min_train_anomalous anomalous images would be left for training (the category is excluded).
    """
    data_root, manifest_dir = Path(data_root), Path(manifest_dir)
    rng = np.random.default_rng(seed)

    # loading the test images with MoViAD (label = defect type, or "good"; mask_path for anomalous images)
    arguments = DatasetArguments(dataset_path=str(data_root / "mvtec"), img_size=(256, 256), gt_mask_size=(256, 256))
    test_samples = MVTecDataset(arguments, category=category, split=Split.TEST).samples
    normal = test_samples[test_samples["label"] == "good"]
    anomalous = test_samples[test_samples["label"] != "good"]

    # checking that enough anomalous images are left for training
    n_left = len(anomalous) - n_test_anomalous
    if n_left < min_train_anomalous:
        print(f"{category}: only {n_left} anomalous images left for training (minimum {min_train_anomalous}), excluded")
        return None

    # choosing the anomalous test images, then the validation images among the remaining ones (both per defect type)
    test_anomalous = anomalous.loc[pick_per_defect_type(anomalous, n_test_anomalous, rng)]
    remaining = anomalous.drop(test_anomalous.index)
    val_anomalous = remaining.loc[pick_per_defect_type(remaining, round(len(remaining) * val_fraction), rng)]
    train_anomalous = remaining.drop(val_anomalous.index)

    # keeping the original normal/anomalous ratio in the reduced test set; the other normal test images are not used
    n_test_normal = round(len(normal) * n_test_anomalous / len(anomalous))
    test_normal = normal.loc[sorted(rng.choice(normal.index, size=n_test_normal, replace=False))]

    # saving one manifest per part, with paths relative to data_root (same convention as the family split)
    output_dir = manifest_dir / category / f"scenario2_test{n_test_anomalous}"
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = {"test": pd.concat([test_normal, test_anomalous]), "train_real": train_anomalous, "val_real": val_anomalous}
    for name, part in parts.items():
        table = pd.DataFrame({
            "source": "mvtec_test",
            "defect_type": part["label"],
            "image_path": [str(Path(path).relative_to(data_root)) for path in part["image_path"]],
            "mask_path": [str(Path(path).relative_to(data_root)) if path else "" for path in part["mask_path"]],
        })
        table.to_csv(output_dir / f"{name}_manifest.csv", index=False)
    print(f"{category}: test {n_test_normal} normal + {len(test_anomalous)} anomalous, "
          f"training {len(train_anomalous)} anomalous, validation {len(val_anomalous)} anomalous, "
          f"{len(normal) - n_test_normal} normal test images not used")
    return parts


    # training dataset
class SupervisedSegmentationDataset(VADDataset):
    """Real normal MVTec images and MIRAGE synthetic anomalies, each with a pixel mask."""

    def __init__(self, dataset_arguments, category, split, manifest_path):
        super().__init__(dataset_arguments, category, split)
        self.samples = pd.read_csv(manifest_path)
        self.data_root = Path(dataset_arguments.dataset_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        image_path = self.data_root / row["image_path"]

        # loading the image (VADDataset's transform converts it to a tensor and resizes it)
        image = self.transform_image(Image.open(image_path).convert("RGB"))

        # making the mask: MIRAGE images have a defect mask next to the image, real anomalous MVTec test images
        # (Scenario 2) have the mask path in the manifest, normal MVTec images get an empty mask (no defect)
        if row["source"] == "mirage":
            mask_path = image_path.parent.parent / "masks" / f"{image_path.stem}_mask.png"
            mask = self.transform_mask(Image.open(mask_path).convert("L")) > 0.5
        elif row["source"] == "mvtec_test":
            mask_path = self.data_root / row["mask_path"]
            mask = self.transform_mask(Image.open(mask_path).convert("L")) > 0.5
        else:
            mask = torch.zeros(1, *self.dataset_arguments.gt_mask_size, dtype=torch.bool)

        # training returns (image, mask); validation returns the format of MoViAD's Evaluator
        if self.split == Split.TRAIN:
            return image, mask.long()
        label = int(mask.any())
        return image, label, mask.long(), str(row["image_path"])

