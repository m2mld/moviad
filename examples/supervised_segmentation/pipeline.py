"""Supervised segmentation for visual anomaly detection in MoViAD.

Data layout (symbolic links):
    DATA_ROOT/mvtec  -> MVTec AD      (<category>/train, test, ground_truth)
    DATA_ROOT/mirage -> MIRAGE mvtec  (<category>/images, masks)
"""
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.models.supervised_segmentation.supervised_segmentation import SegmentationTrainArgs, SupervisedSegmentation
from moviad.trainers.trainer import Trainer
from moviad.trainers.trainer_early_stopping import EarlyStoppingTrainer
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import Dice, F1, IoU, MetricLvl, RocAuc

# paths
HOME = Path.home()
DATA_ROOT = HOME / "datasets" / "thesis_data"
TRACKING_CSV = DATA_ROOT / "mirage" / "generation_tracking.csv"
RUNS_DIR = HOME / "thesis_runs"
MANIFEST_DIR = RUNS_DIR / "manifests"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# metrics
def all_metrics():
    return [RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL), F1(MetricLvl.IMAGE), F1(MetricLvl.PIXEL), IoU(), Dice()]


# family split
def make_family_split(category, val_fraction=0.2, seed=42):
    """Train/val manifests where an MVTec normal image and its MIRAGE images stay in the same split."""

    # finding the folders with the normal MVTec images and the MIRAGE images,
    # and for each MIRAGE image finding the MVTec image it was generated from (its family)
    normal_dir = DATA_ROOT / "mvtec" / category / "train" / "good"
    mirage_dir = DATA_ROOT / "mirage" / category

    tracking = pd.read_csv(TRACKING_CSV)
    category_tracking = tracking[tracking["object"] == category]
    mirage_to_source = {Path(img).stem: Path(base).stem for img, base in zip(category_tracking["img_path"], category_tracking["base_img_path"])}

    # adding one row for each normal MVTec training image
    # a normal image is its own family (family = image name, e.g. "132")
    samples, empty_masks = [], []
    for path in sorted(normal_dir.glob("*.png")):
        if not path.name.startswith("._"): # skipping macOS metadata files, they are not images
            samples.append({"sample_id": f"mvtec_{path.stem}", "source": "mvtec",
                            "image_path": str(path.relative_to(DATA_ROOT)), "family": path.stem})

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
                        "image_path": str(path.relative_to(DATA_ROOT)), "family": mirage_to_source[path.stem]})
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

    output_dir = MANIFEST_DIR / category
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


class ValidationDataset(Dataset):
    """Validation images in the format MoViAD's Evaluator expects: (image, label, mask, path)."""

    def __init__(self, manifest_path, data_root, img_size=(256, 256)):
        self.base = SupervisedSegmentationDataset(manifest_path, data_root, img_size)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, mask = self.base[index]
        label = int(mask.any())
        path = str(self.base.samples.iloc[index]["image_path"])
        return image, label, mask, path

# helpers
def test_dataset_for(category, size):
    args = DatasetArguments(dataset_path=str(DATA_ROOT / "mvtec"), img_size=size, gt_mask_size=size)
    return MVTecDataset(args, category=category, split=Split.TEST)


def measure_inference(model, loader, n_images, device):
    """Average time per image and peak GPU memory, model only (no metrics)."""
    model.eval()
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for images, *_ in loader:
            model(images.to(device))
    if use_cuda:
        torch.cuda.synchronize()
    ms = (time.time() - start) * 1000 / n_images
    vram = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None
    return ms, vram


def save_run(name, category, seed, model, summary, save, last_state=None, history=None):
    """model.pt holds the model that was evaluated (the best epoch when early stopping is used)."""
    print(json.dumps(summary, indent=2))
    if not save:
        print("Test run: nothing saved.")
        return None
    run_dir = RUNS_DIR / "runs" / f"{name}_{category}_seed{seed}_{datetime.now():%Y%m%d_%H%M}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    if last_state is not None:
        torch.save(last_state, run_dir / "last_model.pt")
    if history is not None:
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved to:", run_dir)
    return run_dir


# supervised segmentation (any smp architecture)
def train_and_evaluate_segmentation(category, arch="pspnet", seed=42, batch_size=8, max_epochs=50, patience=5,
                                    device="cuda", subset_size=None, save=True, encoder_weights="imagenet"):
    """Train on the family train split, select the best epoch on the family validation split
    (pixel max-F1, early stopping), then evaluate the best model on the MVTec AD test set.
    subset_size: use only the first N training images (for a quick test)."""

    # loading the data (train, validation, test) and choosing the metrics
    device = torch.device(device)
    train_dataset = SupervisedSegmentationDataset(MANIFEST_DIR / category / "train_manifest_family.csv", DATA_ROOT)
    val_dataset = ValidationDataset(MANIFEST_DIR / category / "val_manifest_family.csv", DATA_ROOT)
    if subset_size is not None:
        train_dataset = Subset(train_dataset, list(range(subset_size)))
    test_dataset = test_dataset_for(category, (256, 256))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    metrics = all_metrics()

    # creating the model and the trainer (with early stopping)
    set_seed(seed)
    model = SupervisedSegmentation(arch, encoder_weights=encoder_weights).to(device)
    train_args = SegmentationTrainArgs(batch_size=batch_size, epochs=max_epochs, evaluation_epoch_interval=1)
    trainer = EarlyStoppingTrainer(train_args, model, train_dataset, val_dataset, [F1(MetricLvl.PIXEL)], device,
                                   patience=patience)

    # training, measuring the training time and the peak GPU memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_minutes = (time.time() - start) / 60
    train_vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None

    # measuring the inference speed and evaluating the best model on the test set
    inference_ms, inference_vram = measure_inference(model, test_loader, len(test_dataset), device)
    report = Evaluator.evaluate(model, test_loader, metrics, device, postprocess=lambda x: x)

    # collecting all settings and results in one summary
    summary = {
        "model": f"{arch}_resnet50_smp", "training": "early_stopping", "category": category, "seed": seed,
        "split": "family",
        "train_images": len(train_dataset), "val_images": len(val_dataset), "test_images": len(test_dataset),
        "max_epochs": max_epochs, "patience": patience, "epochs_run": len(trainer.history),
        "best_epoch": trainer.best_epoch, "best_val_" + trainer.monitor: round(trainer.best_score, 4),
        "batch_size": batch_size, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "train_minutes": round(train_minutes, 2), "train_peak_vram_mb": train_vram and round(train_vram, 1),
        "inference_ms_per_image": round(inference_ms, 1), "inference_peak_vram_mb": inference_vram and round(inference_vram, 1),
        "metrics": {name: round(float(value), 4) for name, value in report.items()},
    }

    # saving the model and the results in the run folder
    save_run(arch, category, seed, model, summary, save, last_state=trainer.last_state, history=trainer.history)
    return summary


# Dinomaly
def train_and_evaluate_dinomaly(category, seed=42, batch_size=32, total_iterations=5000, device="cuda",
                                save=True):
    device = torch.device(device)
    args = DatasetArguments(dataset_path=str(DATA_ROOT / "mvtec"), img_size=(224, 224), gt_mask_size=(224, 224))
    train_dataset = MVTecDataset(args, category=category, split=Split.TRAIN)
    test_dataset = MVTecDataset(args, category=category, split=Split.TEST)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    metrics = all_metrics()

    iterations_per_epoch = math.ceil(len(train_dataset) / batch_size)
    epochs = max(1, total_iterations // iterations_per_epoch)
    print(f"{category}: {len(train_dataset)} normal images, {iterations_per_epoch} iterations/epoch, {epochs} epochs")

    set_seed(seed)
    model = Dinomaly("deit_tiny_patch16_224.fb_in1k").to(device)
    train_args = DinomalyTrainArgs(batch_size=batch_size, epochs=epochs, evaluation_epoch_interval=epochs + 1)
    trainer = Trainer(train_args, model, train_dataset, None, metrics, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_minutes = (time.time() - start) / 60
    train_vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None

    inference_ms, inference_vram = measure_inference(model, test_loader, len(test_dataset), device)
    report = Evaluator.evaluate(model, test_loader, metrics, device)  # default min-max normalisation

    summary = {
        "model": "dinomaly_deit_tiny", "category": category, "seed": seed,
        "train_images": len(train_dataset), "test_images": len(test_dataset),
        "epochs": epochs, "iterations": epochs * iterations_per_epoch, "batch_size": batch_size,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "train_minutes": round(train_minutes, 2), "train_peak_vram_mb": train_vram and round(train_vram, 1),
        "inference_ms_per_image": round(inference_ms, 1), "inference_peak_vram_mb": inference_vram and round(inference_vram, 1),
        "metrics": {name: round(float(value), 4) for name, value in report.items()},
    }
    save_run("dinomaly", category, seed, model, summary, save)
    return summary
