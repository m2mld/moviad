"""Supervised segmentation for visual anomaly detection in MoViAD.

Data layout (symbolic links):
    DATA_ROOT/mvtec  -> MVTec AD      (<category>/train, test, ground_truth)
    DATA_ROOT/mirage -> MIRAGE mvtec  (<category>/images, masks)
"""
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.trainers.trainer import Trainer
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import F1, Metric, MetricLvl, RocAuc

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
    torch.cuda.manual_seed_all(seed)


# metrics
class IoU(Metric):
    """Intersection over Union of the anomalous class at a fixed threshold, pooled over all pixels."""

    def __init__(self, level: MetricLvl = MetricLvl.PIXEL, threshold: float = 0.5):
        super().__init__(level)
        self.threshold = threshold

    @property
    def name(self):
        return f"{self.level.value}_iou"

    def compute(self, gt, pred):
        gt_binary = np.asarray(gt).flatten() > 0
        pred_binary = np.asarray(pred).flatten() >= self.threshold
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()
        return float(intersection / union) if union > 0 else 1.0


class Dice(Metric):
    """Dice coefficient (F1 at a fixed threshold) of the anomalous class, pooled over all pixels."""

    def __init__(self, level: MetricLvl = MetricLvl.PIXEL, threshold: float = 0.5):
        super().__init__(level)
        self.threshold = threshold

    @property
    def name(self):
        return f"{self.level.value}_dice"

    def compute(self, gt, pred):
        gt_binary = np.asarray(gt).flatten() > 0
        pred_binary = np.asarray(pred).flatten() >= self.threshold
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        total = gt_binary.sum() + pred_binary.sum()
        return float(2 * intersection / total) if total > 0 else 1.0


def all_metrics():
    return [RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL), F1(MetricLvl.PIXEL), IoU(), Dice()]


# family split
def make_family_split(category, val_fraction=0.2, seed=42):
    """Train/val manifests where an MVTec normal image and its MIRAGE images stay in the same split."""
    normal_dir = DATA_ROOT / "mvtec" / category / "train" / "good"
    mirage_dir = DATA_ROOT / "mirage" / category

    tracking = pd.read_csv(TRACKING_CSV)
    rows = tracking[tracking["object"] == category]
    source_of = {Path(img).stem: Path(base).stem for img, base in zip(rows["img_path"], rows["base_img_path"])}

    records, empty_masks = [], []
    for path in sorted(normal_dir.glob("*.png")):
        if not path.name.startswith("._"):
            records.append({"sample_id": f"mvtec_{path.stem}", "source": "mvtec",
                            "image_path": str(path.relative_to(DATA_ROOT)), "family": path.stem})
    for path in sorted((mirage_dir / "images").glob("*.png")):
        if path.name.startswith("._"):
            continue
        mask = np.array(Image.open(mirage_dir / "masks" / f"{path.stem}_mask.png").convert("L")) > 127
        if not mask.any():
            empty_masks.append(path.stem)
            continue
        records.append({"sample_id": f"mirage_{path.stem}", "source": "mirage",
                        "image_path": str(path.relative_to(DATA_ROOT)), "family": source_of[path.stem]})
    table = pd.DataFrame(records)

    families = sorted(table["family"].unique())
    random.Random(seed).shuffle(families)
    val_families = set(families[: round(len(families) * val_fraction)])
    table["split"] = np.where(table["family"].isin(val_families), "val", "train")

    families_in_both = (table.groupby("family")["split"].nunique() > 1).sum()
    assert families_in_both == 0

    output_dir = MANIFEST_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        part = table[table["split"] == split].drop(columns="split")
        part.to_csv(output_dir / f"{split}_manifest_family.csv", index=False)
        print(f"{category} {split}: {part['family'].nunique()} families, {len(part)} images "
              f"({(part.source == 'mvtec').sum()} MVTec + {(part.source == 'mirage').sum()} MIRAGE)")
    print("Excluded empty masks:", empty_masks)
    print("Families in both splits:", families_in_both)
    return table


# training dataset
class SupervisedSegmentationDataset(Dataset):
    """Real normal MVTec images and MIRAGE synthetic anomalies, each with a pixel mask."""

    def __init__(self, manifest_path, data_root, img_size=(256, 256)):
        self.samples = pd.read_csv(manifest_path)
        self.data_root = Path(data_root)
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        image_path = self.data_root / row["image_path"]
        image = TF.to_tensor(Image.open(image_path).convert("RGB"))
        if row["source"] == "mirage":
            mask_path = image_path.parent.parent / "masks" / f"{image_path.stem}_mask.png"
            mask = np.array(Image.open(mask_path).convert("L")) > 127
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2])
        image = TF.resize(image, list(self.img_size), antialias=True)
        mask = TF.resize(mask, list(self.img_size), interpolation=InterpolationMode.NEAREST)
        return image, mask.long()


# model
@dataclass
class SegmentationTrainArgs(TrainingArgs):
    """Training settings: which optimizer and which loss."""
    learning_rate: float = 0.01

    def init_train(self, model):
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate,
                                             momentum=0.9, weight_decay=0.0005)
        if self.loss_function is None:
            self.loss_function = F.cross_entropy


class SupervisedSegmentation(VADModel):
    """PSPNet trained with pixel masks, packaged so MoViAD can train and evaluate it."""

    def __init__(self, encoder_name="resnet50"):
        super().__init__()
        self.network = smp.PSPNet(encoder_name=encoder_name, encoder_weights="imagenet", classes=2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        logits = self.network((images - self.mean) / self.std)
        if self.training:
            return logits
        anomaly_maps = logits.softmax(dim=1)[:, 1]
        return anomaly_maps, anomaly_maps.amax(dim=(1, 2))

    def train_step(self, batch, training_args):
        images, masks = batch
        device = self.mean.device
        images, targets = images.to(device), masks[:, 0].to(device)
        logits = self(images)
        loss = training_args.loss_function(logits, targets)
        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()
        return loss.item()


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


def save_run(name, category, seed, model, summary, save):
    print(json.dumps(summary, indent=2))
    if not save:
        print("Test run: nothing saved.")
        return None
    run_dir = RUNS_DIR / "runs" / f"{name}_{category}_seed{seed}_{datetime.now():%Y%m%d_%H%M}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved to:", run_dir)
    return run_dir


# PSPNet
def train_and_evaluate_pspnet(category, seed=42, batch_size=8, epochs=20, device="cuda",
                              subset_size=None, save=True):
    """subset_size: use only the first N training images (for a quick test)."""
    device = torch.device(device)
    train_dataset = SupervisedSegmentationDataset(MANIFEST_DIR / category / "train_manifest_family.csv", DATA_ROOT)
    if subset_size is not None:
        train_dataset = Subset(train_dataset, list(range(subset_size)))
    test_dataset = test_dataset_for(category, (256, 256))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    metrics = all_metrics()

    set_seed(seed)
    model = SupervisedSegmentation().to(device)
    train_args = SegmentationTrainArgs(batch_size=batch_size, epochs=epochs, evaluation_epoch_interval=epochs + 1)
    trainer = Trainer(train_args, model, train_dataset, None, metrics, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_minutes = (time.time() - start) / 60
    train_vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None

    inference_ms, inference_vram = measure_inference(model, test_loader, len(test_dataset), device)
    report = Evaluator.evaluate(model, test_loader, metrics, device, postprocess=lambda x: x)

    summary = {
        "model": "pspnet_resnet50_smp", "category": category, "seed": seed, "split": "family",
        "train_images": len(train_dataset), "test_images": len(test_dataset),
        "epochs": epochs, "batch_size": batch_size, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "train_minutes": round(train_minutes, 2), "train_peak_vram_mb": train_vram and round(train_vram, 1),
        "inference_ms_per_image": round(inference_ms, 1), "inference_peak_vram_mb": inference_vram and round(inference_vram, 1),
        "metrics": {name: round(float(value), 4) for name, value in report.items()},
    }
    save_run("pspnet", category, seed, model, summary, save)
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
