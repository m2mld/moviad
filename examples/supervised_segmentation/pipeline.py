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
from torch.utils.data import DataLoader, Subset

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.supervised_segmentation.supervised_segmentation_dataset import SupervisedSegmentationDataset, make_family_split, make_real_anomaly_split
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
    dataset_args = DatasetArguments(dataset_path=str(DATA_ROOT), img_size=(256, 256), gt_mask_size=(256, 256))
    train_dataset = SupervisedSegmentationDataset(dataset_args, category, Split.TRAIN,
                                                  MANIFEST_DIR / category / "train_manifest_family.csv")
    val_dataset = SupervisedSegmentationDataset(dataset_args, category, Split.VAL,
                                                MANIFEST_DIR / category / "val_manifest_family.csv")
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

# re-evaluation of saved runs (adds metrics that did not exist when a run was trained, e.g. image F1)
def reevaluate_run(run_dir, device="cuda"):
    """Load the saved model of a run, evaluate it again on the test set with all_metrics() and update summary.json.
    The metrics computed at training time are kept in summary["metrics_original"]."""
    device = torch.device(device)
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())

    # rebuilding the model of the run (same settings as in training) and loading the saved weights
    if summary["model"].startswith("dinomaly"):
        model = Dinomaly("deit_tiny_patch16_224.fb_in1k")
        image_size, batch_size = (224, 224), 32
    else:
        arch = summary["model"].split("_")[0]  # e.g. "unet_resnet50_smp" -> "unet"
        model = SupervisedSegmentation(arch, encoder_weights=None)  # the weights come from model.pt
        image_size, batch_size = (256, 256), 8
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.to(device).eval()

    # evaluating on the test set with the same evaluator and post-processing as at training time
    test_loader = DataLoader(test_dataset_for(summary["category"], image_size), batch_size=batch_size, shuffle=False)
    if summary["model"].startswith("dinomaly"):
        report = Evaluator.evaluate(model, test_loader, all_metrics(), device)  # default min-max normalisation
    else:
        report = Evaluator.evaluate(model, test_loader, all_metrics(), device, postprocess=lambda x: x)

    # keeping the original metrics and saving the new ones
    summary.setdefault("metrics_original", summary["metrics"])
    summary["metrics"] = {name: round(float(value), 4) for name, value in report.items()}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

def reevaluate_all_runs(device="cuda", include_fixed_epochs=False):
    """Re-evaluate every saved run; fixed-epoch PSPNet runs only if include_fixed_epochs is True."""
    for summary_path in sorted((RUNS_DIR / "runs").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("training") != "early_stopping" and not summary["model"].startswith("dinomaly") \
                and not include_fixed_epochs:
            continue
        summary = reevaluate_run(summary_path.parent, device)

        # check: the metrics that already existed must not change
        largest_change = max(abs(summary["metrics"][name] - value) for name, value in summary["metrics_original"].items())
        print(f"{summary_path.parent.name}: img_f1 {summary['metrics']['img_f1']:.4f}, "
              f"largest change of the old metrics {largest_change:.4f}")