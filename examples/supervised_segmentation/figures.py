"""Figures and result tables for the thesis, from the saved server runs.

Usage:
    python figures.py compare bottle      # PSPNet vs Dinomaly maps (first test image of each defect type)
    python figures.py examples bottle     # PSPNet map and prediction at 0.5 (same as examples.py)
    python figures.py mirage bottle       # random sample of MIRAGE images with mask contours
    python figures.py table               # all summary.json files -> CSV (per run and mean ± std)
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no screen on the server: draw directly into a file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import pipeline
from examples import latest_run_dir, save_pspnet_examples
from moviad.models.dinomaly.dinomaly import Dinomaly

FIG_DIR = pipeline.RUNS_DIR / "figures"


def first_index_per_type(dataset):
    labels = dataset.samples["label"].tolist()
    return {defect_type: labels.index(defect_type) for defect_type in sorted(set(labels))}


def compare_maps(category, seed=42, device="cpu"):
    """Image, ground truth, PSPNet defect probability and Dinomaly score (min-max over the whole test set)."""
    device = torch.device(device)
    pspnet = pipeline.SupervisedSegmentation().to(device)
    pspnet.load_state_dict(torch.load(latest_run_dir("pspnet", category, seed) / "model.pt", map_location=device))
    pspnet.eval()
    dinomaly = Dinomaly("deit_tiny_patch16_224.fb_in1k").to(device)
    dinomaly.load_state_dict(torch.load(latest_run_dir("dinomaly", category, seed) / "model.pt", map_location=device))
    dinomaly.eval()

    test_256 = pipeline.test_dataset_for(category, (256, 256))
    test_224 = pipeline.test_dataset_for(category, (224, 224))

    # Dinomaly scores have no natural scale: normalise with min and max over the whole test set,
    # as MoViAD's Evaluator does (min_max_norm).
    dinomaly_maps = []
    with torch.no_grad():
        for images, *_ in torch.utils.data.DataLoader(test_224, batch_size=16, shuffle=False):
            maps, _ = dinomaly(images.to(device))
            dinomaly_maps.append(maps.cpu())
    dinomaly_maps = torch.cat(dinomaly_maps)
    low, high = dinomaly_maps.min(), dinomaly_maps.max()
    dinomaly_maps = (dinomaly_maps - low) / (high - low)

    rows = first_index_per_type(test_256)
    figure, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)), squeeze=False)
    for row, (defect_type, index) in enumerate(rows.items()):
        image, label, mask, path = test_256[index]
        with torch.no_grad():
            pspnet_map, _ = pspnet(image.unsqueeze(0).to(device))
        dinomaly_map = F.interpolate(dinomaly_maps[index][None, None], size=(256, 256), mode="bilinear")[0, 0]

        axes[row, 0].imshow(image.permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f"{defect_type} ({Path(path).name})")
        axes[row, 1].imshow(mask[0], cmap="gray")
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(pspnet_map[0].cpu().numpy(), cmap="jet", vmin=0, vmax=1)
        axes[row, 2].set_title("PSPNet (defect probability)")
        axes[row, 3].imshow(dinomaly_map.numpy(), cmap="jet", vmin=0, vmax=1)
        axes[row, 3].set_title("Dinomaly (normalised score)")
        for axis in axes[row]:
            axis.axis("off")

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIG_DIR / f"{category}_pspnet_vs_dinomaly_seed{seed}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(figure)
    print("Saved to:", output_path)


def examples(category, seed=42, device="cpu"):
    source = save_pspnet_examples(category, seed=seed, device=device)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    target = FIG_DIR / f"{category}_pspnet_examples_seed{seed}.png"
    shutil.copy(source, target)
    print("Copied to:", target)


def mirage_sample(category, n=30, sample_seed=0, size=512):
    """Random MIRAGE images of one category with the mask outline in red."""
    mirage_dir = pipeline.DATA_ROOT / "mirage" / category
    paths = sorted(path for path in (mirage_dir / "images").glob("*.png") if not path.name.startswith("._"))
    rng = np.random.default_rng(sample_seed)
    chosen = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    columns = 6
    rows = int(np.ceil(len(chosen) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, sorted(chosen)):
        path = paths[index]
        image = np.asarray(Image.open(path).convert("RGB").resize((size, size)))
        mask = np.asarray(Image.open(mirage_dir / "masks" / f"{path.stem}_mask.png").convert("L").resize((size, size), Image.NEAREST)) > 127
        axis.imshow(image)
        axis.contour(mask, levels=[0.5], colors="red", linewidths=0.8)
        axis.set_title(path.stem, fontsize=9)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIG_DIR / f"{category}_mirage_sample_{n}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(figure)
    print("Saved to:", output_path)


def results_table():
    """One row per saved run, plus mean and standard deviation per model and category."""
    rows = []
    for path in sorted((pipeline.RUNS_DIR / "runs").glob("*/summary.json")):
        summary = json.loads(path.read_text())
        rows.append({
            "run": path.parent.name, "model": summary["model"], "category": summary["category"],
            "training": summary.get("training", "fixed_epochs"),
            "seed": summary["seed"], "train_minutes": summary["train_minutes"],
            "train_peak_vram_mb": summary["train_peak_vram_mb"],
            "inference_ms_per_image": summary["inference_ms_per_image"],
            "inference_peak_vram_mb": summary["inference_peak_vram_mb"],
            **summary["metrics"],
        })
    table = pd.DataFrame(rows)
    numeric = table.drop(columns=["run", "seed"]).groupby(["category", "model", "training"])
    summary_table = numeric.mean().round(4).astype(str) + " ± " + numeric.std().round(4).fillna(0).astype(str)
    summary_table.insert(0, "runs", numeric.size())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(FIG_DIR / "results_per_run.csv", index=False)
    summary_table.to_csv(FIG_DIR / "results_mean_std.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary_table[["runs", "img_roc_auc", "pxl_roc_auc", "pxl_f1", "pxl_iou", "train_minutes"]])
    print("Saved to:", FIG_DIR / "results_per_run.csv", "and", FIG_DIR / "results_mean_std.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=["compare", "examples", "mirage", "table"])
    parser.add_argument("category", nargs="?")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.figure == "compare":
        compare_maps(args.category, seed=args.seed, device=args.device)
    elif args.figure == "examples":
        examples(args.category, seed=args.seed, device=args.device)
    elif args.figure == "mirage":
        mirage_sample(args.category)
    else:
        results_table()
