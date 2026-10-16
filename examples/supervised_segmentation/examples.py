"""Example figure: first test image of each defect type with ground truth, PSPNet anomaly map and prediction.

Usage:
    python examples.py bottle            # latest PSPNet run of bottle, seed 42, on CPU
    python examples.py hazelnut --seed 0
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no screen on the server: draw directly into a file
import matplotlib.pyplot as plt
import numpy as np
import torch

import pipeline


def latest_run_dir(model_name, category, seed):
    runs = sorted((pipeline.RUNS_DIR / "runs").glob(f"{model_name}_{category}_seed{seed}_*"))
    assert runs, f"No saved run for {model_name} {category} seed {seed}"
    return runs[-1]


def save_pspnet_examples(category, seed=42, device="cpu"):
    device = torch.device(device)
    run_dir = latest_run_dir("pspnet", category, seed)
    model = pipeline.SupervisedSegmentation().to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()

    test_dataset = pipeline.test_dataset_for(category, (256, 256))
    labels = test_dataset.samples["label"].tolist()
    defect_types = sorted(set(labels))
    first_index = {defect_type: labels.index(defect_type) for defect_type in defect_types}

    figure, axes = plt.subplots(len(defect_types), 4, figsize=(12, 3 * len(defect_types)), squeeze=False)
    for row, defect_type in enumerate(defect_types):
        image, label, mask, path = test_dataset[first_index[defect_type]]
        with torch.no_grad():
            anomaly_map, score = model(image.unsqueeze(0).to(device))
        anomaly_map = anomaly_map[0].cpu().numpy()
        image_np = image.permute(1, 2, 0).numpy()

        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"{defect_type} ({Path(path).name})")
        axes[row, 1].imshow(mask[0], cmap="gray")
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(anomaly_map, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Anomaly map (max {score.item():.2f})")
        axes[row, 3].imshow(image_np)
        axes[row, 3].imshow(np.ma.masked_where(anomaly_map < 0.5, anomaly_map), cmap="autumn", alpha=0.6)
        axes[row, 3].set_title("Prediction at 0.5")
        for axis in axes[row]:
            axis.axis("off")

    plt.tight_layout()
    output_path = run_dir / "examples_per_defect_type.png"
    plt.savefig(output_path, dpi=120)
    plt.close(figure)
    print("Saved to:", output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    save_pspnet_examples(args.category, seed=args.seed, device=args.device)
