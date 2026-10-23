"""Command-line entry point.

Examples:
    python run.py split bottle
    python run.py seg bottle --arch unet --device cpu --max-epochs 1 --subset 16 --no-save   # quick CPU test
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run.py seg bottle --arch pspnet
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run.py dinomaly hazelnut
"""
import argparse

import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=["split", "seg", "dinomaly"])
parser.add_argument("category")
parser.add_argument("--arch", default="pspnet", help="seg only: smp architecture, e.g. pspnet, unet, deeplabv3")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="cuda")
parser.add_argument("--max-epochs", type=int, default=50, help="seg only")
parser.add_argument("--patience", type=int, default=5, help="seg only: epochs without improvement before stopping")
parser.add_argument("--subset", type=int, default=None, help="seg only: first N training images")
parser.add_argument("--iterations", type=int, default=5000, help="dinomaly only")
parser.add_argument("--no-save", action="store_true")
args = parser.parse_args()

if args.task == "split":
    pipeline.make_family_split(args.category)
elif args.task == "seg":
    pipeline.train_and_evaluate_segmentation(args.category, arch=args.arch, seed=args.seed,
                                             max_epochs=args.max_epochs, patience=args.patience,
                                             device=args.device, subset_size=args.subset, save=not args.no_save)
else:
    pipeline.train_and_evaluate_dinomaly(args.category, seed=args.seed, total_iterations=args.iterations,
                                         device=args.device, save=not args.no_save)
