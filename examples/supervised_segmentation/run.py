"""Command-line entry point.

Examples:
    python run.py split bottle
    python run.py pspnet bottle --device cpu --epochs 1 --subset 16 --no-save   # quick CPU test
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run.py pspnet bottle --seed 42
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run.py dinomaly hazelnut
"""
import argparse

import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=["split", "pspnet", "dinomaly"])
parser.add_argument("category")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="cuda")
parser.add_argument("--epochs", type=int, default=20, help="PSPNet only")
parser.add_argument("--subset", type=int, default=None, help="PSPNet only: first N training images")
parser.add_argument("--iterations", type=int, default=5000, help="Dinomaly only")
parser.add_argument("--no-save", action="store_true")
args = parser.parse_args()

if args.task == "split":
    pipeline.make_family_split(args.category)
elif args.task == "pspnet":
    pipeline.train_and_evaluate_pspnet(args.category, seed=args.seed, epochs=args.epochs, device=args.device,
                                       subset_size=args.subset, save=not args.no_save)
else:
    pipeline.train_and_evaluate_dinomaly(args.category, seed=args.seed, total_iterations=args.iterations,
                                         device=args.device, save=not args.no_save)
