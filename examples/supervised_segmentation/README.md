# Supervised segmentation for visual anomaly detection

Thesis code (Milan Mladenovic, MSc, University of Padova).
Segmentation models from segmentation_models_pytorch (PSPNet, U-Net, DeepLabV3, ResNet-50 encoder) are trained on MVTec AD normal images plus MIRAGE synthetic anomalies with pixel masks, with early stopping on a validation split, and compared with Dinomaly on the real MVTec AD test set.

## Components added to MoViAD (src/moviad/)
- datasets/supervised_segmentation/: `SupervisedSegmentationDataset` (subclass of `VADDataset`) and `make_family_split`
- models/supervised_segmentation/: `SupervisedSegmentation` (subclass of `VADModel`, any smp architecture) and `SegmentationTrainArgs`
- trainers/trainer_early_stopping.py: `EarlyStoppingTrainer` (subclass of `Trainer`)
- utilities/evaluation/metrics.py: `IoU` and `Dice`

MoViAD components used as they are: `MVTecDataset` (test set), `Trainer`, `Evaluator`, metrics, `Dinomaly`.

## Files in this folder
- pipeline.py: paths, metric selection, train/evaluate functions, re-evaluation of saved runs
- run.py: command line entry point
- figures.py: results table and figures from the saved runs
- examples.py: PSPNet prediction examples for one run

## Data
pipeline.py expects two links in ~/datasets/thesis_data:
- mvtec  -> MVTec AD
- mirage -> MIRAGE, MVTec part (with generation_tracking.csv)

## Usage
    python run.py split bottle                                   # train/val manifests (family split)
    python run.py seg bottle --arch unet --device cpu --max-epochs 1 --subset 16 --no-save   # quick CPU test
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run.py seg bottle --arch unet
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run.py dinomaly bottle
    python run.py reeval all                                     # evaluate all saved runs again (e.g. to add new metrics)
    python figures.py table                                      # results table, all runs
    python figures.py compare bottle                             # PSPNet vs Dinomaly maps
    python figures.py mirage bottle                              # 30 MIRAGE images with mask outlines

Outputs go to ~/thesis_runs: manifests, one folder per run, figures. A supervised run folder contains
model.pt (best validation epoch, the one that is tested), last_model.pt, history.csv and summary.json;
a Dinomaly run folder contains model.pt and summary.json.

## Training and evaluation
- Supervised models: 256x256, batch 8, SGD (lr 0.01, momentum 0.9, weight decay 5e-4), cross-entropy,
  early stopping on the validation pixel max-F1 (patience 5, at most 50 epochs), seed 42.
- Dinomaly: MoViAD implementation, DeiT-tiny encoder, 224x224, 5000 iterations, seed 42.
- Metrics: image AUROC, pixel AUROC, image F1, pixel F1 (max-F1), and IoU and Dice at threshold 0.5.

## Family split
A family is one MVTec normal training image plus all MIRAGE images generated from it
(base_img_path in generation_tracking.csv). Whole families go to train or validation,
so validation does not contain near-copies of training images. MIRAGE images with an
empty mask are excluded.

## Known limitations
- The validation set contains synthetic (MIRAGE) anomalies only, so the best validation epoch is not always the best on real defects.
- Supervised models use 80% of train/good (the rest is validation), Dinomaly uses all of it.
- Metrics are computed at 256x256 (supervised models) and 224x224 (Dinomaly).
- IoU and Dice are not meaningful for Dinomaly, whose scores are min-max normalised over the test set.
- Inference time includes data loading; inference VRAM includes memory left from training.
- Dinomaly gets images without ImageNet normalisation (as in the MoViAD Dinomaly test).
- figures.py compare and examples.py use the latest PSPNet run only.