# Supervised segmentation for visual anomaly detection

Thesis code (Milan Mladenovic, MSc, University of Padova). 
A segmentation model (smp PSPNet, ResNet-50) is trained on MVTec AD normal images plus MIRAGE synthetic anomalies with pixel masks, and compared with Dinomaly on the real MVTec AD test set.
MoViAD components used: MVTecDataset, Trainer, Evaluator, metrics, Dinomaly.

## Files
- pipeline.py: metrics (IoU, Dice), family split, training dataset, model, train/evaluate functions
- run.py: command line entry point
- figures.py: results table and figures from the saved runs
- examples.py: PSPNet prediction examples for one run

## Data
pipeline.py expects two links in ~/datasets/thesis_data:
- mvtec  -> MVTec AD
- mirage -> MIRAGE, MVTec part (with generation_tracking.csv)

## Usage
    python run.py split bottle                  # train/val manifests (family split)
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run.py pspnet bottle --seed 42
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python run.py dinomaly bottle
    python figures.py table                     # mean and std over seeds, all categories

Outputs go to ~/thesis_runs (manifests, one folder per run with model.pt and summary.json, figures).

## Family split
A family is one MVTec normal training image plus all MIRAGE images generated from it
(base_img_path in generation_tracking.csv). Whole families go to train or validation,
so validation does not contain near-copies of training images. MIRAGE images with an
empty mask are excluded.

## Known limitations
- The validation set is not used yet: the model after the last epoch is tested.
- PSPNet uses 80% of train/good (rest is validation), Dinomaly uses all of it.
- Metrics are computed at 256x256 (PSPNet) and 224x224 (Dinomaly).
- Inference time includes data loading; inference VRAM includes memory left from training.
- Dinomaly gets images without ImageNet normalisation (as in the MoViAD Dinomaly test).