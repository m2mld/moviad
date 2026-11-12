from dataclasses import dataclass

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel

from torch import nn

HF_CHECKPOINTS = {"segformer": "nvidia/mit-b2",  # ImageNet-pretrained MiT-B2 encoder
                  "mask2former": "facebook/mask2former-swin-tiny-ade-semantic"}  # full model, ADE20K
LABELS = {"id2label": {0: "normal", 1: "anomaly"}, "label2id": {"normal": 0, "anomaly": 1}}


class HFSegformer(nn.Module):
    """Hugging Face SegFormer that returns logits at the input resolution, like the smp models.

    SegFormer predicts at 1/4 of the input size, so the logits are upsampled to the image size."""

    def __init__(self, checkpoint, pretrained=True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            self.network = SegformerForSemanticSegmentation.from_pretrained(checkpoint, num_labels=2, **LABELS)
        else:  # the weights will be loaded from a saved run
            config = SegformerConfig.from_pretrained(checkpoint, num_labels=2, **LABELS)
            self.network = SegformerForSemanticSegmentation(config)

    def forward(self, images):
        logits = self.network(pixel_values=images).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

class HFMask2Former(nn.Module):
    """Hugging Face Mask2Former (pretrained on ADE20K semantic segmentation), fine-tuned for normal/anomaly.

    Mask2Former predicts 100 candidate masks, each with a class (normal, anomaly or "no object").
    Training uses its own loss; for evaluation the candidate masks are combined into per-pixel class
    probabilities, returned as log-probabilities so that a softmax gives back the probabilities."""

    def __init__(self, checkpoint, pretrained=True):
        super().__init__()
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
        if pretrained:
            self.network = Mask2FormerForUniversalSegmentation.from_pretrained(
                checkpoint, num_labels=2, ignore_mismatched_sizes=True, **LABELS)
        else:  # the weights will be loaded from a saved run
            config = Mask2FormerConfig.from_pretrained(checkpoint, num_labels=2, **LABELS)
            self.network = Mask2FormerForUniversalSegmentation(config)

    def compute_loss(self, images, targets):
        """Mask2Former loss; targets (batch x H x W, 0/1) become one binary mask per class present in each image."""
        mask_labels = [torch.stack([(target == c).float() for c in target.unique()]) for target in targets]
        class_labels = [target.unique() for target in targets]
        return self.network(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels).loss

    def forward(self, images):
        outputs = self.network(pixel_values=images)
        class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]  # drop "no object"
        masks = outputs.masks_queries_logits.sigmoid()
        maps = torch.einsum("bqc,bqhw->bchw", class_probs, masks)  # batch x 2 x H/4 x W/4
        maps = F.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False)
        probs = maps / maps.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return probs.clamp_min(1e-6).log()


HF_WRAPPERS = {"segformer": HFSegformer, "mask2former": HFMask2Former}

@dataclass
@dataclass
class SegmentationTrainArgs(TrainingArgs):
    """Training settings: which optimizer and which loss.

    The smp models and SegFormer use SGD; Mask2Former uses AdamW with gradient clipping, as in its paper,
    because with SGD and a learning rate of 0.01 its weights diverge to NaN in the first epoch."""
    learning_rate: float = 0.01
    optimizer_name: str = "sgd"
    weight_decay: float = 0.0005
    grad_clip_norm: float | None = None

    def init_train(self, model):
        if self.optimizer is None:
            if self.optimizer_name == "adamw":
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate,
                                                   weight_decay=self.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate,
                                                 momentum=0.9, weight_decay=self.weight_decay)
        if self.loss_function is None:
            self.loss_function = F.cross_entropy


class SupervisedSegmentation(VADModel):
    """An smp segmentation model trained with pixel masks, packaged so MoViAD can train and evaluate it.

    arch: smp architecture name, e.g. "pspnet", "unet", "deeplabv3".
    """

    def __init__(self, arch="pspnet", encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
        if arch in HF_CHECKPOINTS:
            self.network = HF_WRAPPERS[arch](HF_CHECKPOINTS[arch], pretrained=encoder_weights is not None)
        else:
            self.network = smp.create_model(arch, encoder_name=encoder_name, encoder_weights=encoder_weights, classes=2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        logits = self.network((images - self.mean) / self.std)
        if self.training:
            return logits
        anomaly_maps = logits.softmax(dim=1)[:, 1]
        return anomaly_maps, anomaly_maps.amax(dim=(1, 2))

    def train_epoch(self, epoch, train_dataloader, training_args):
        """Train for one epoch and return the average training loss.

        Unlike unsupervised VAD models, which receive only images, each batch here
        contains images and their pixel masks (normal images have an empty mask)"""
        total_loss = 0
        for batch in tqdm(train_dataloader):
            total_loss += self.train_step(batch, training_args)
        return total_loss / len(train_dataloader)

    def train_step(self, batch, training_args):
        images, masks = batch
        device = self.mean.device
        images, targets = images.to(device), masks[:, 0].to(device)
        if hasattr(self.network, "compute_loss"):  # models with their own loss (Mask2Former)
            loss = self.network.compute_loss((images - self.mean) / self.std, targets)
        else:
            loss = training_args.loss_function(self(images), targets)
        training_args.optimizer.zero_grad()
        loss.backward()
        if training_args.grad_clip_norm is not None:  # limits the size of the update (Mask2Former)
            torch.nn.utils.clip_grad_norm_(self.parameters(), training_args.grad_clip_norm)
        training_args.optimizer.step()
        return loss.item()