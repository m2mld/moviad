from dataclasses import dataclass

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel

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
    """An smp segmentation model trained with pixel masks, packaged so MoViAD can train and evaluate it.

    arch: smp architecture name, e.g. "pspnet", "unet", "deeplabv3".
    """

    def __init__(self, arch="pspnet", encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
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
        logits = self(images)
        loss = training_args.loss_function(logits, targets)
        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()
        return loss.item()