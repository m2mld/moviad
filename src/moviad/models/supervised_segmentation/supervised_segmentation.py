from dataclasses import dataclass

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel

from torch import nn

HF_CHECKPOINTS = {"segformer": "nvidia/mit-b2",  # ImageNet-pretrained MiT-B2 encoder
                  "mask2former": "facebook/mask2former-swin-tiny-ade-semantic",  # full model, ADE20K
                  "oneformer": "shi-labs/oneformer_ade20k_swin_tiny",  # full model, ADE20K
                  "sam3": "facebook/sam3",  # full model, text prompts (gated, needs Hugging Face login)
                  "sam2tiny": "facebook/sam2.1-hiera-tiny",  # full model, box/point prompts
                  "sam2small": "facebook/sam2.1-hiera-small"}  # full model, box/point prompts
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

TEXT_LENGTH = 16  # our texts have at most ~7 tokens; 77 (the default) does not fit into GPU memory

class HFOneFormer(nn.Module):
    """Hugging Face OneFormer (Swin-T, pretrained on ADE20K), fine-tuned for normal/anomaly in semantic mode.

    Like Mask2Former it predicts candidate masks with classes (150 here). It also needs the task text
    "the task is semantic" and, for training, one text per mask ("a photo with a anomaly"), used by its
    query-text contrastive loss. The text encoder is only used for this loss (the public checkpoint does
    not contain it, so it is trained from scratch). OneFormer switches its text branch and dropout with its
    own flag `is_training`, which is therefore set in train() and eval()."""

    def __init__(self, checkpoint, pretrained=True):
        super().__init__()
        from transformers import OneFormerConfig, OneFormerForUniversalSegmentation, OneFormerProcessor
        if pretrained:
            self.network = OneFormerForUniversalSegmentation.from_pretrained(
                checkpoint, is_training=True, num_labels=2, ignore_mismatched_sizes=True,
                text_encoder_context_length=TEXT_LENGTH, **LABELS)
        else:  # the weights will be loaded from a saved run
            config = OneFormerConfig.from_pretrained(checkpoint, is_training=True, num_labels=2,
                                                     text_encoder_context_length=TEXT_LENGTH, **LABELS)
            self.network = OneFormerForUniversalSegmentation(config)
        config = self.network.config
        self.num_text = config.num_queries - config.text_encoder_n_ctx  # 150 - 16 = 134 texts per image
        processor = OneFormerProcessor.from_pretrained(checkpoint)  # CLIP tokenizer: text -> token ids

        def tokenize(texts, length):
            return processor._preprocess_text(texts, max_length=length)

        # token ids are fixed, so they are computed once (buffers, not saved in model.pt)
        self.register_buffer("task_tokens", tokenize(["the task is semantic"], processor.task_seq_length), persistent=False)
        self.register_buffer("class_tokens", tokenize(["a photo with a normal", "a photo with a anomaly"],
                                                      TEXT_LENGTH), persistent=False)
        self.register_buffer("filler_tokens", tokenize(["a semantic photo"], TEXT_LENGTH), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        for module in self.network.modules():
            if hasattr(module, "is_training"):
                module.is_training = mode
        return self

    def compute_loss(self, images, targets):
        """OneFormer loss; one binary mask, class and text per class present in each image."""
        mask_labels = [torch.stack([(target == c).float() for c in target.unique()]) for target in targets]
        class_labels = [target.unique() for target in targets]
        text_inputs = torch.stack([torch.cat([self.class_tokens[classes],
                                              self.filler_tokens.expand(self.num_text - len(classes), -1)])
                                   for classes in class_labels])
        task_inputs = self.task_tokens.expand(len(images), -1)
        return self.network(pixel_values=images, task_inputs=task_inputs, text_inputs=text_inputs,
                            mask_labels=mask_labels, class_labels=class_labels).loss

    def forward(self, images):
        outputs = self.network(pixel_values=images, task_inputs=self.task_tokens.expand(len(images), -1))
        class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]  # drop "no object"
        masks = outputs.masks_queries_logits.sigmoid()
        maps = torch.einsum("bqc,bqhw->bchw", class_probs, masks)
        maps = F.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False)
        probs = maps / maps.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return probs.clamp_min(1e-6).log()

SAM3_PROMPT = "defect"
SAM3_SIZE = 1008  # fixed by the rotary position embeddings of the SAM 3 image encoder


class HFSam3(nn.Module):
    """SAM 3 with the text prompt "defect". The image and text encoders are frozen (about 808 M parameters);
    only the decoders (about 33 M) can be trained.

    The semantic segmentation head gives one defect logit z per pixel. It is returned as the two-class
    logits [0, z], so that the softmax and cross-entropy used for all models equal a sigmoid and a binary
    cross-entropy on z. Images arrive normalized with the ImageNet mean and std; SAM 3 expects
    1008 x 1008 images normalized with mean 0.5 and std 0.5."""

    def __init__(self, checkpoint, pretrained=True):
        super().__init__()
        from transformers import Sam3Config, Sam3Model, Sam3Processor
        if pretrained:
            self.network = Sam3Model.from_pretrained(checkpoint)
        else:  # the weights will be loaded from a saved run
            self.network = Sam3Model(Sam3Config.from_pretrained(checkpoint))
        for module in (self.network.vision_encoder, self.network.text_encoder, self.network.text_projection):
            module.requires_grad_(False)

        # the prompt is fixed, so its token ids are computed once (buffers, not saved in model.pt)
        tokens = Sam3Processor.from_pretrained(checkpoint).tokenizer(
            SAM3_PROMPT, return_tensors="pt", padding="max_length", max_length=32)
        self.register_buffer("input_ids", tokens["input_ids"], persistent=False)
        self.register_buffer("attention_mask", tokens["attention_mask"], persistent=False)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, images):
        pixels = (images * self.imagenet_std + self.imagenet_mean - 0.5) / 0.5  # ImageNet -> SAM 3 normalization
        pixels = F.interpolate(pixels, size=(SAM3_SIZE, SAM3_SIZE), mode="bilinear", align_corners=False)
        with torch.no_grad():  # frozen encoders: no activations are kept for the backward pass
            vision = self.network.get_vision_features(pixels)
            text = self.network.get_text_features(input_ids=self.input_ids, attention_mask=self.attention_mask)
        batch_size = len(images)
        text.pooler_output = text.pooler_output.expand(batch_size, -1, -1)
        outputs = self.network(vision_embeds=vision, text_embeds=text,
                               attention_mask=self.attention_mask.expand(batch_size, -1))
        logits = F.interpolate(outputs.semantic_seg, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([torch.zeros_like(logits), logits], dim=1)

SAM2_SIZE = 1024  # input size of the SAM 2.1 image encoder


class HFSam2(nn.Module):
    """SAM 2.1 (Hiera image encoder), prompted with one box over the whole image and fine-tuned to return
    the defect mask (an empty mask for normal images). SAM 2.1 has no text prompt, so it cannot be used
    without training. The mask decoder gives one logit z per pixel at 256 x 256, returned as the two-class
    logits [0, z] at the input size, as for SAM 3. SAM 2.1 uses the ImageNet normalization, like our input."""

    def __init__(self, checkpoint, pretrained=True):
        super().__init__()
        from transformers import Sam2Config, Sam2Model
        if pretrained:
            self.network = Sam2Model.from_pretrained(checkpoint)
        else:  # the weights will be loaded from a saved run
            self.network = Sam2Model(Sam2Config.from_pretrained(checkpoint))
        # box in the coordinates of the 1024 x 1024 input: the whole image
        # partial fine-tuning: the image encoder is frozen, only the prompt encoder and the mask decoder are
        # trained; training the whole model at 1024 x 1024 does not fit into 12 GB
        self.network.vision_encoder.requires_grad_(False)
        self.register_buffer("box", torch.tensor([[[0.0, 0.0, SAM2_SIZE, SAM2_SIZE]]]), persistent=False)

    def forward(self, images):
        pixels = F.interpolate(images, size=(SAM2_SIZE, SAM2_SIZE), mode="bilinear", align_corners=False)
        outputs = self.network(pixel_values=pixels, input_boxes=self.box.expand(len(images), -1, -1),
                               multimask_output=False)
        logits = outputs.pred_masks[:, 0]  # batch x 1 x 256 x 256 (one box, one mask)
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([torch.zeros_like(logits), logits], dim=1)

HF_WRAPPERS = {"segformer": HFSegformer, "mask2former": HFMask2Former, "oneformer": HFOneFormer, 
               "sam3": HFSam3, "sam2tiny": HFSam2, "sam2small": HFSam2}

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