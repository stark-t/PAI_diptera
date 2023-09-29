import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm
import pytorch_lightning as lit
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline
from torch import Tensor
import matplotlib.pyplot as plt

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_augs_basic: bool = True,
                 apply_augs_color: bool = False,
                 apply_augs_geom: bool = False,) -> None:
        super().__init__()
        self._apply_augs_basic = apply_augs_basic
        self._apply_augs_color = apply_augs_color
        self._apply_augs_geom = apply_augs_geom

        self.augs_basic = nn.Sequential(
            RandomHorizontalFlip(p=0.75),
        )

        self.augs_color = nn.Sequential(
            ColorJitter(0.5, 0.5, 0.5, 0.5, p=0.75),
            RandomChannelShuffle(p=0.75),
            RandomHorizontalFlip(p=0.75),
            RandomThinPlateSpline(p=0.75),
        )

        self.augs_geom = nn.Sequential(
            RandomThinPlateSpline(p=0.75),
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        # BxCxHxW
        if self._apply_augs_basic:
            x = self.augs_basic(x)
        if self._apply_augs_color:
            x = self.augs_color(x)
        if self._apply_augs_geom:
            x = self.augs_geom(x)

        return x


class LitClassifier(lit.LightningModule):
    def __init__(self, config='pathtoconfig'):
        super().__init__()
        self.config = config
        self.learningrate = config['parameters']['learningrate']
        self.NUM_CLASSES = config['parameters']['num_classes']
        self.transform = DataAugmentation(apply_augs_basic=True, apply_augs_color=True, apply_augs_geom=True)  # per batch augmentation_kornia

        listmodels = timm.list_models(config['parameters']['model'])
        if len(listmodels) > 1:
            print("Use specific timm model")
            print("Models selected:")
            print(listmodels)
            print(1/0)

        timm_model = listmodels[0]

        # Load the pretrained model from Timm with the specified name
        self.feature_extractor = timm.create_model(timm_model, pretrained=config["parameters"]["pretrained"],
                                                   num_classes=0, global_pool='')

        # Determine the input size for the classifier dynamically
        num_features = self.feature_extractor.num_features

        # Add a dropout layer
        self.dropout = nn.Dropout(p=config["parameters"]["dropout"])

        # Add a linear layer for classification
        self.classifier = nn.Linear(num_features, config["parameters"]["num_classes"])

    def show_batch(self, win_size=(10, 10)):
        def _to_vis(data):
            return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

        # get a batch from the training set: try with `val_datlaoader` :)
        imgs, labels = next(iter(self.train_dataloader()))
        imgs_aug = self.transform(imgs)  # apply augmentations
        # use matplotlib to visualize
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs))
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs_aug))

    def on_after_batch_transfer(self, batch, batch_idx):
        x, y = batch
        if self.trainer.training:
            x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        train_loss = loss_fn(y_pred, y)

        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y_pred = self(x)
        y_pred = self(x)

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        val_loss = loss_fn(y_pred, y)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        test_loss = torch.nn.functional.cross_entropy(y_pred, y, label_smoothing=0.1)
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        if self.config['parameters']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=float(self.learningrate))

        elif self.config['parameters']['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=float(self.learningrate),
                                          weight_decay=1e-3)

        if self.config['parameters']['learningrate_sheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                    T_0=8,  # Number of iterations for the first restart
                                                    T_mult=1,  # A factor increases TiTi​ after a restart
                                                    eta_min=1e-6)  # Minimum learning rate
            return [optimizer], [scheduler]
        else:
            scheduler = None
            return [optimizer]
            # scheduler = ReduceLROnPlateau(optimizer, 'min')

    def predict_step(self, batch, batch_idx):
        # predict outputs for input batch and return as dictionary
        x, y = batch
        y_hat = self(x)
        return y_hat

    def forward(self, x):
        # Forward pass through the network
        features = self.feature_extractor(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))  # Global average pooling
        features = features.view(features.size(0), -1)  # Flatten
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits