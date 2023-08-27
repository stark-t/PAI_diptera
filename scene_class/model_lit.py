import os
import torch
import torch.nn as nn
import timm
import pytorch_lightning as lit
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

import STnet_CNN

class LitClassifier(lit.LightningModule):
    def __init__(self, config='pathtoconfig'):
        super().__init__()
        self.config = config
        self.learningrate = config['parameters']['learningrate']
        self.NUM_CLASSES = len([entry.name.lower() for entry in
                                  os.scandir(self.config['paths']['original_data_path'])
                                  if entry.is_dir()])

        if self.config['parameters']['model'] == 'STnet':
            self.model = STnet_CNN.STnet(input_channel=3, num_classes=self.NUM_CLASSES)

        elif self.config['parameters']['model'] == 'resnet18':
            self.model = timm.create_model('resnet18', pretrained=False)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.NUM_CLASSES)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        train_loss = loss_fn(y_pred, y)

        self.log("train_loss", train_loss, on_step=True, prog_bar=False)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        val_loss = loss_fn(y_pred, y)

        self.log("val_loss", val_loss, on_step=True, prog_bar=False)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

        test_loss = torch.nn.functional.cross_entropy(y_pred, y, label_smoothing=0.1)
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        if self.config['parameters']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=float(self.learningrate))

        elif self.config['parameters']['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(),
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

    # def forward(self, batch, batch_idx):
    def forward(self, x):
        # x, y = batch
        y_pred = self.model(x)
        return y_pred, x
