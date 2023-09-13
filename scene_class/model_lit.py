import os
import torch
import torch.nn as nn
import timm
import pytorch_lightning as lit
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

import STnet_CNN

class LitClassifier(lit.LightningModule):
    def __init__(self, config='pathtoconfig'):
        super().__init__()
        self.config = config
        self.learningrate = config['parameters']['learningrate']
        self.NUM_CLASSES = config['parameters']['num_classes']

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