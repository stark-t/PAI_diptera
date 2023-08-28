import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from glob import iglob
from sklearn.model_selection import train_test_split
# import glob

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

import pytorch_lightning as lit
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning.callbacks import TQDMProgressBar

from dataset import Dataset
from model_lit import LitClassifier




def main(config):
    seed_everything(42, workers=True)

    all_files = list(iglob(config['paths']['data_path'] + os.sep + '*.jpeg'))
    file_df = pd.DataFrame({'file_path': all_files})
    file_df['file_name'] = file_df['file_path'].apply(os.path.basename)
    file_df[['family', 'species', 'genus', 'id1', 'id2_suffix']] = file_df['file_name'].str.split('_', expand=True)
    file_df.drop(columns=['id1', 'id2_suffix'], inplace=True, errors='ignore')
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    groups = file_df.groupby('family', group_keys=False)
    # Split each group into train, val, and test
    for _, group in groups:
        group_train, group_temp = train_test_split(group, test_size=0.4, random_state=42)
        group_val, group_test = train_test_split(group_temp, test_size=0.5, random_state=42)
        df_train = pd.concat([df_train, group_train])
        df_val = pd.concat([df_val, group_val])
        df_test = pd.concat([df_test, group_test])
    # Get data
    train_set = Dataset(path=df_train['file_path'].tolist(), config=config, norm=config['parameters']['normalization'])
    val_set = Dataset(path=df_val['file_path'].tolist(), config=config, norm=config['parameters']['normalization'])
    test_set = Dataset(path=df_test['file_path'].tolist(), config=config, norm=config['parameters']['normalization'])

    if config['parameters']['verbose'] > 1:
        grouped_df = df_train.groupby("family").nunique()
        print(grouped_df)
        randints = torch.randint(low=0, high=len(train_set), size=(10,))
        for i in randints:
            image, label = train_set[i]
            plt.imshow(image.numpy()[0, :, :])
            plt.title(str(label))
            plt.show()
            plt.close()

    # Initialize DataLoader
    n_cpu = os.cpu_count()
    train_loader = DataLoader(
        train_set, batch_size=8, num_workers=n_cpu, shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=8, num_workers=n_cpu)
    test_loader = DataLoader(
        test_set, batch_size=8, num_workers=n_cpu)

    # Logger and Callbacks

    class NoValidationProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            bar = super().init_validation_tqdm()
            bar.disable = True
            return bar

    log_path = os.path.join(os.getcwd(), 'logs')
    logger = TensorBoardLogger(log_path)
    checkpoint = lit.callbacks.ModelCheckpoint(
                                               save_top_k=1,
                                               every_n_epochs=None,
                                               every_n_train_steps=None,
                                               train_time_interval=None,
                                               save_on_train_epoch_end=True,
                                               monitor="val_loss",
                                               mode='min')

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=5000, patience=200, verbose=False, mode="min")
    callbacks = [early_stop_callback, checkpoint, NoValidationProgressBar()]

    # Model
    model = LitClassifier(config=config)

    # Trainer
    trainer = lit.Trainer(
        accelerator="gpu",
        precision=16,
        max_epochs=config['parameters']['epochs'],
        devices=torch.cuda.device_count(),
        logger=logger,
        callbacks=callbacks,
        deterministic=False
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                default='/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/scene_class/config.yaml',
                help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)