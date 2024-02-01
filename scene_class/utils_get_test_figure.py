import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from glob import iglob
from sklearn.model_selection import train_test_split
import sys
from datetime import datetime
import warnings

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

import pytorch_lightning as lit
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar

from dataset import Dataset
from model_lit import LitClassifier
from model_STnet import LitClassifier as LitClassifier_STnet
import os

torch.set_float32_matmul_precision("medium")

def main(config, log_itmes):
    seed_everything(config["parameters"]["seed"], workers=True)

    all_files = list(iglob(config["paths"]["data_path"] + os.sep + "*.jpeg"))
    file_df = pd.DataFrame({"file_path": all_files})
    file_df["file_name"] = file_df["file_path"].apply(os.path.basename)
    file_df[["family", "species", "genus", "id1", "id2_suffix"]] = file_df[
        "file_name"
    ].str.split("_", expand=True)
    file_df.drop(columns=["id1", "id2_suffix"], inplace=True, errors="ignore")
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    groups = file_df.groupby("family", group_keys=False)
    # Split each group into train, val, and test
    for _, group in groups:
        group_train, group_temp = train_test_split(
            group, test_size=0.4, random_state=42
        )
        group_val, group_test = train_test_split(
            group_temp, test_size=0.5, random_state=42
        )
        df_train = pd.concat([df_train, group_train])
        df_val = pd.concat([df_val, group_val])
        df_test = pd.concat([df_test, group_test])

    # Loop through ./logs and get all prediction files
    log_files = []
    model_names = []
    times = []
    for root, dirs, files in os.walk("./logs"):
        for file in files:
            if file.endswith("probabilities.csv"):
                log_files.append(os.path.join(root, file))
                model_names.append(dirs[1])
                times.append(dirs[2])

    # Create dataframe with all predictions




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/scene_class/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # run training
    main(config)
