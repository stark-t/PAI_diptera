import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from glob import iglob
from sklearn.model_selection import train_test_split
import sys
import shutil
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


def main(config):
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
            group, test_size=0.4, random_state=config["parameters"]["seed"]
        )
        group_val, group_test = train_test_split(
            group_temp, test_size=0.5, random_state=config["parameters"]["seed"]
        )
        # _ = pd.concat([df_train, group_train])
        # _ = pd.concat([df_val, group_val])
        df_test = pd.concat([df_test, group_test])

    # print column names
    # print(df_test.columns)
    # print(df_test["file_name"].head())

    pretrained_BB_predictions = list(iglob("data/results/pretrained_BB/*.csv"))

    # Create dataframe with all predictions from the list of pretrained_BB_predictions
    combined_df = pd.DataFrame()
    for file in pretrained_BB_predictions:
        df = pd.read_csv(file)
        model_name = file.split(os.sep)[-1].split(".csv")[0]
        df["model_name"] = model_name
        # Rename probabilties and predictions by joining the model_name to the column name
        df = df.rename(
            columns={
                "probabilities": "probabilities_" + model_name,
                "prediction": "prediction_" + model_name,
            }
        )
        # From df select only prediction and probabilities columns
        df_selection = df[["prediction_" + model_name, "probabilities_" + model_name]]
        # print(df_selection.head())

        df_test.reset_index(drop=True, inplace=True)
        df_selection.reset_index(drop=True, inplace=True)
        df_test = df_test.join(df_selection)

    # print column names
    # print(df_test.columns)

    # Calculate the standard deviation for the columns "probabilities_ResNet", "probabilities_Efficientnet", and "probabilities_mobilenet"
    df_test["std_probabilities"] = df_test[
        [
            "probabilities_ResNet-18",
            "probabilities_EfficientNet_b4",
            "probabilities_MobileNetV3",
        ]
    ].std(axis=1)

    # Group the dataframe by "family"
    grouped_by_family = df_test.groupby("family")

    # Iterate over each group
    for family, group in grouped_by_family:
        # Find the filename with the highest std
        highest_std_filename = group.loc[
            group["std_probabilities"].idxmax(), "file_path"
        ]
        # Find the filename with the lowest std
        lowest_std_filename = group.loc[
            group["std_probabilities"].idxmin(), "file_path"
        ]

        highest_std_MobileNetV3 = group.loc[
            group["std_probabilities"].idxmax(), "probabilities_MobileNetV3"
        ]
        highest_std_ResNet18 = group.loc[
            group["std_probabilities"].idxmax(), "probabilities_ResNet-18"
        ]
        highest_std_EfficientNet = group.loc[
            group["std_probabilities"].idxmax(), "probabilities_EfficientNet_b4"
        ]

        lowest_std_MobileNetV3 = group.loc[
            group["std_probabilities"].idxmin(), "probabilities_MobileNetV3"
        ]
        lowest_std_ResNet18 = group.loc[
            group["std_probabilities"].idxmin(), "probabilities_ResNet-18"
        ]
        lowest_std_EfficientNet = group.loc[
            group["std_probabilities"].idxmin(), "probabilities_EfficientNet_b4"
        ]

        # Create a new directory for the figures
        figures_dir = "./data/results/figures"
        os.makedirs(figures_dir, exist_ok=True)

        # Print the results
        # Copy highest_std_filename and lowest_std_filename into data/results/std_figures
        os.makedirs("data/results/std_figures", exist_ok=True)
        shutil.copy(highest_std_filename, "data/results/std_figures")
        shutil.copy(lowest_std_filename, "data/results/std_figures")

        print(f"Family: {family}")
        print(f"Highest Std Filename: {highest_std_filename}")
        print(f"MobileNetV3: {highest_std_MobileNetV3}")
        print(f"ResNet-18: {highest_std_ResNet18}")
        print(f"EfficientNet: {highest_std_EfficientNet}")
        print()
        print(f"Lowest Std Filename: {lowest_std_filename}")
        print(f"MobileNetV3: {lowest_std_MobileNetV3}")
        print(f"ResNet-18: {lowest_std_ResNet18}")
        print(f"EfficientNet: {lowest_std_EfficientNet}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="scene_class/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # run training
    main(config)
