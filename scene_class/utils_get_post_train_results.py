import argparse
import yaml
from tqdm import tqdm
from glob import iglob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
import pytorch_lightning as lit

from dataset import Dataset
from model_lit import LitClassifier
from utils_get_console_output import get_console_output

from confusion_matrix import cm_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    top_k_accuracy_score,
    cohen_kappa_score,
)
from shutil import rmtree

torch.set_float32_matmul_precision("medium")
sns.set(rc={"figure.figsize": (13, 13)})


def run_results(config, prediction_path):
    # Get data
    df = pd.read_csv(prediction_path)

    # Extract the first three letters of labels_family for legend
    df["family_short"] = df["family"].str[:3]
    print(df.head())

    # calculate accuracy metrics
    acc_balanced = balanced_accuracy_score(df["labels"], df["prediction"])
    acc_kappa = cohen_kappa_score(df["labels"], df["prediction"])
    print("Overall Accuracy:  {:.2f}".format(acc_balanced * 100))
    print("Kappa:  {:.2f}".format(acc_kappa * 100))

    # # Create a confidence matrix
    # # Create a 2x2 matrix where "Confident Correct", "Confident Incorrect", "Not Confident Correct", and "Not Confident Incorrect" are the classes
    # # Filter the dataframe based on the conditions
    # matrix_confident_correct = df[(df["labels"] == df["prediction"]) & (df["probabilities"] > 0.5)].shape[0]
    # matrix_confident_incorrect = df[(df["labels"] != df["prediction"]) & (df["probabilities"] > 0.5)].shape[0]
    # matrix_not_confident_correct = df[(df["labels"] == df["prediction"]) & (df["probabilities"] <= 0.5)].shape[0]
    # matrix_not_confident_incorrect = df[(df["labels"] != df["prediction"]) & (df["probabilities"] <= 0.5)].shape[0]

    # # Create a dataframe from the matrix
    # matrix_data = {
    #     "Confident Correct": [matrix_confident_correct],
    #     "Confident Incorrect": [matrix_confident_incorrect],
    #     "Not Confident Correct": [matrix_not_confident_correct],
    #     "Not Confident Incorrect": [matrix_not_confident_incorrect]
    # }
    # matrix_df = pd.DataFrame(matrix_data)
    # print(matrix_df)

    # # Create a seaborn heatmap of the confidence matrix
    # sns.heatmap(matrix_df, annot=True, fmt="d", cmap="binary", cbar=False)
    # plt.title("Confidence Matrix")
    # plt.show()

    # d=1


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

    # checkpoint_path
    prediction_path = "logs/efficientnet_b4/24030109/probabilities.csv"
    run_results(config, prediction_path)
