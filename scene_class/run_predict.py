import argparse
import yaml
from glob import iglob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from glob import glob

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
import pytorch_lightning as lit

from dataset import Dataset
from model_lit import LitClassifier
from model_STnet import LitClassifier as LitClassifier_STnet
from utils_get_console_output import get_console_output

from confusion_matrix import cm_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
)
from shutil import rmtree

torch.set_float32_matmul_precision("medium")
sns.set(rc={"figure.figsize": (13, 13)})


def convert_seconds_to_hh_mm_ss(seconds):
    hours, remainder = divmod(int(seconds), 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute
    return hours, minutes, seconds


def get_nth_directory_from_end(path, n):
    # Split the path into its components and reverse the list
    components = path.split(os.path.sep)[:-n]
    return os.sep + os.path.join(*components)


def run_predict(config, ckpt="checkpoint_path"):
    seed_everything(config["parameters"]["seed"], workers=True)

    label_familynames = [
        entry.name.lower()
        for entry in os.scandir(config["paths"]["original_data_path"])
        if entry.is_dir()
    ]
    label_familynames_sorted = sorted(label_familynames)
    # label_familynames_sorted = label_familynames_sorted[:-3]
    label_int = list(range(len(label_familynames_sorted)))  # noqa: F841

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
        df_train = pd.concat([df_train, group_train])
        df_val = pd.concat([df_val, group_val])
        df_test = pd.concat([df_test, group_test])

    test_set = Dataset(
        path=df_test["file_path"].tolist(),
        config=config,
        norm=config["parameters"]["normalization"],
    )

    all_predictions = []

    for _ in range(config["parameters"]["test_time_iterations"]):
        test_loader = DataLoader(test_set, batch_size=16, num_workers=10, shuffle=False)

        # Model
        if config["parameters"]["model"] == "STnet":
            model = LitClassifier_STnet(config=config)
        else:
            model = LitClassifier(config=config)
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint["state_dict"])

        # Set the model in training mode to enable dropout during prediction
        model.train()

        trainer = lit.Trainer(
            accelerator="gpu",
            precision=16,
            devices=torch.cuda.device_count(),
            deterministic=False,
        )

        predictions = trainer.predict(model, test_loader)

        # Step 1: Concatenate all batched predictions into a single tensor
        concatenated_predictions = torch.cat(predictions, dim=0)

        # Step 2: Convert the concatenated tensor to a NumPy array
        preds_array_mci = concatenated_predictions.numpy()

        # Step 3: Perform argmax to get class predictions for each item
        preds_class_mci = np.argmax(preds_array_mci, axis=1)

        # Save the predictions for this iteration
        all_predictions.append(preds_class_mci)

    # Calculate the mean predictions across all iterations
    preds_class_arr, counts = stats.mode(all_predictions, axis=0)
    preds_class = preds_class_arr.squeeze().tolist()
    preds_class_probability = [
        f / config["parameters"]["test_time_iterations"] for f in counts
    ]
    # preds_class_probability = preds_class_probability.squeeze().tolist()

    # Get label names
    labels_family = df_test["family"].tolist()
    # Get unique sorted names from the family_list
    # Create a dictionary to map names to their sorted positions
    name_to_position = {name: i for i, name in enumerate(label_familynames_sorted)}
    # Create the class_int list by mapping names to their positions
    labels_class = [name_to_position[name.lower()] for name in labels_family]

    # calculate accuracy metrics
    acc_balanced = balanced_accuracy_score(labels_class, preds_class)
    # acc_top3 = top_k_accuracy_score(labels_class, preds_arr, k=3)
    acc_kappa = cohen_kappa_score(labels_class, preds_class)
    print("Overall Accuracy:  {:.2f}".format(acc_balanced * 100))
    print("Kappa:  {:.2f}".format(acc_kappa * 100))
    # if there are more labels than prediction classes fix this
    if len(np.unique(preds_class)) != len(np.unique(labels_class)):
        print("Unique Preds: {}".format(np.unique(preds_class)))
        print("Unique Lables: {}".format(np.unique(labels_class)))
        # Get the missing predictions
        missing_preds = set(labels_class) - set(preds_class)
        print("Missing Predictions: {}".format(missing_preds))
        # Get the missing labels
        missing_labels = set(preds_class) - set(labels_class)
        print("Missing Labels: {}".format(missing_labels))
        # Add each missing item from missing_preds to preds_class
        for item in missing_preds:
            preds_class.extend([item])
            labels_class.extend([item])
            preds_class_probability.extend([0.0])
            labels_family.extend([item])
        # Add each missing item from missing_labels to labels_class
        for item in missing_labels:
            preds_class.extend([item])
            labels_class.extend([item])
            preds_class_probability.extend([0.0])
            labels_family.extend([item])
        # preds_class.extend(label_int)
        # labels_class.extend(label_int)

    # get figure path
    log_console_path = get_nth_directory_from_end(checkpoint_path, 2)

    # get confusion matrix
    label_plot_name = [name[:3].capitalize() for name in label_familynames_sorted]
    _ = cm_analysis(
        labels_class,
        preds_class,
        labels=label_plot_name,
        figsize=(13, 13),
        plot=True,
        filename=os.path.join(log_console_path, "cm.png"),
    )

    # Class probability
    # Create a DataFrame
    data = {
        "labels": labels_class,
        "probabilities": preds_class_probability,
        "prediction": preds_class,
        "family": labels_family,
    }

    # Print length of values and their names
    for name, values in data.items():
        print(f"Length of {name}: {len(values)}")

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(log_console_path, "probabilities.csv"))

    # Extract the first three letters of labels_family for legend
    df["family_short"] = df["family"].str[:3]

    # Create a grouped error boxplot using Seaborn
    # plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.boxplot(x="family_short", y="probabilities", data=df)  # , palette="Set3")
    plt.title("Diptera Probabilities")
    plt.xlabel("Family")
    plt.ylabel("Probabilities")
    plt.show()
    plt.savefig(os.path.join(log_console_path, "box.png"), dpi=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/scene_class/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="None",
        help="Path to checkpoint file",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # checkpoint_path
    if args.checkpoint_path == "None":
        model_name = config["parameters"]["model"]
        model_base_path = os.path.join(
            "/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/logs",
            model_name,
        )
        model_dir = [
            d
            for d in os.listdir(model_base_path)
            if os.path.isdir(os.path.join(model_base_path, d))
        ][-1]
        checkpoint_path = glob(
            os.path.join(model_base_path, model_dir, "checkpoints", "*.ckpt")
        )[0]
    else:
        checkpoint_path = args.checkpoint_path

    # get mean step time and train val loss
    log_path = get_nth_directory_from_end(checkpoint_path, 2)
    log_console_path = os.path.join(log_path, "log_console.txt")
    secondsperepoch, train_loss, val_loss = get_console_output(
        log_console_path=log_console_path
    )

    # get predictions
    t0 = time.time()
    run_predict(config, ckpt=checkpoint_path)
    t1 = time.time()
    print(
        "{}-Monte Carlo Interation took:".format(
            config["parameters"]["test_time_iterations"]
        )
    )
    inference_hours, inference_minutes, inference_seconds = convert_seconds_to_hh_mm_ss(
        t1 - t0
    )
    print(f"{inference_hours:02d}:{inference_minutes:02d}:{inference_seconds:02d}")

    # remove lightninglogdir
    lighntinglogdir = os.path.join(os.getcwd(), "lightning_logs")
    rmtree(lighntinglogdir)

    # print time
    epochs = int(checkpoint_path.split("epoch=")[-1].split("-step")[0])
    total_seconds = secondsperepoch * epochs

    train_hours, train_minutes, train_seconds = convert_seconds_to_hh_mm_ss(
        total_seconds
    )
    print("{}-Epcohs took:".format(epochs))
    print(f"{train_hours:02d}:{train_minutes:02d}:{train_seconds:02d}")

    # Write the variables to a text file
    with open(os.path.join(log_path, "train_inference_time.txt"), "w") as output_file:
        output_file.write(
            "{}-Monte Carlo Iteration took:\n".format(
                config["parameters"]["test_time_iterations"]
            )
        )
        output_file.write(
            f"{inference_hours:02d}:{inference_minutes:02d}:{inference_seconds:02d}\n"
        )

        output_file.write("{}-Epochs took:\n".format(epochs))
        output_file.write(
            f"{train_hours:02d}:{train_minutes:02d}:{train_seconds:02d}\n"
        )
