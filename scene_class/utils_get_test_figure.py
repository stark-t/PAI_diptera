import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from glob import iglob
from sklearn.model_selection import train_test_split
import shutil
import torch
from pytorch_lightning import seed_everything
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
        df_train = pd.concat([df_train, group_train])
        df_val = pd.concat([df_val, group_val])
        df_test = pd.concat([df_test, group_test])


    # Create a markdown table
    print("| Family | Number of Images | Unique Species | Unique Genus |")
    print("|--------|-----------------|----------------|--------------|")
    for family in sorted(file_df["family"].unique()):
        num_images = file_df[file_df["family"] == family].shape[0]
        unique_species = file_df[file_df["family"] == family]["species"].nunique()
        unique_genus = file_df[file_df["family"] == family]["genus"].nunique()
        print(f"| {family} | {num_images} | {unique_species} | {unique_genus} |")

    print("| Family | Train | Validation | Test |")
    print("|--------|-------|------------|------|")
    for family in sorted(df_train["family"].unique()):
        train_count = df_train[df_train["family"] == family].shape[0]
        val_count = df_val[df_val["family"] == family].shape[0]
        test_count = df_test[df_test["family"] == family].shape[0]
        print(f"| {family} | {train_count} | {val_count} | {test_count} |")


    # print("| Family | Train Species | Train Genus | Validation Species | Validation Genus | Test Species | Test Genus |")
    # print("|--------|---------------|-------------|--------------------|------------------|--------------|------------|")
    # for family in sorted(df_train["family"].unique()):
    #     train_species_count = df_train[df_train["family"] == family]["species"].nunique()
    #     train_genus_count = df_train[df_train["family"] == family]["genus"].nunique()
    #     val_species_count = df_val[df_val["family"] == family]["species"].nunique()
    #     val_genus_count = df_val[df_val["family"] == family]["genus"].nunique()
    #     test_species_count = df_test[df_test["family"] == family]["species"].nunique()
    #     test_genus_count = df_test[df_test["family"] == family]["genus"].nunique()
    #     print(f"| {family} | {train_species_count} | {train_genus_count} | {val_species_count} | {val_genus_count} | {test_species_count} | {test_genus_count} |")

    df_test.reset_index(drop=True, inplace=True)
    df_test["index"] = df_test.index
    print(df_test.head())
    probability_files_BB = list(iglob("data/results/pretrained_BB/*.csv"))
    probability_files_noBB = list(iglob("data/results/pretrained_noBB/*.csv"))

    # Read all csv files as separate dataframes
    df_modellist = []
    for file in probability_files_BB + probability_files_noBB:
        if "noBB" in file:
            BB = False
        else:
            BB = True

        modelname = file.split(os.sep)[-1].split(".csv")[0]

        df = pd.read_csv(file)

        df["modelname"] = modelname
        df["BB"] = BB
        df["index"] = df.index

        df_merge = df_test.merge(df, on="index")

        df_modellist.append(df_merge)

    dfs = pd.concat(df_modellist)

    dfs.drop(columns=["family_y"], inplace=True)
    dfs.rename(columns={"family_x": "family"}, inplace=True)
    print(dfs.columns)

    # Group the dataframe by "file_path"
    grouped_by_file = dfs.groupby("file_path")

    # Calculate the standard deviation for the probabilities column
    df_std = grouped_by_file["probabilities"].std()
    df_std = pd.DataFrame(df_std)
    df_std.rename(columns={"probabilities": "std"}, inplace=True)

    dfs = dfs.merge(df_std, on="file_path", how="left")

    dfs.drop(columns=["file_name", "genus", "index", "Unnamed: 0"], inplace=True)

    print(dfs.head())

    # Get some wrong classifications
    fanniidae_wrong_classifications = dfs[
        (dfs["family"] == "Fanniidae")
        & ((dfs["prediction"] == 7) | (dfs["prediction"] == 14))
        & (dfs["modelname"] == "EfficientNet_b4")
        & (dfs["BB"] == True)
    ]
    print("fanniidae_wrong_classifications")
    print(fanniidae_wrong_classifications["file_path"].apply(os.path.basename))

    muscidae_wrong_classifications = dfs[
        (dfs["family"] == "Muscidae")
        & ((dfs["prediction"] == 5) | (dfs["prediction"] == 14))
        & (dfs["modelname"] == "EfficientNet_b4")
        & (dfs["BB"] == True)
    ]
    print("muscidae_wrong_classifications")
    print(muscidae_wrong_classifications["file_path"].apply(os.path.basename))

    tachinidae_wrong_classifications = dfs[
        (dfs["family"] == "Tachinidae")
        & ((dfs["prediction"] == 5) | (dfs["prediction"] == 7))
        & (dfs["modelname"] == "EfficientNet_b4")
        & (dfs["BB"] == True)
    ]
    print("tachinidae_wrong_classifications")
    print(tachinidae_wrong_classifications["file_path"].apply(os.path.basename))

    # Concatenate the dataframes into a single one
    wrong_classifications = pd.concat(
        [
            fanniidae_wrong_classifications,
            muscidae_wrong_classifications,
            tachinidae_wrong_classifications,
        ]
    )

    # Save the concatenated dataframe as a CSV file
    save_path = "/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/paper/figures/all_images"
    wrong_classifications.to_csv(
        os.path.join(save_path, "wrong_classifications.csv"), index=False
    )

    # Copy files in wrong_classifications filepath to save_path
    save_path_wrong_classifications = os.path.join(save_path, "wrong_classifications")
    os.makedirs(save_path_wrong_classifications, exist_ok=True)
    for index, row in wrong_classifications.iterrows():
        file_name = os.path.basename(row["file_path"])

        src_path = os.path.join(
            "/mnt/data2/PAI_diptera/image_data", row["family"], file_name
        )
        destination_path = os.path.join(save_path_wrong_classifications, file_name)
        shutil.copy(src_path, destination_path)

    dfs = dfs[dfs["labels"] == dfs["prediction"]]

    # Group the dataframe by "family"
    grouped_by_family = dfs.groupby("family")

    # Iterate over each group
    records = []
    for family, group in grouped_by_family:

        # Sort group by "std" column
        group = group.sort_values(by="std", ascending=False)

        same_std_rows = group[
            group["std"].duplicated(keep=False)
            & (group["std"].map(group["std"].value_counts()) == 6)
        ]

        # Save df to csv
        save_path = "/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/paper/figures/all_images"
        save_path_family = os.path.join(save_path, family)
        os.makedirs(save_path_family, exist_ok=True)
        same_std_rows.to_csv(os.path.join(save_path_family, "stats.csv"), index=False)

        # Copy file_path to save_path
        save_path_family_images = os.path.join(save_path, family, "images")
        os.makedirs(save_path_family_images, exist_ok=True)
        for index, row in same_std_rows.iterrows():
            file_name = os.path.basename(row["file_path"])

            src_path = os.path.join(
                "/mnt/data2/PAI_diptera/image_data", family, file_name
            )
            destination_path = os.path.join(save_path_family_images, file_name)
            shutil.copy(src_path, destination_path)


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
