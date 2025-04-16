import argparse
import yaml
from tqdm import tqdm
from glob import iglob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
import pytorch_lightning as lit

from dataset import Dataset
from model_lit import LitClassifier
from utils_get_console_output import get_console_output

from confusion_matrix import cm_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, top_k_accuracy_score, cohen_kappa_score
from shutil import rmtree

def convert_seconds_to_hh_mm_ss(seconds):
    hours, remainder = divmod(int(seconds), 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute
    return hours, minutes, seconds



def get_nth_directory_from_end(path, n):
    # Split the path into its components and reverse the list
    components = path.split(os.path.sep)[:-n]
    return os.sep + os.path.join(*components)


def run_info(config):
    seed_everything(config['parameters']['seed'], workers=True)

    label_familynames = [entry.name.lower() for entry in os.scandir(config['paths']['original_data_path']) if
                  entry.is_dir()]
    label_familynames_sorted = sorted(label_familynames)
    # label_familynames_sorted = label_familynames_sorted[:-3]
    label_int = list(range(len(label_familynames_sorted)))

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

    # test_set = Dataset(path=df_test['file_path'].tolist(), config=config, norm=config['parameters']['normalization'])
    # test_loader = DataLoader(test_set, batch_size=16, num_workers=10)

    dfs = [df_train, df_val, df_test]
    dfs_names = ['Train', 'Validation', 'Test']  # Names for the dataframes

    # Iterate through the dataframes and create markdown tables
    for i, df in enumerate(dfs):
        # Calculate unique counts for species and genus for each family
        family_counts = df.groupby('family').agg({'species': 'nunique', 'genus': 'nunique'}).reset_index()

        # Rename the columns for clarity
        family_counts = family_counts.rename(columns={'species': 'Unique Species', 'genus': 'Unique Genus'})

        # Create a markdown table
        markdown_table = family_counts.to_markdown(index=False)

        # Print the markdown table with the dataframe name
        print(f"### {dfs_names[i]} Dataframe\n")
        print(markdown_table)
        print("\n")

    # Create a list of dataframes
    dataframes = [df_train, df_val, df_test]

    # Initialize a dictionary to store the results
    results = {}

    # Loop through the dataframes and count unique occurrences in the 'species' column
    for i, df in enumerate(dataframes, start=1):
        unique_counts = df['family'].value_counts()
        results[f'df_{i}'] = unique_counts

    # Print the results in Markdown table format
    print("| Dataframe | Unique family | Count |")
    print("|-----------|----------------|-------|")

    for df_name, counts in results.items():
        for family, count in counts.items():
            print(f"| {df_name} | {family} | {count} |")

    d=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                default='/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/scene_class/config.yaml',
                help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # get predictions
    run_info(config)

