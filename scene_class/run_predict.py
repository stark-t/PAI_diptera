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


def run_predict(config, ckpt='checkpoint_path'):
    seed_everything(config['parameters']['seed'], workers=True)

    label_familynames = [entry.name.lower() for entry in os.scandir(config['paths']['original_data_path']) if
                  entry.is_dir()]
    label_familynames_sorted = sorted(label_familynames)
    label_familynames_sorted = label_familynames_sorted[:-3]
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

    test_set = Dataset(path=df_test['file_path'].tolist(), config=config, norm=config['parameters']['normalization'])
    test_loader = DataLoader(test_set, batch_size=16, num_workers=10)

    if config['parameters']['verbose'] > 1:
        grouped_df = df_train.groupby("family").nunique()
        print(grouped_df)
        randints = torch.randint(low=0, high=len(test_set), size=(10,))
        for i in randints:
            image, label = test_set[i]
            plt.imshow(image.numpy()[0, :, :])
            plt.title(str(label))
            plt.show()
            plt.close()


    model = LitClassifier(config=config)

    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint["state_dict"])

    trainer = lit.Trainer(
        accelerator="gpu",
        precision=16,
        devices=torch.cuda.device_count(),
        deterministic=False
    )
    predictions = trainer.predict(model, test_loader)

    # Step 1: Concatenate all batched predictions into a single tensor
    concatenated_predictions = torch.cat(predictions, dim=0)

    # Step 2: Convert the concatenated tensor to a NumPy array
    preds_array = concatenated_predictions.numpy()

    # Step 3: Perform argmax to get class predictions for each item
    preds_class = np.argmax(preds_array, axis=1)

    # Get label names
    labels_family = df_test['family'].tolist()
    # Get unique sorted names from the family_list
    # Create a dictionary to map names to their sorted positions
    name_to_position = {name: i for i, name in enumerate(label_familynames_sorted)}
    # # Create the class_int list by mapping names to their positions
    labels_class = [name_to_position[name.lower()] for name in labels_family]

    # calculate accuracy metrics
    acc_balanced = balanced_accuracy_score(labels_class, preds_class)
    # acc_top3 = top_k_accuracy_score(labels_class, preds_arr, k=3)
    acc_kappa = cohen_kappa_score(labels_class, preds_class)
    print("Overall Accuracy:  {:.2f}".format(acc_balanced*100))
    print("Kappa:  {:.2f}".format(acc_kappa*100))
    # if there are more labels than prediction classes fix this
    if len(np.unique(preds_class)) != len(np.unique(labels_class)):
        preds_class.extend(label_int)
        labels_class.extend(label_int)

    # get confusion matrix
    cm_pd = cm_analysis(labels_class, preds_class, labels=label_int, plot=True)

    d=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                default='/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/scene_class/config.yaml',
                help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    # checkpoint_path
    checkpoint_path = '/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/scene_class/logs/resnet18/23091809/checkpoints/epoch=15-step=38640.ckpt'

    # get mean step time and train val loss
    log_console_path = get_nth_directory_from_end(checkpoint_path, 2)
    log_console_path = os.path.join(log_console_path, 'log_console.txt')
    secondsperepoch, train_loss, val_loss = get_console_output(log_console_path=log_console_path)

    # get predictions
    run_predict(config, ckpt=checkpoint_path)

    # remove lightninglogdir
    lighntinglogdir = os.path.join(os.getcwd(), 'lightning_logs')
    rmtree(lighntinglogdir)

    # print time
    epochs = int(checkpoint_path.split('epoch=')[-1].split('-step')[0])
    total_seconds = secondsperepoch * epochs

    hours, minutes, seconds = convert_seconds_to_hh_mm_ss(total_seconds)
    print('{}-Epcohs took:'.format(epochs))
    print(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
