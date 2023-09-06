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

from confusion_matrix import cm_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, top_k_accuracy_score, cohen_kappa_score

def run_predict(config):
    # Your code goes here
    seed_everything(42, workers=True)

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

    checkpoint = torch.load("/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/scene_class/lightning_logs/lightning_logs/version_0/checkpoints/epoch=13-step=33810.ckpt")
    model.load_state_dict(checkpoint["state_dict"])

    trainer = lit.Trainer(
        accelerator="gpu",
        precision=16,
        devices=torch.cuda.device_count(),
        deterministic=False
    )
    predictions = trainer.predict(model, test_loader)

    pred_batches_tensor = []
    for item in tqdm(predictions):
        batch_items = item[0]
        for batch_item in batch_items:
            pred_batches_tensor.append(batch_item)

    preds_arr = [f.numpy() for f in pred_batches_tensor]
    preds_class = [np.argmax(f) for f in preds_arr]



    labels_family = df_test['family'].tolist()
    # Get unique sorted names from the family_list
    # Create a dictionary to map names to their sorted positions
    name_to_position = {name: i for i, name in enumerate(label_familynames_sorted)}
    # Create the class_int list by mapping names to their positions
    labels_class = [name_to_position[name.lower()] for name in labels_family]
    missing_classes = set(labels_class + preds_class)
    print("Missing the following classe(s):  {}".format(missing_classes))
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
    run_predict(config)