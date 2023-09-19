import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
import array

import torch
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset



class Dataset(TorchDataset):
    """Tabular and Image dataset."""

    def __init__(self, path, config,  norm='simple_noramlization'):
        self.path = path
        self.norm = norm
        self.config = config
        self.file_paths = path  # list(iglob(self.path))


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image stats
        if self.config['parameters']['normalization'] == 'znorm':
            stats = pd.read_csv(self.config["paths"]["images_stats_path"])
            img_means = get_stats(stats['Mean'].to_list())
            img_stds = get_stats(stats['Std'].to_list())

        # read image
        image = cv2.imread(self.file_paths[idx])
        image = cv2.resize(image, (self.config["parameters"]["image_size"], self.config["parameters"]["image_size"]))
        if self.config['parameters']['normalization'] == 'znorm':
            image = normalize(image, img_means, img_stds)
        elif self.config['parameters']['normalization'] == 'imagenet':
            image = normalize_imagenet(image)
        else:
            image = normalize_simple(image)

        # get label
        label_family = self.file_paths[idx].split('/')[-1].split('_')[0].lower()

        labels_all = [entry.name.lower() for entry in os.scandir(self.config['paths']['original_data_path']) if entry.is_dir()]
        labels_all = sorted(labels_all)
        label = labels_all.index(label_family)

        label = np.array(label)
        label_tensor = torch.from_numpy(label).to(torch.long)

        return torch.FloatTensor(image.transpose(2, 0, 1)), label_tensor

def get_stats(stats_list):
    stats = stats_list[0].strip("()")
    stats = stats.split(',')
    stats = [float(f) for f in stats]
    return array.array('f', stats)

def normalize(arr: np.array, means: np.ndarray, stds: np.ndarray) -> np.array:
    """Z-score normalization a 3D array with 1D statistics."""
    arr = arr - means
    arr = arr / stds
    arr = arr / 255.0
    return arr

def normalize_simple(arr: np.array) -> np.array:
    """simple normalization a 3D array with 1D statistics."""
    arr = arr / 255.0
    return arr

def normalize_imagenet(arr: np.array) -> np.array:
    """imagenet stats normalization a 3D array with 1D statistics."""
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    arr = arr / 255.0
    arr = arr - means
    arr = arr / stds
    return arr

