import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from glob import iglob
import cv2
from tqdm import tqdm
import numpy as np
import csv


def calculate_mean_std(image_paths):
    # Initialize lists to store channel-wise pixel sums
    sum_r, sum_g, sum_b = 0, 0, 0
    num_pixels = 0

    for image_path in tqdm(image_paths):
        # Open the image using Pillow
        image = cv2.imread(image_path)

        if image is None:
            print('Delete file {} since its corrupted')
            os.remove(image_path)
            continue
        # Convert the image to NumPy array
        img_array = np.array(image)

        # Sum up the pixel values for each channel
        sum_r += np.sum(img_array[:, :, 0])
        sum_g += np.sum(img_array[:, :, 1])
        sum_b += np.sum(img_array[:, :, 2])

        # Count the total number of pixels in all images
        num_pixels += img_array.size // 3

    # Calculate mean for each channel
    mean_r = sum_r / num_pixels
    mean_g = sum_g / num_pixels
    mean_b = sum_b / num_pixels

    # Calculate overall mean (mean of means)
    overall_mean = (mean_r + mean_g + mean_b) / 3

    # Calculate standard deviation for each channel
    sum_sq_diff_r = np.sum((img_array[:, :, 0] - mean_r) ** 2)
    sum_sq_diff_g = np.sum((img_array[:, :, 1] - mean_g) ** 2)
    sum_sq_diff_b = np.sum((img_array[:, :, 2] - mean_b) ** 2)
    std_r = np.sqrt(sum_sq_diff_r / num_pixels)
    std_g = np.sqrt(sum_sq_diff_g / num_pixels)
    std_b = np.sqrt(sum_sq_diff_b / num_pixels)

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

def main(config):

    all_files = list(iglob(config['paths']['data_path'] + os.sep + '*.jpeg'))
    mean, std = calculate_mean_std(all_files)

    data = {
        'Channel': ['R', 'G', 'B'],
        'Mean': mean,
        'Std': std
    }

    with open(config['paths']['images_stats_path'], 'w', newline='') as csvfile:
        fieldnames = ['Channel', 'Mean', 'Std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(data)

    print("Mean and Std saved to:", config['paths']['images_stats_path'])

    print("Mean: (R, G, B)", mean)
    print("Std (R, G, B):", std)
    d=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                default='/.../PAI_diptera/scene_class/config.yaml',
                help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)