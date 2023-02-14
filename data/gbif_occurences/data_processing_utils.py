# A collection of functions to process data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import sys
import os
import random
from PIL import Image
from IPython.display import HTML


def get_column_names(file_path):
    """
    Function to get the column names from the occurrence.txt file
    without loading the entire file in the RAM.
    """
    with open(file_path, 'r') as f:
        first_line = f.readline()
    column_names = first_line.strip().split('\t')
    return column_names


def display_images(df, column_name, plot_nrows=2, plot_ncols=5, 
                    figsize=(20, 10), fontsize=15):
    """
    Display images from a dataframe column containing image URLs.
    This is useful for visual inspections.
    """
    
    # Create the compound plot.
    fig, axes = plt.subplots(plot_nrows, plot_ncols, figsize=figsize)
    axes = axes.ravel() 

    # Get the image URLs from the dataframe and display them one by one.
    df = df.reset_index()
    for index, row in df.iterrows():
        response = requests.get(row[column_name], stream=True)
        image = Image.open(response.raw)
        image = np.array(image)
        img_height, img_width, _ = image.shape
        axes[index].imshow(image)
        img_name = row['genus'] + ' ' + row['specificEpithet']
        axes[index].set_title(img_name, fontsize=fontsize)
        axes[index].axis('off')
         
    plt.tight_layout()
    plt.show()


def count_lines_in_file(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for line in f)


def read_random_sample(file_path, n_lines):
    """
    Read a random sample of n_lines from a tab separated text file.
    Returns the lines as a pandas dataframe.
    """
    # The first line is skipped, as it contains the column names
    n = count_lines_in_file(file_path) - 1
    # Fix the random seed
    random.seed(42)
    # Generate a sorted list of line numbers to skip, so that 
    # only n_lines random lines are read
    skip = sorted(random.sample(range(1, n + 1), n - n_lines))
    # Read the file, skipping the lines in skip
    sample = pd.read_csv(file_path, sep='\t', dtype=str, on_bad_lines='skip', 
                         encoding="UTF-8", skiprows=skip)
    return sample


def count_missing_vals(df, column1, column2):
    """
    Count the number of records without a value in column1 and no value in column2.
    """
    mask = df[column1].isna() & df[column2].isna()
    n = df[mask].shape[0]
    return f"Number of records without {column1} and no {column2}: {n}"



if __name__ == '__main__':
    print("A collection of functions to process data. Do not run this file directly.")