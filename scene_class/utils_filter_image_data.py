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
import json
import random

def main(config):

    if not os.path.isfile('/mnt/ushelf_star_th/data/PAI_diptera/annotation_data/json.csv'):
        directory = '/mnt/ushelf_star_th/data/PAI_diptera/annotation_data/json_felicitas'
        # Create an empty DataFrame with the desired column names
        df = pd.DataFrame(columns=['file_path', 'x', 'y', 'h', 'w'])

        for filename in os.listdir(directory):
            if not filename.endswith('.json'):
                continue
            print('Processing {}'.format(filename))
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    for jpeg in data['_via_img_metadata']:
                        jpeg_item = data['_via_img_metadata'][jpeg]
                        jpeg_path = os.path.join('/mnt/ushelf_star_th/data/PAI_diptera/image_data',
                                                 filename.split('.json')[0],
                                                 jpeg_item['filename'])
                        if jpeg_item['regions']:
                            region = jpeg_item['regions'][0]['shape_attributes']

                            df = pd.concat(
                                [df, pd.DataFrame({'file_path': [jpeg_path],
                                                   'x': [region['x']],
                                                   'y': [region['y']],
                                                   'h': [region['height']],
                                                   'w': [region['width']]})])

        # Create a dataframe from the list of valid JSON data
        df.to_csv('/mnt/ushelf_star_th/data/PAI_diptera/annotation_data/json.csv')
    else:
        df = pd.read_csv('/mnt/ushelf_star_th/data/PAI_diptera/annotation_data/json.csv')

    d = 1
    print(df.head())

    # sanity bounding box check
    df_length = len(df)
    random_number = random.randint(0, df_length - 1)
    # item = df['regionBB'].iloc[random_number]
    x, y, width, height = df['x'].iloc[random_number], df['y'].iloc[random_number], \
        df['w'].iloc[random_number], df['h'].iloc[random_number]

    img = cv2.imread(df['file_path'].iloc[random_number])
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
    plt.imshow(img[...,::-1])
    plt.show()
    plt.close()


    # crop image to BB and write to image data selection path
    img_save_base_path = r'/mnt/ushelf_star_th/data/PAI_diptera/image_data_filtered'

    # Ensure the output directory exists, create it if it doesn't
    os.makedirs(img_save_base_path, exist_ok=True)

    # Initialize tqdm with the total number of rows in the DataFrame
    total_rows = len(df)
    with tqdm(total=total_rows, unit="image") as pbar:
        # Loop through the DataFrame
        for index, row in df.iterrows():
            file_path = row['file_path']

            # Get the filename from the original file path
            _, filename = os.path.split(file_path)

            # Save the cropped image with the same filename in the output directory
            output_path = os.path.join(img_save_base_path, filename)

            if os.path.exists(output_path):
                continue
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            x = max(0, x)
            y = max(0, y)
            # Read the image using OpenCV
            image = cv2.imread(file_path)

            # Crop the image based on the bounding box coordinates
            cropped_image = image[y:y + h, x:x + w]


            cv2.imwrite(output_path, cropped_image)

            # Update the progress bar
            pbar.update(1)

    d=2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                default='/mnt/ushelf_star_th/projects/2023_PAI/2023_PAI_diptera/PAI_diptera/scene_class/config.yaml',
                help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)