import os
import pandas as pd
import requests
from tqdm import tqdm

# Define the folder containing the CSV files and the output folder for images
csv_folder = "path/to/csv_folder"
output_folder = "path/to/output_folder"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Base URL for GBIF image downloads
gbif_image_base_url = "https://api.gbif.org/v1/occurrence"

def download_image(gbif_id, output_path):
    """Download an image from GBIF using the gbifID."""
    try:
        # Construct the URL for the occurrence
        url = f"{gbif_image_base_url}/{gbif_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Check if there are media entries
        if "media" in data and data["media"]:
            # Get the first media item (you can modify this to handle multiple images)
            image_url = data["media"][0]["identifier"]
            image_response = requests.get(image_url, stream=True)
            image_response.raise_for_status()

            # Save the image to the output path
            with open(output_path, "wb") as f:
                for chunk in image_response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {output_path}")
        else:
            print(f"No media found for gbifID: {gbif_id}")
    except Exception as e:
        print(f"Failed to download image for gbifID {gbif_id}: {e}")

# Iterate through all CSV files in the folder
for csv_file in os.listdir(csv_folder):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(csv_folder, csv_file)
        print(f"Processing file: {csv_path}")

        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure the gbifID column exists
        if "gbifID" not in df.columns:
            print(f"Skipping file {csv_file}: 'gbifID' column not found.")
            continue

        # Iterate through each gbifID and download the corresponding image
        for gbif_id in tqdm(df["gbifID"].dropna().unique(), desc="Downloading images"):
            output_path = os.path.join(output_folder, f"{gbif_id}.jpg")
            if not os.path.exists(output_path):  # Avoid re-downloading
                download_image(gbif_id, output_path)