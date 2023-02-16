# Script to download images from the data frame with the URLs and file names.
# The needed columns in the data frame must be named 'identifier' and 'file_name'.

# Usage:

# python3 url_download.py --df_path path/to/df.txt --img_dir path/to/img/destination/folder/ --family FamilyName

# Example for the Windows terminal server, run in the Power Shell:

# cd I:\Artificial-Intelligence\pai_p3_diptera_families\data\gbif_occurences
# python url_download.py --df_path "I:\Artificial-Intelligence\pai_p3_diptera_families\data\gbif_occurences\sampled\df_sample_20230213.txt" --img_dir "I:\Artificial-Intelligence\pai_p3_diptera_families\data\gbif_occurences\sampled\images\" --family "Calliphoridae"

# Example for Linux:

# python3 url_download.py \
# --df_path "./sampled/df_sample_20230213.txt" \
# --img_dir "./sampled/images" \
# --family "Calliphoridae" # quotes are optional


import argparse
import requests
import time
import json
import os
import pandas as pd


# Create a parser to get the arguments from the command line.
parser = argparse.ArgumentParser(
    description="Download images from a data frame with the URLs and file names.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--df_path", type=str, help="path/to/df.txt")
parser.add_argument("--img_dir", type=str, help="path/to/destination/folder/images/")
parser.add_argument("--family", type=str, help="Family name, e.g.: 'Hybotidae'")
args = parser.parse_args()



def download_image(url: str, file_name: str, path: str):
    """
    Download an image from a URL and save it to the disk.

    Parameters:
    - `url`: the URL of the image.
    - `file_name`: the name of the file (without extension) to save the image to.
        The extension will be guessed from the content type of the URL response.
    - `path`: the path to save the image to.
    """
    response = requests.get(url, stream=True, timeout=5)
    # Check if the response is successful
    if response.status_code == 200:
        # Get the content type from the response headers
        content_type = response.headers.get('content-type')
        # Attempt to to guess the file extension.
        # The content type can be something like 'image/jpeg', 'image/png', etc.
        ext = content_type.split('/')[-1]
        # Save the image to the disk
        img_path = os.path.join(path, file_name + '.' + ext)
        with open(img_path, 'wb') as f:
            # Iterate over the response data using 1KB chunks
            for chunk in response.iter_content(chunk_size=1024):
                if chunk: # write only alive chunks
                    f.write(chunk)
    else:
        # If the response was not successful, raise an exception
        response.raise_for_status()




if __name__ == '__main__':
    # Time the execution
    start = time.perf_counter()

    # Get the data frame path from the command line, then read it the df.
    df_path = args.df_path
    df = pd.read_csv(df_path, sep='\t', dtype=str, encoding="UTF-8")

    # Get the path to save the images to.
    img_dir_family = os.path.join(args.img_dir, args.family)
    if not os.path.exists(args.img_dir):
        os.mkdir(args.img_dir)
    # Check if the family directory already exists. If it does, raise an error.
    if os.path.exists(img_dir_family):
        raise ValueError(f"The directory {img_dir_family} already exists.")
    else:
        os.mkdir(img_dir_family)        

    # Filter for the given family
    df = df[df['family'].isin([args.family])]
    # Reset the index. This is important for the indexing in the loop below.
    df = df.reset_index(drop=True)

    # Download the images sequentially. I tried to use multiprocessing, but it
    # didn't work. I think multiple requests in parallel from the same IP address
    # are blocked by the server or they time out.
    errors_msg = []
    errors_ids = []
    for i in range(df.shape[0]):
        try:
            url = df.at[i, 'identifier']
            file_name = df.at[i, 'file_name']
            download_image(url, file_name, img_dir_family)
        except Exception as e:
            errors_ids.append(i)
            errors_msg.append(e)
            print(f'Error downloading file {file_name} at {url}, row {i}: {e}')


    # Save the errors_msg & errors_ids to txt files.
    err_dir = os.path.join(args.img_dir, args.family + '_errors')
    os.mkdir(err_dir)

    path_errors_msg= os.path.join(err_dir, 'errors.txt')
    with open(file=path_errors_msg, mode='w', encoding='utf-8') as file:
        # Use the json.dump() function to save the list to the file.
        # Convert HTTPError objects to strings before saving, otherwise the
        # json.dump() function will raise the error:
        # TypeError: Object of type HTTPError is not JSON serializable
        errors_msg = [str(error) for error in errors_msg]
        json.dump(errors_msg, file)
    
    path_errors_ids = os.path.join(err_dir, 'errors_i.txt')
    with open(file=path_errors_ids, mode='w', encoding='utf-8') as file:
        json.dump(errors_ids, file)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
