# Script to download images from a data frame with the URLs and file names.

# Usage:
# python3 url_download.py path/to/df.txt path/to/img/destination/folder/
# or
# python url_download.py path/to/df.txt path/to/img/destination/folder/
# Note that the path to images must end with a slash.

# Example for the Windows terminal server, run in the Power Shell:
# cd I:\Artificial-Intelligence\pai_p3_diptera_families\data\gbif_occurences
# python url_download.py "I:\Artificial-Intelligence\pai_p3_diptera_families\data\gbif_occurences\sampled\df_sample_20230213.txt" "I:\Artificial-Intelligence\pai_p3_diptera_families\data\gbif_occurences\sampled\images\"

import requests
import time
import sys
import json
import pandas as pd
# import concurrent.futures



def download_image(url: str, file_name: str, path: str):
    """
    Download an image from a URL and save it to the disk.

    Parameters:
    - `url`: the URL of the image.
    - `file_name`: the name of the file (without extension) to save the image to.
        The extension will be guessed from the content type of the URL response.
    - `path`: the path to save the image to.
    """
    try:
        response = requests.get(url, stream=True, timeout=5)
        # Check if the response is successful
        if response.status_code == 200:
            # Get the content type from the response headers
            content_type = response.headers.get('content-type')
            # Attempt to to guess the file extension.
            # The content type can be something like 'image/jpeg', 'image/png', etc.
            ext = content_type.split('/')[-1]
            # Save the image to the disk
            with open(path + file_name + "." + ext, 'wb') as f:
                # Iterate over the response data using 1KB chunks
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk: # write only alive chunks
                        f.write(chunk)
    except Exception as e:
        print(f'Error downloading file {file_name} at {url}: {e}')



def download_images_concurrent(urls: list, files: list, path: str):
    """
    Download images concurrently using a thread pool.
    The parallel download doesn't work fully though. Not sure why at the moment. 
    It might be that the servers do not allow multiple downloads at the same time 
    from the same IP (e.g the servers of iNaturalist).
    """
    # See https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_image, url, file_name, path) for url, file_name, path in zip(urls, files, path)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print('%r generated an exception: %s' % e)



if __name__ == '__main__':
    # Time the execution
    start = time.perf_counter()

    # The parallel download doesn't work though. Download the images sequentially
    # download_images_concurrent(urls, files, path)

    # Get the data frame path from the command line
    df_path = sys.argv[1]
    df = pd.read_csv(df_path, sep='\t', dtype=str, encoding="UTF-8")

    # Test filter for the first 3 rows only
    # df = df.head(3)

    # Filter only for these families for now:
    families = ['Hybotidae', 'Sarcophagidae', 'Fanniidae']
    df = df[df['family'].isin(families)]
    # Reset the index. This is important for the indexing in the loop below.
    df = df.reset_index(drop=True)

    # Get the path to save the images to from the command line
    path = sys.argv[2]


    # Download the images sequentially
    errors = []
    errors_i = []
    for i in range(df.shape[0]):
        try:
            url = df.at[i, 'identifier']
            file_name = df.at[i, 'file_name']
            download_image(url, file_name, path)
        except Exception as e:
            errors_i.append(i)
            errors.append(e)
            print(f'Error downloading file {file_name} at {url}, row {i}: {e}')


    # Save the errors to a file
    with open(path + 'errors.txt', 'w') as file:
        # Use the json.dump() function to save the list to the file
        json.dump(errors, file)
    
    with open(path + 'errors_i.txt', 'w') as file:
        # Use the json.dump() function to save the list to the file
        json.dump(errors_i, file)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
