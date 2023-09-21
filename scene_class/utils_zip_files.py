import os
import shutil
import zipfile
from glob import iglob

# Set the source directory
source_dir = '/mnt/ushelf_star_th/data/PAI_diptera/original_data'

# Set the destination directory for the extracted folders
destination_dir = '/mnt/ushelf_star_th/data/PAI_diptera/image_data'

# Set the destination directory for the "json" folder extraction
annotation_dir = '/mnt/ushelf_star_th/data/PAI_diptera/annotation_data'

# Get a list of all files and folders in the source directory
# files = os.listdir(source_dir)
files = list(iglob(source_dir + os.sep + "*.zip"))
files = [f for f in files if not 'json' in f]
# Counter to keep track of the number of folders extracted
folder_count = 0

# Iterate over the files in the source directory
for file in files:
    # file_path = os.path.join(source_dir, file)
    file_name = file.split(os.sep)[-1]
    file_name = file_name.split('.')[0]
    # # Check if the file is a directory
    # if os.path.isdir(file_path):
    # Copy the folder to the destination directory
    new_dir = os.path.join(destination_dir, file_name)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(new_dir)