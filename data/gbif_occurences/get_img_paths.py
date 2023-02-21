# Description: Create a txt file with the paths to images from a directory.

# Usage:

# python get_img_paths.py \
# --img_dir path/to/images/dir \
# --output path/to/output/images.txt


import os
import argparse


# Create a parser to get the arguments from the command line.
parser = argparse.ArgumentParser(
    description="Create a txt file with the paths to images from a directory.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--img_dir", type=str, help="path/to/images/dir")
parser.add_argument("--output", type=str, help="path/to/output/images.txt")
args = parser.parse_args()


# Get the image directory path from the command line.
img_dir = args.img_dir

# Get the output path from the command line.
output = args.output


# Create a list of the paths to the images. Use os.walk()
# to get the paths to the images in the directory and its subdirectories.
img_paths = []
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file != 'Thumbs.db':   # Filter out Thumbs.db files
            img_paths.append(os.path.join(root, file))

# Sort the list of image paths alphabetically
img_paths.sort()

# Write the paths to the images to a txt file.
with open(output, 'w') as f:
    for path in img_paths:
        f.write(path + '\n')

print(f"{len(img_paths)} lines written to {output}")
