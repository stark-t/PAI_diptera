#!/bin/bash

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Slurm job options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The requested GPU resources with hardware constraints and requested time 
# will be passed at execution time to the sbatch command like:
# sbatch --constraint='a100-vram-40G' --time='0-10:00:00' /path/to/this_script.sh ...
# The requested time must be given in the format d-hh:mm:ss or hh:mm:ss.

#SBATCH --job-name=detect_y5
#SBATCH --cpus-per-task=4 # Number of CPUs requested
#SBATCH --mem-per-cpu=8G # RAM per CPU core requested

# Define paths for job-id.log & job-id.err files:
#SBATCH --output=/data/idiv_knight/Valentin/PAI_diptera/architectures/logs_detect_jobs/%j.log
#SBATCH --error=/data/idiv_knight/Valentin/PAI_diptera/architectures/logs_detect_jobs/%j.err

# Define email options:
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Variables & Arguments/positional parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Check if the number of arguments passed is as expected.
if [ $# -ne 7 ]; then
    echo "Error: Expecting 2 sbatch options and 7 arguments."
    
    echo "Usage:
    sbatch <options> \
    /path/to/this_script.sh \
    repo_path \
    weights_path \
    source_path \
    img_size \
    conf \
    iou \
    max_det

    Example:
    
    sbatch --constraint='a100-vram-40G' --time='0-10:00:00' \
    /data/idiv_knight/Valentin/PAI_diptera/data/gbif_occurences/detect_insects_yolov5.sh \
    /data/idiv_knight/Valentin/PAI_diptera \
    architectures/yolov5/weights_paip1/s_best.pt \
    data/gbif_occurences/sampled/images.txt \
    640 \
    0.3 \
    0.6
    1000"
    exit 1
else
    echo "You passed these arguments: $@"
    echo "repo_path: $1"
    echo "weights_path: $2"
    echo "source_path: $3"
    echo "img_size: $4"
    echo "conf: $5"
    echo "iou: $6"
    echo "max_det: $7"
fi

# Assign the arguments to variables.

repo_path=$1
# e.g. '/data/idiv_knight/Valentin/PAI_diptera'
weights_path=$2 # path to weights file (relative to project path)
# e.g. 'architectures/yolov5/weights_paip1/s_best.pt'
weights="$repo_path"/"$weights_path"
source_path=$3  # path to folder with images or a txt file with absolute paths to images (one path per line)
# e.g. 'data/gbif_occurences/sampled/images.txt'
source="$repo_path"/"$source_path"
img_size=$4 # e.g. 640
conf=$5 # confidence threshold
iou=$6 # NMS IoU threshold
max_det=$7 # maximum number of detections per image


# Create the name of the results folder based on the given input arguments
output_folder=predlbl_imgsize_"$img_size"_maxdet_"$max_det"_conf_"$conf"_iou_"$iou"

printf "\n"
echo "The detection results will be stored at:"
echo "$repo_path"/architectures/yolov5/runs/detect/job_"$SLURM_JOB_ID"/"$output_folder"
printf "\n"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cd $repo_path

# Start with a clean environment
module purge
# Load the needed modules from the software tree 
# (same ones used when we created the environment)
# module load fosscuda/2020b TensorFlow/2.5.0
module load fosscuda/2019b TensorFlow/2.1.0-Python-3.7.4
# Activate corresponding virtual environment
source env/yolov5_2019b_tf21/bin/activate

# Call the helper script session_info.sh which will print in the *.log file info 
# about the used environment and hardware.
source utils/session_info.sh yolov5_2019b_tf21
# The argument here, passed to $1, is the environment name set at .../env/
# Use `source` instead of `bash`, so that session_info.sh describes the environment 
# activated in this script (the parent script from which it is called). 
# See https://askubuntu.com/a/965496/772524


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Detect
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cd "$repo_path"/architectures/yolov5

python detect.py \
--weights $weights \
--source $source \
--img-size $img_size \
--conf-thres $conf \
--iou-thres $iou \
--max-det $max_det \
--save-txt \
--save-conf \
--nosave \
--project runs/detect/job_"$SLURM_JOB_ID" \
--name $output_folder

# Deactivate virtual environment
deactivate