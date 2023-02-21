#!/bin/bash

# Script to display information about the used environment and hardware.

# Disply NVIDIA System Management Interface and store it in the output log file.
echo '========================================================================'
echo 'Output of: nvidia-smi'
echo '========================================================================'
nvidia-smi
printf '\n'


# Display the name of to virtual environment folder passed as the $1 argument to this script. 
# Must be the name of the environment created at .../env/
# For example, $1 can be yolov5 or yolov8
echo '========================================================================'
echo 'Virtual environment name: '$1
echo '========================================================================'
printf '\n'


echo '========================================================================'
echo 'Output of: cat /etc/os-release'
echo '========================================================================'
cat /etc/os-release
printf '\n'


echo '========================================================================'
echo 'Linux host: hostnamectl'
echo '========================================================================'
hostnamectl
printf '\n'


echo '========================================================================'
echo 'Information about the current locale'
echo '========================================================================'
locale
printf '\n'


echo '========================================================================'
echo 'List of Linux kernel loaded modules (in alphabetical order)'
echo '========================================================================'
lsmod | sort
printf '\n'


# Print loaded Python version
echo '========================================================================'
echo 'Python version'
echo '========================================================================'
python -c 'import sys; print(sys.version)'
printf '\n'


echo '========================================================================'
echo 'List of Python packages installed in the environment (in alphabetical order)'
echo '========================================================================'
pip list # pip displays the packages and their verison in alphabetical order by default.
printf '\n'