#!/bin/bash

# Script to display information about the used environment and hardware in the *.log files. 
# This is useful for assuring reproducibility of our results as much as possible.
# See also https://wiki.ufz.de/eve/index.php/Reproducible_Research


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


# Display information about the operating system
echo '========================================================================'
echo 'Output of: cat /etc/os-release'
echo '========================================================================'
cat /etc/os-release
printf '\n'


# Display the current hostname, operating system release, kernel version, and 
# other system information
echo '========================================================================'
echo 'Linux host: hostnamectl'
echo '========================================================================'
hostnamectl
printf '\n'


# Display the current locale settings, including the language, character encoding, 
# date and time formatting, monetary formatting, etc.
echo '========================================================================'
echo 'Information about the current locale'
echo '========================================================================'
locale
printf '\n'


# List all currently loaded kernel modules in the Linux system
echo '========================================================================'
echo 'List of Linux kernel loaded modules (in alphabetical order)'
echo '========================================================================'
lsmod | sort
printf '\n'


# List all currently loaded modules in the environment.
echo '========================================================================'
echo 'List the toolchain (in alphabetical order)'
echo 'EasyBuild was used by EVE cluster administrators to deploy software.'
echo '========================================================================'
module --terse list | sort
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


# Display all environment variables. This will be long!
echo '========================================================================'
echo 'List all environment variables'
echo '========================================================================'
env | sort
printf '\n'