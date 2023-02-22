#!/bin/bash

# Script to display information about the used environment and hardware in the *.log files. 
# This is useful for assuring reproducibility of our results as much as possible.
# See also https://wiki.ufz.de/eve/index.php/Reproducible_Research

# Arguments:
# $1: name of the virtual environment folder (e.g. yolov5, created at .../env/)
# $2: path to the log file. Needed for force-printing some of the information below suing 2>&1.
# $3: path to the repository of the used architecture. Needed for printing the git tag.

# For diagnostic purposes, print the arguments passed to the script
echo '========================================================================'
echo 'Virtual environment name: '$1
echo 'log file path: '$2
echo 'Architecture repository path: '$3
echo '========================================================================'
printf '\n'


# Disply NVIDIA System Management Interface and store it in the output log file.
echo '========================================================================'
echo 'Output of: nvidia-smi'
echo '========================================================================'
nvidia-smi
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
# NOTE: 
# I had to use 2>&1 so that it redirects the output to the log file, instead of
# printing it to the error file. It seems that `module --terse list` prints in
# the standard error by default and not in the standard output (*.log file).
# Also, `module --terse list | sort >> $2 2>&1` still prints in the error file.
echo '========================================================================'
echo 'List the toolchain modules'
echo 'EasyBuild was used by EVE cluster administrators to deploy software.'
echo '========================================================================'
module --terse list >> $2 2>&1
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


# cd to the architecture repository and print the git tag
echo '========================================================================'
echo 'Git tag of the architecture:'
echo '========================================================================'
cd $3
git describe --tags >> $2 2>&1
echo 'Architecture repository path: '$3
printf '\n'


# Display all environment variables. This will be long!
echo '========================================================================'
echo 'List all environment variables'
echo '========================================================================'
env | sort
printf '\n'