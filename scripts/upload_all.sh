#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -d <directory-path> -e <non-zero-integer> [-w <non-zero-integer>]"
    exit 1
}

# Function to clean up child processes
cleanup() {
    echo "Terminating all child processes..."
    kill 0
    exit 1
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Initialize variables
DIR_PATH=""
EVERY=""
WAIT=""

# Parse command-line arguments with getopts
while getopts "d:e:w:" opt; do
    case $opt in
        d) DIR_PATH="$OPTARG" ;;  # Directory path
        e) EVERY="$OPTARG" ;;                 # Every value
        w) WAIT="$OPTARG" ;;                  # Initial wait time in minutes
        *) usage ;;                           # Invalid option
    esac
done

# Validate the provided arguments
if [ -z "$DIR_PATH" ] || [ -z "$EVERY" ]; then
    echo "Error: Both -d (directory path) and -e (non-zero integer) are required."
    usage
fi

# Validate the EVERY value to ensure it's a non-zero integer
if ! [[ "$EVERY" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: The -e value must be a non-zero integer."
    usage
fi

# Validate the optional WAIT value to ensure it's a non-zero integer
if [ -n "$WAIT" ] && ! [[ "$WAIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: The -w value must be a non-zero integer."
    usage
fi

# Wait for the specified time before starting the script
if [ -n "$WAIT" ]; then
    echo "Waiting for $WAIT minutes before starting the script..."
    sleep "$WAIT"m
fi

DIR_PATH=$(realpath "$DIR_PATH")

# Get the directory of the current script
CURR_DIR=$(dirname "$(realpath "$0")")

# Extract the stem of the given directory path
DUMP=$(basename "$DIR_PATH")

# Loop through all direct subdirectories
for SUBDIR in "$DIR_PATH"/*/; do
    # Check if it is a directory
    if [ -d "$SUBDIR" ]; then
        # Extract the stem of the subdirectory
        LANGUAGE=$(basename "$SUBDIR")
        
        # Execute the Python script with the required arguments
        echo "Executing: python \"$CURR_DIR/upload.py\" \"$SUBDIR\" -c \"$DUMP--$LANGUAGE\" --every $EVERY"
        python "$CURR_DIR/upload.py" --local_path "$SUBDIR" --config_name "$DUMP--$LANGUAGE" --hf_repo BramVanroy/CommonCrawl-CreativeCommons --robust --every "$EVERY" --include_text &
        # Sleep to avoid race-conditions when creating the repository
        sleep 30
    fi
done

wait