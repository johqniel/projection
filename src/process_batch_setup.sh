#!/bin/bash

# NOTE: Run this script from the project root directory!
# ./src/process_batch_setup.sh

# List of file IDs to process (Interactive Setup)
FILES=(
  "00018"
  "00019"
  "00020"
  "00021"
  "00022"
  "00023"
  "00024"
  "00027"
)

# Directory containing samples relative to project root
SAMPLE_DIR="samples"

echo "Starting interactive batch setup for ${#FILES[@]} files..."
echo "You will need to define crop and traffic light zones for each video."

for id in "${FILES[@]}"; do
    input_file="${SAMPLE_DIR}/${id}.MTS"
    
    if [ ! -f "$input_file" ]; then
        echo "WARNING: File $input_file not found. Skipping."
        continue
    fi

    echo "--------------------------------------------------------"
    echo "Processing $input_file (SETUP MODE)"
    echo "--------------------------------------------------------"

    # Default mode is setup
    python src/apply_on_sample.py "$input_file" --mode setup
    
    echo "Finished $id"
    echo "Press Enter to continue to next video (or Ctrl+C to stop)..."
    read
done

echo "--------------------------------------------------------"
echo "Batch processing complete."
