#!/bin/bash

# NOTE: Run this script from the project root directory!
# ./src/process_batch.sh

# List of file IDs to process
# Extension is .MTS (case-sensitive on some systems)
FILES=(
  "00003"
  "00004"
  "00008"
  "00009"
  "00010"
  "00013"
  "00015"
  "00016"
  "00029"
  "00032"
)

# Directory containing samples relative to project root
SAMPLE_DIR="samples"

echo "Starting batch processing for ${#FILES[@]} files..."

for id in "${FILES[@]}"; do
    input_file="${SAMPLE_DIR}/${id}.MTS"
    
    if [ ! -f "$input_file" ]; then
        echo "WARNING: File $input_file not found. Skipping."
        continue
    fi

    echo "--------------------------------------------------------"
    echo "Processing $input_file"
    echo "--------------------------------------------------------"

    # 1. People Only
    echo "Running Mode: PEOPLE ONLY"
    python src/apply_on_sample.py "$input_file" --mode people_only
    
    # 2. Random Red
    echo "Running Mode: RANDOM RED"
    python src/apply_on_sample.py "$input_file" --mode random_red
    
    echo "Finished $id"
done

echo "--------------------------------------------------------"
echo "Batch processing complete."
