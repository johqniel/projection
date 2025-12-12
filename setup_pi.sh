#!/bin/bash

# Setup script for Raspberry Pi / Linux

echo "--- Setting up Environment ---"

# 1. Update and install dependencies
echo "1. Installing System Dependencies..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libatlas-base-dev

# 2. Create Virtual Environment
echo "2. Creating Virtual Environment (vm)..."
if [ ! -d "vm" ]; then
    python3 -m venv vm
    echo "   Virtual environment created."
else
    echo "   Virtual environment already exists."
fi

# 3. specific fixes for raspberry pi
# If running on Pi, we might need tflite_runtime instead of full tensorflow if possible, 
# but requirements.txt lists tensorflow. 
# We will use the requirements.txt as requested.

# 4. Install Requirements
echo "3. Installing Python Requirements..."
source vm/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found."
    exit 1
fi

echo "--- Setup Complete ---"
echo "To run the program, use:"
echo "source vm/bin/activate"
echo "python src/main.py"
