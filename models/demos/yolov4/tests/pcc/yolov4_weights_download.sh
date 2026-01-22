#!/bin/bash

# Check if gdown is installed as a Python module
if ! python3 -m gdown --version &> /dev/null; then
    echo "gdown is not installed. Installing..."
    uv pip install gdown
fi

# Google Drive file ID
FILE_ID="1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ"
# Output filename
OUTPUT="models/demos/yolov4/tests/pcc/yolov4.pth"

# Download the file
python3 -m gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT}"
