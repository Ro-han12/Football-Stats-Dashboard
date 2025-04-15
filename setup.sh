#!/bin/bash
echo "Setting up Football Stats Dashboard environment..."

# Create necessary directories
mkdir -p input_videos
mkdir -p output_videos
mkdir -p results
mkdir -p stubs
mkdir -p models

# Check if models directory is empty
if [ ! "$(ls -A models)" ]; then
    echo "Note: The models directory is empty. You need to download the YOLOv8 model."
    echo "Download the best.pt model and place it in the models directory."
fi

# Create a results directory if it doesn't exist
if [ ! -d "results" ]; then
    mkdir -p results
    echo "Created results directory"
fi

echo "Setup complete. You can now run the dashboard with:"
echo "./run_dashboard.sh" 