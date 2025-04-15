# Football-stats

A computer vision-based football player tracking system that analyzes football videos to track and annotate player movements.

## Features

- Video processing and player tracking
- Object detection using YOLOv8 model
- Track visualization and annotation
- Support for video input/output in various formats
- Interactive dashboard for visualizing results
- Advanced metrics including shots, passes, and dribbles analysis

## Prerequisites

- requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ro-han12/Football-stats.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Processing

1. Place your input video in the `input_videos` directory
2. Run the main script:
```bash
python main.py
```
3. The processed video will be saved in the `output_videos` directory
4. Analysis results will be saved in the `results` directory as JSON files

### Interactive Dashboard

1. Run the dashboard:
```bash
./run_dashboard.sh
```
Or
```bash
streamlit run dashboard.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
3. Upload a video file through the dashboard interface
4. Click "Process Video" to analyze the video
5. View the results in the interactive dashboard

## Project Structure

- `main.py`: Main script for video processing
- `dashboard.py`: Streamlit dashboard for visualization
- `utils.py`: Utility functions for video handling
- `trackers.py`: Object tracking implementation
- `models/`: Directory containing YOLOv8 model files
- `input_videos/`: Directory for input videos
- `output_videos/`: Directory for processed videos
- `results/`: Directory for analysis results in JSON format
- `stubs/`: Directory for tracking stubs

## Dashboard Features

The interactive dashboard provides:

- Team statistics comparison
- Shot analysis with success rates
- Dribble detection and classification
- Pass analysis with different pass types
- Player-by-player performance metrics
- Video playback of the processed footage

## Video Compatibility

The system outputs video in both AVI and MP4 formats:
- MP4 is used for web display in the dashboard
- AVI provides higher quality for offline viewing

If you encounter issues with video playback in the dashboard:
1. Make sure both AVI and MP4 files exist in the output_videos directory
2. You can use the provided convert_video.py script to convert AVI to MP4:
```bash
# Convert a specific video file
python convert_video.py output_videos/output_video.avi

# Specify output path
python convert_video.py output_videos/output_video.avi -o output_videos/output_video.mp4

# Use OpenCV instead of ffmpeg for conversion
python convert_video.py output_videos/output_video.avi -m opencv
```

## Directory Setup

Before running, make sure these directories exist:
- create a 'input_videos' folder in root directory for source videos
- create a 'output_videos' folder in root directory for processed videos
- create a 'results' folder in root directory for analysis data
- create a 'stubs' folder in the root directory for tracking data storage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

