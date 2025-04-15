#!/usr/bin/env python3
"""
Video Converter Utility for Football Stats Dashboard
Converts AVI videos to MP4 format for better web compatibility
"""

import os
import argparse
import cv2
import subprocess

def convert_using_opencv(input_path, output_path):
    """Convert video from AVI to MP4 using OpenCV."""
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open input video {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer with a web-compatible codec
    # Try different codecs in order of web compatibility
    codecs_to_try = ['avc1', 'H264', 'mp4v']
    out = None
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if test_out.isOpened():
                out = test_out
                print(f"Using codec: {codec}")
                break
            else:
                test_out.release()
        except Exception as e:
            print(f"Failed to use codec {codec}: {e}")
    
    if out is None:
        print("Failed to create video writer with any codec")
        return False
    
    print(f"Converting {total_frames} frames...")
    frame_count = 0
    
    # Process all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        # Show progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Conversion completed: {input_path} → {output_path}")
    return True

def convert_using_ffmpeg(input_path, output_path):
    """Convert video from AVI to MP4 using FFmpeg."""
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        
        print(f"Converting using FFmpeg: {input_path} → {output_path}")
        
        # Run the conversion with browser-optimized parameters
        result = subprocess.run([
            'ffmpeg', '-i', input_path, 
            '-vcodec', 'libx264', '-profile:v', 'baseline', '-level', '3.0',
            '-pix_fmt', 'yuv420p',  # Required for browser compatibility
            '-preset', 'fast', '-crf', '23',  # Good balance between quality and file size
            '-movflags', '+faststart',  # Optimize for web streaming
            '-acodec', 'aac', '-ac', '2', '-b:a', '128k',
            output_path
        ], check=True, capture_output=True)
        
        print("Conversion completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert video from AVI to MP4 format')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('--output', '-o', help='Output video file path (default: input name with .mp4 extension)')
    parser.add_argument('--method', '-m', choices=['opencv', 'ffmpeg'], default='ffmpeg',
                        help='Conversion method to use (default: ffmpeg)')
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if not args.output:
        input_base = os.path.splitext(args.input)[0]
        args.output = f"{input_base}.mp4"
    
    print(f"Converting {args.input} to {args.output} using {args.method}...")
    
    if args.method == 'ffmpeg':
        success = convert_using_ffmpeg(args.input, args.output)
        if not success:
            print("FFmpeg conversion failed, falling back to OpenCV...")
            success = convert_using_opencv(args.input, args.output)
    else:
        success = convert_using_opencv(args.input, args.output)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main() 