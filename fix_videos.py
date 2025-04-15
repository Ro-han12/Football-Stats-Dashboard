#!/usr/bin/env python3
"""
Fix existing videos for better web compatibility
This script processes all videos in the output_videos directory
"""

import os
import sys
import shutil
import subprocess
import cv2
from pathlib import Path

def fix_mp4_video(input_path, output_dir=None):
    """Fix an MP4 video for better web browser compatibility."""
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    # Create temp filename
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"web_{base_name}")
    
    print(f"Fixing MP4 video: {input_path} -> {output_path}")
    
    try:
        # Try using FFmpeg first
        if shutil.which('ffmpeg'):
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vcodec', 'libx264', '-profile:v', 'baseline', '-level', '3.0',
                '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
                '-movflags', '+faststart',
                '-acodec', 'aac', '-ac', '2', '-b:a', '128k',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Successfully fixed using FFmpeg")
            return output_path
        else:
            # Fall back to OpenCV
            print("FFmpeg not found, using OpenCV")
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error: Could not open {input_path}")
                return None
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create writer with a web-compatible codec
            codecs = ['avc1', 'H264', 'mp4v']
            out = None
            
            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        print(f"Using codec: {codec}")
                        break
                    else:
                        out.release()
                except Exception as e:
                    print(f"Failed with codec {codec}: {e}")
            
            if out is None:
                print("Failed to create video writer")
                return None
            
            # Process frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")
            
            cap.release()
            out.release()
            
            print(f"Successfully fixed using OpenCV - {frame_count} frames processed")
            return output_path
            
    except Exception as e:
        print(f"Error fixing video: {e}")
        return None

def convert_avi_to_mp4(input_path, output_dir=None):
    """Convert AVI to web-compatible MP4."""
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.mp4")
    
    print(f"Converting AVI to MP4: {input_path} -> {output_path}")
    
    # Use the same logic as in fix_mp4_video
    return fix_mp4_video(input_path, output_dir=output_dir)

def main():
    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Find all videos
    avi_files = list(Path(output_dir).glob("*.avi"))
    mp4_files = list(Path(output_dir).glob("*.mp4"))
    
    # Process each file
    if not avi_files and not mp4_files:
        print("No video files found in the output_videos directory.")
        return
    
    print(f"Found {len(avi_files)} AVI files and {len(mp4_files)} MP4 files")
    
    # Fix AVI files - convert to MP4
    for avi_file in avi_files:
        avi_path = str(avi_file)
        mp4_path = os.path.splitext(avi_path)[0] + ".mp4"
        
        # Check if MP4 already exists
        if os.path.exists(mp4_path):
            print(f"MP4 version already exists for {avi_path}")
            continue
        
        print(f"Converting {avi_path} to MP4...")
        result_path = convert_avi_to_mp4(avi_path)
        
        if result_path and os.path.exists(result_path):
            # Rename to standard name
            if result_path != mp4_path:
                shutil.move(result_path, mp4_path)
                print(f"Renamed {result_path} to {mp4_path}")
    
    # Fix existing MP4 files
    for mp4_file in mp4_files:
        mp4_path = str(mp4_file)
        backup_path = mp4_path + ".backup"
        
        # Make backup of original
        if not os.path.exists(backup_path):
            shutil.copy2(mp4_path, backup_path)
            print(f"Created backup: {backup_path}")
        
        # Fix the MP4
        print(f"Fixing {mp4_path} for web compatibility...")
        fixed_path = fix_mp4_video(mp4_path)
        
        if fixed_path and os.path.exists(fixed_path):
            # Replace original with fixed version
            if fixed_path != mp4_path:
                shutil.move(fixed_path, mp4_path)
                print(f"Replaced {mp4_path} with fixed version")
    
    print("Video processing complete!")

if __name__ == "__main__":
    main() 