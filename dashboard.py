import streamlit as st
import json
import os
import subprocess
import pandas as pd
import numpy as np
import time
from pathlib import Path
import tempfile
# Make cv2 import optional
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV (cv2) not available. Some video processing features may be limited.")

# Set page config
st.set_page_config(
    page_title="Football Stats Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Initialize session state for storing the last processed video path
if 'last_processed_video' not in st.session_state:
    st.session_state.last_processed_video = None

def load_latest_results(results_dir="results"):
    """Load the latest results file from the results directory."""
    if not os.path.exists(results_dir):
        return None
        
    results_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) 
                    if f.startswith("tracking_results_") and f.endswith(".json")]
    
    if not results_files:
        return None
        
    latest_file = max(results_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def process_video(video_path):
    """Process the uploaded video file using the main.py script."""
    # Get the uploaded file name instead of using a hardcoded name
    file_name = video_path.name
    
    # Create input directory if it doesn't exist
    os.makedirs("input_videos", exist_ok=True)
    target_path = os.path.join("input_videos", file_name)
    
    # Expected output path
    base_name = os.path.splitext(file_name)[0]
    output_name = f"output_{base_name}.avi" 
    output_path = os.path.join("output_videos", output_name)
    
    # Save the uploaded file with its original name
    with open(target_path, "wb") as f:
        f.write(video_path.read())
    
    # Run the main.py script with the specific input file
    with st.spinner(f"Processing video '{file_name}'... This may take several minutes."):
        try:
            # Pass the input file path as an argument to main.py
            cmd = ["python3", "main.py", "--input", target_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify AVI output file exists
            if os.path.exists(output_path):
                st.success(f"Video processing complete! Download the AVI file to view results.")
                return True, output_path
            else:
                # Check if the default output exists as fallback
                default_output = "output_videos/output_video.avi"
                if os.path.exists(default_output):
                    st.success("Video processing complete! Download the AVI file to view results.")
                    return True, default_output
                else:
                    st.error("Processing completed but no output video was created.")
                    return False, None
        except FileNotFoundError:
            # Try python command instead
            try:
                cmd = ["python", "main.py", "--input", target_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Verify AVI output file exists
                if os.path.exists(output_path):
                    st.success(f"Video processing complete! Download the AVI file to view results.")
                    return True, output_path
                else:
                    # Check if the default output exists as fallback
                    default_output = "output_videos/output_video.avi"
                    if os.path.exists(default_output):
                        st.success("Video processing complete! Download the AVI file to view results.")
                        return True, default_output
                    else:
                        st.error("Processing completed but no output video was created.")
                        return False, None
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                return False, None
        except subprocess.CalledProcessError as e:
            st.error(f"Error processing video: {e.stderr}")
            return False, None

def display_team_stats(results):
    """Display team statistics in a formatted way."""
    team_stats = results["team_stats"]
    
    # Create two columns for team stats
    col1, col2 = st.columns(2)
    
    # Team 1 stats in first column
    with col1:
        st.subheader("Team 1 Statistics")
        st.metric("Possession", f"{team_stats['team1']['possession_percentage']:.1f}%")
        
        # Display metrics in a flat structure
        st.metric("Total Distance", f"{team_stats['team1']['total_distance']:.2f} km")
        st.metric("Avg Speed", f"{team_stats['team1']['avg_speed']:.2f} km/h")
        
        total_shots = sum(team_stats['team1']['shots']['close'].values()) + sum(team_stats['team1']['shots']['long'].values())
        successful_shots = team_stats['team1']['shots']['close']['successful'] + team_stats['team1']['shots']['long']['successful']
        
        st.metric("Shots", total_shots)
        st.metric("Shot Success", f"{successful_shots}/{total_shots}" if total_shots > 0 else "0/0")
    
    # Team 2 stats in second column
    with col2:
        st.subheader("Team 2 Statistics")
        st.metric("Possession", f"{team_stats['team2']['possession_percentage']:.1f}%")
        
        # Display metrics in a flat structure
        st.metric("Total Distance", f"{team_stats['team2']['total_distance']:.2f} km")
        st.metric("Avg Speed", f"{team_stats['team2']['avg_speed']:.2f} km/h")
        
        total_shots = sum(team_stats['team2']['shots']['close'].values()) + sum(team_stats['team2']['shots']['long'].values())
        successful_shots = team_stats['team2']['shots']['close']['successful'] + team_stats['team2']['shots']['long']['successful']
        
        st.metric("Shots", total_shots)
        st.metric("Shot Success", f"{successful_shots}/{total_shots}" if total_shots > 0 else "0/0")

def display_shot_stats(results):
    """Display shot statistics."""
    st.subheader("Shot Analysis")
    
    shots = results["shots"]
    shot_stats = results["shot_stats"]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shots", shot_stats["total_shots"])
    with col2:
        st.metric("Close Shots", shot_stats["close_shots"]["successful"] + shot_stats["close_shots"]["unsuccessful"])
    with col3:
        st.metric("Long Shots", shot_stats["long_shots"]["successful"] + shot_stats["long_shots"]["unsuccessful"])
    with col4:
        successful = shot_stats["close_shots"]["successful"] + shot_stats["long_shots"]["successful"]
        total = shot_stats["total_shots"]
        success_rate = (successful / total * 100) if total > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Show shots table if there are any shots
    if shots:
        shot_df = pd.DataFrame(shots)
        if 'frame' in shot_df.columns:
            shot_df = shot_df.sort_values('frame')
            
        # Format the DataFrame for display
        display_cols = ['frame', 'type', 'successful', 'distance_to_goal', 'speed']
        if 'player_id' in shot_df.columns:
            display_cols.insert(1, 'player_id')
        if 'team' in shot_df.columns:
            display_cols.insert(1, 'team')
            
        st.dataframe(shot_df[display_cols], use_container_width=True)

def display_dribble_stats(results):
    """Display dribble statistics."""
    st.subheader("Dribble Analysis")
    
    dribbles = results["dribbles"]
    dribble_stats = results["dribble_stats"]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Dribbles", dribble_stats["total_dribbles"])
    with col2:
        successful = (dribble_stats["dribble"]["successful"] + 
                     dribble_stats["one_vs_one"]["successful"] + 
                     dribble_stats["one_vs_two"]["successful"] + 
                     dribble_stats["special"]["successful"])
        st.metric("Successful", successful)
    with col3:
        unsuccessful = (dribble_stats["dribble"]["unsuccessful"] + 
                       dribble_stats["one_vs_one"]["unsuccessful"] + 
                       dribble_stats["one_vs_two"]["unsuccessful"] + 
                       dribble_stats["special"]["unsuccessful"])
        st.metric("Unsuccessful", unsuccessful)
    with col4:
        total = dribble_stats["total_dribbles"]
        success_rate = (successful / total * 100) if total > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Show dribbles table if there are any dribbles
    if dribbles:
        dribble_df = pd.DataFrame(dribbles)
        if 'start_frame' in dribble_df.columns:
            dribble_df = dribble_df.sort_values('start_frame')
            
        # Format the DataFrame for display
        display_cols = ['start_frame', 'end_frame', 'type', 'successful', 'duration', 'max_speed']
        if 'player_id' in dribble_df.columns:
            display_cols.insert(1, 'player_id')
        if 'team' in dribble_df.columns:
            display_cols.insert(1, 'team')
            
        st.dataframe(dribble_df[display_cols], use_container_width=True)

def display_pass_stats(results):
    """Display pass statistics."""
    st.subheader("Pass Analysis")
    
    passes = results["passes"]
    pass_stats = results["pass_stats"]
    pass_types = pass_stats["pass_types"]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Passes", pass_stats["total_passes"])
    with col2:
        st.metric("Successful", pass_stats["successful_passes"])
    with col3:
        st.metric("Unsuccessful", pass_stats["unsuccessful_passes"])
    with col4:
        total = pass_stats["total_passes"]
        success_rate = (pass_stats["successful_passes"] / total * 100) if total > 0 else 0
        st.metric("Completion Rate", f"{success_rate:.1f}%")
    
    # Pass types breakdown
    st.subheader("Pass Types")
    
    # Display pass types in a separate section outside of columns
    st.write("Long Ground Passes:", f"{pass_types['long_ground_pass']['successful']}/{pass_types['long_ground_pass']['total']} successful")
    st.write("Long Aerial Passes:", f"{pass_types['long_aerial_pass']['successful']}/{pass_types['long_aerial_pass']['total']} successful")
    st.write("Short Ground Passes:", f"{pass_types['short_ground_pass']['successful']}/{pass_types['short_ground_pass']['total']} successful")
    st.write("Short Aerial Passes:", f"{pass_types['short_aerial_pass']['successful']}/{pass_types['short_aerial_pass']['total']} successful")
    
    # Pass type distribution
    if sum(pass_types[pt]["total"] for pt in pass_types) > 0:
        st.subheader("Pass Type Distribution")
        pass_type_data = {
            "Pass Type": ["Long Ground", "Long Aerial", "Short Ground", "Short Aerial"],
            "Count": [
                pass_types["long_ground_pass"]["total"],
                pass_types["long_aerial_pass"]["total"],
                pass_types["short_ground_pass"]["total"],
                pass_types["short_aerial_pass"]["total"]
            ]
        }
        pass_type_df = pd.DataFrame(pass_type_data)
        st.bar_chart(pass_type_df.set_index("Pass Type"))
    
    # Show passes table if there are any passes
    if passes:
        st.subheader("Pass Details")
        pass_df = pd.DataFrame(passes)
        if 'start_frame' in pass_df.columns:
            pass_df = pass_df.sort_values('start_frame')
            
        # Format the DataFrame for display
        display_cols = ['start_frame', 'end_frame', 'pass_type', 'successful', 'pass_distance', 'initial_ball_speed']
        if 'sender_id' in pass_df.columns:
            display_cols.insert(1, 'sender_id')
        if 'sender_team' in pass_df.columns:
            display_cols.insert(1, 'sender_team')
        if 'receiver_id' in pass_df.columns:
            display_cols.append('receiver_id')
            
        st.dataframe(pass_df[display_cols], use_container_width=True)

def display_player_stats(results):
    """Display player statistics."""
    st.subheader("Player Statistics")
    
    players = results["players"]
    player_data = []
    
    for player_id, data in players.items():
        # Skip players with minimal tracking
        if data["frames_tracked"] < 10:
            continue
            
        player_info = {
            "Player ID": player_id,
            "Team": data["team"],
            "Distance (km)": data["total_distance"],
            "Max Speed (km/h)": data["max_speed"],
            "Avg Speed (km/h)": data["avg_speed"],
            "Ball Possession (frames)": data["ball_possession_frames"],
            "Shots": sum([sum(data["shots"]["close"].values()), sum(data["shots"]["long"].values())]),
            "Successful Dribbles": data["dribbles"]["successful"],
            "Unsuccessful Dribbles": data["dribbles"]["unsuccessful"],
            "Successful Passes": data["passes"]["successful"],
            "Unsuccessful Passes": data["passes"]["unsuccessful"]
        }
        player_data.append(player_info)
    
    if player_data:
        player_df = pd.DataFrame(player_data)
        player_df = player_df.sort_values(["Team", "Distance (km)"], ascending=[True, False])
        st.dataframe(player_df, use_container_width=True)

def convert_avi_to_mp4(input_path):
    """Convert AVI to MP4 for better browser preview compatibility."""
    try:
        import subprocess
        import tempfile
        import os
        
        # Create a temporary MP4 file
        output_path = os.path.join(tempfile.gettempdir(), "temp_preview.mp4")
        
        # Use ffmpeg with optimized settings
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',  # Video codec
            '-preset', 'ultrafast',  # Fastest encoding
            '-crf', '28',       # Lower quality for preview (higher number = lower quality)
            '-c:a', 'aac',      # Audio codec
            '-b:a', '128k',     # Lower audio bitrate for preview
            '-movflags', '+faststart',  # Optimize for web streaming
            '-y',               # Overwrite output file
            output_path
        ]
        
        # Run the conversion
        subprocess.run(cmd, check=True, capture_output=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path
        return None
        
    except Exception as e:
        print(f"Error converting video: {str(e)}")
        return None

def main():
    # Title and description
    st.title("⚽ Football Stats Dashboard")
    st.markdown("Upload a football video to analyze player movements, shots, passes, and dribbles.")
    
    # Sidebar for video upload and processing
    with st.sidebar:
        st.header("Video Upload")
        uploaded_file = st.file_uploader("Upload a football video", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Display video info
            file_details = {
                "Filename": uploaded_file.name, 
                "Size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "Type": uploaded_file.type
            }
            st.write("**File Information:**")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            # Button to process video with the filename in it
            process_button = st.button(f"Process '{uploaded_file.name}'")
            
            if process_button:
                success, output_path = process_video(uploaded_file)
                if success:
                    # Store the output path in session state
                    st.session_state.last_processed_video = output_path
                    
                    # Also store that we've just processed a video
                    st.session_state.just_processed = True
                    st.rerun()
        
        # Video troubleshooting section
        if 'last_processed_video' in st.session_state and st.session_state.last_processed_video:
            st.divider()
            st.header("Video Download")
            
            # Check if we have a processed video
            output_path = st.session_state.last_processed_video
            
            if output_path and os.path.exists(output_path):
                video_name = os.path.basename(output_path)
                
                # Add sidebar preview option
                sidebar_preview = st.checkbox("Quick preview", value=False, key="sidebar_preview")
                
                if sidebar_preview:
                    try:
                        # For AVI files, we might need to convert to MP4 for browser compatibility
                        preview_path = output_path
                        converted = False
                        
                        if output_path.lower().endswith('.avi'):
                            with st.spinner("Preparing video for preview..."):
                                mp4_path = convert_avi_to_mp4(output_path)
                                if mp4_path and os.path.exists(mp4_path):
                                    preview_path = mp4_path
                                    converted = True
                        
                        # Read the video file
                        with open(preview_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                            
                            if converted:
                                st.caption("Preview shows converted video. Download original for full quality.")
                            else:
                                st.caption("Video preview may have reduced quality. Download for full quality.")
                    except Exception as e:
                        st.error(f"Error playing video: {str(e)}")
                        st.info("Some video formats may not play directly in the browser. Download to view.")
                
                st.write(f"Download the processed video file:")
                
                # Offer AVI download option
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label=f"Download {video_name}",
                        data=f,
                        file_name=video_name,
                        mime="video/x-msvideo",
                        use_container_width=True,
                        key="sidebar_download_button"
                    )
                
                # Add file size info
                file_size = os.path.getsize(output_path) / (1024*1024)  # Convert to MB
                st.caption(f"File size: {file_size:.2f} MB")
        
        st.divider()
        st.header("About")
        st.info("""
        This dashboard analyzes football videos to extract player movements, 
        detect events like shots, passes, and dribbles, and provides detailed statistics.
        
        Upload your video and click 'Process Video' to begin analysis.
        """)
    
    # Check if we just processed a video or have results already
    if 'just_processed' in st.session_state and st.session_state.just_processed:
        # Load the latest results because we just processed a video
        results = load_latest_results()
        # Reset the flag
        st.session_state.just_processed = False
    elif 'results' in st.session_state and st.session_state.results is not None:
        # Use cached results if available
        results = st.session_state.results
    else:
        # Default to None, showing the upload-first screen
        results = None
        
    # If no results yet, show a clean slate
    if results is None:
        st.info("👈 Upload a video on the left to get started with analysis.")
        return
    
    # Store the results in session state for future use
    st.session_state.results = results
    
    # Display analysis timestamp and source video if available
    timestamp_msg = f"Analysis timestamp: {results['timestamp']}"
    if 'source_video' in results and results['source_video']:
        timestamp_msg += f" | Source: {results['source_video']}"
    st.caption(timestamp_msg)
    
    # Display video and team stats
    video_col, stats_col = st.columns([2, 3])
    
    with video_col:
        # Video header
        st.subheader("Processed Video")
        
        # Use the same output path stored in session state
        output_path = st.session_state.last_processed_video
        
        # Show video and download option
        if output_path and os.path.exists(output_path):
            video_name = os.path.basename(output_path)
            
            # Add preview option with a toggle
            show_preview = st.checkbox("Show video preview", value=False)
            
            if show_preview:
                try:
                    # For AVI files, we might need to convert to MP4 for browser compatibility
                    preview_path = output_path
                    converted = False
                    
                    if output_path.lower().endswith('.avi'):
                        with st.spinner("Preparing video for preview..."):
                            mp4_path = convert_avi_to_mp4(output_path)
                            if mp4_path and os.path.exists(mp4_path):
                                preview_path = mp4_path
                                converted = True
                    
                    # Read the video file
                    with open(preview_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                        
                        if converted:
                            st.caption("Preview shows converted video. Download original for full quality.")
                        else:
                            st.caption("Video preview may have reduced quality. Download for full quality.")
                except Exception as e:
                    st.error(f"Error playing video: {str(e)}")
                    st.info("Some video formats may not play directly in the browser. Download to view.")
            
            # Download section
            st.info(f"Download the processed video for full quality viewing:")
            with open(output_path, 'rb') as f:
                st.download_button(
                    label=f"Download {video_name}",
                    data=f,
                    file_name=video_name,
                    mime="video/x-msvideo",
                    use_container_width=True,
                    key="main_download_button"
                )
            
            # Add file size info
            file_size = os.path.getsize(output_path) / (1024*1024)  # Convert to MB
            st.caption(f"File size: {file_size:.2f} MB")
        else:
            st.warning("Video file not available.")
    
    with stats_col:
        # Team stats
        display_team_stats(results)
    
    # Tabs for different stats
    tab1, tab2, tab3, tab4 = st.tabs(["Shots", "Dribbles", "Passes", "Players"])
    
    with tab1:
        display_shot_stats(results)
    
    with tab2:
        display_dribble_stats(results)
    
    with tab3:
        display_pass_stats(results)
    
    with tab4:
        display_player_stats(results)

if __name__ == "__main__":
    main() 