"""Given a CSV file, prepares data as needed for progress captioning."""
import os
import argparse
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import json
from colorama import init, Fore, Back, Style
import decord

# Initialize colorama for cross-platform colored output
init(autoreset=True)


def print_success(message):
    """Print a success message in green."""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_info(message):
    """Print an info message in blue."""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")


def print_warning(message):
    """Print a warning message in yellow."""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")


def print_error(message):
    """Print an error message in red."""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def print_header(message):
    """Print a header message with background."""
    print(f"{Back.BLUE}{Fore.WHITE} {message} {Style.RESET_ALL}")


def print_progress(message):
    """Print a progress message in cyan."""
    print(f"{Fore.CYAN}→ {message}{Style.RESET_ALL}")


def extract_frames_by_time(video_path, timestamps):
    # print_progress(f"Extracting frames from video: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second (fps) of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # print_info(f"Video FPS: {fps:.2f}, Total frames: {total_frames}")
    
    # Convert timestamps to frame indices
    indices = [int(ts * fps) for ts in timestamps]
    
    for i in indices:
        if i < 0 or i > total_frames:
            print_warning(f"Invalid time index for {video_path}, index {i}, total frames {total_frames}")
            continue  # Skip if the index is out of bounds
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    # print_success(f"Successfully extracted {len(frames)} frames")
    return frames

def get_duration(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    return duration


def get_duration_decord(video_file):
    vr = decord.VideoReader(video_file)
    duration = len(vr) / vr.get_avg_fps()
    return duration


def extract_and_save_frames(video_file, selected_frame_idx, total_frames,frame_save_path):
    # print_progress(f"Processing video: {os.path.basename(video_file)}")
    duration = get_duration_decord(video_file)
    selected_time_idx = [duration * (f / total_frames) for f in selected_frame_idx]
    frames = extract_frames_by_time(video_file, selected_time_idx)
    os.makedirs(frame_save_path, exist_ok=True)
    image_files = []
    for t, f in zip(selected_frame_idx, frames):
        f.save(f"{frame_save_path}/frame{t:03d}.png")
        image_files.append(f"{frame_save_path}/frame{t:03d}.png")
    # print_success(f"Saved {len(image_files)} frames to {frame_save_path}")
    return image_files


def read_args():
    """Read arguments from command line."""
    parser = argparse.ArgumentParser(
        description='Prepare data for video captioning',
    )
    # Input CSV file
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to the input CSV file',
    )
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--action_col", type=str, default="class")
    parser.add_argument("--desc_mode", type=str, default="c")
    parser.add_argument("--frame_sampling", type=str, default="uniform")
    parser.add_argument("--n_frames", type=int, default=6)
    # Video directory
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Path to the video directory',
    )
    # Output directory
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output directory',
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--suffix", type=str, default="")
    # Add ext with mp4 as the default
    parser.add_argument("--ext", type=str, default="mp4")
    args = parser.parse_args()
    return args


def configure_output_dir(df, id_col, output_dir, debug, suffix=""):
    """Configure the output directory."""
    print_progress("Setting up output directory structure...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a directory to save frames
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    print_success(f"Created frame directory: {frame_dir}")

    # Add frame_dir to df
    df['frame_dir'] = df[id_col].apply(
        lambda x: os.path.join(frame_dir, f"{x}")
    )
    
    # Configure a single json file to save all the data
    # TODO: This may not be feasible for large datasets
    # but I will handle that later
    suffix = f"_{suffix}" if suffix else ""
    json_path = os.path.join(output_dir, f"input_data{suffix}.json")
    print_info(f"Output JSON will be saved to: {json_path}")
    
    # Add debug flag to df
    if debug:
        print_info("Debug mode is enabled, will only process the first 10 rows")
        df = df.head(10)
    
    return df, json_path


desc_dict = {
        'a': "object's state during the action",
        'b': "subject's body pose during the action",
        'c': "action"
    }


def prepare_query(n_frames, action_name=None, desc_mode='a', requirement=True):
    desc = desc_dict[desc_mode]
    action = '' if action_name is None else f' depicting {action_name}'
    query1 = f"These are {n_frames} frames extracted from a video sequence{action}, provide a description for each frame.\n" 
    query2 = f"Requirement: (1) Ensure each frame's description is specific to the corresponding frame, not referencing to other frames; (2) The description should focus on the specific action being performed, capturing the {desc} and progression of the action. There is no need to comment on other elements, such as the background or unrelated objects.\n" if requirement else ""
    query3 = "Reply with the following format:\n<Frame 1>: Your description\n"
    query4 = '...\n' if n_frames > 2 else '' 
    query5 = f"<Frame {n_frames}>: Your description\n"
    query = query1 + query2 + query3 + query4 + query5
    return query


def process_row(row, data_list, id_col,action_col, desc_mode, frame_sampling, n_frames):
    """
    Process a single row of the dataframe.
    
    1. Select indices of frames to sample
    2. Extract and save frames
    3. Update the data_list with the processed data
    """
    # print_progress(f"Processing video ID: {row[id_col]}")
    
    vr = decord.VideoReader(row["video_path"])
    total_frames = len(vr)
    duration = total_frames / vr.get_avg_fps()
    
    # 1. Select timestamps of frames to sample
    if frame_sampling == "uniform":
        # Ok to convert to int since we do not care about FPS < 1
        selected_frame_idx = np.linspace(
            0, total_frames, n_frames, endpoint=False, dtype=int,
        ).tolist()
        # print_info(f"Using uniform sampling: {len(selected_frame_idx)} frames")
    elif frame_sampling == "kmeans":
        raise NotImplementedError(
            "K-means sampling is not implemented yet"
        )
    else:
        raise ValueError(
            f"Invalid frame sampling method: {frame_sampling}"
        )
    
    # 2. Extract and save frames
    image_files = extract_and_save_frames(
        row["video_path"], selected_frame_idx, total_frames, row["frame_dir"]
    )
    
    # 3. Update the data_list with the processed data
    data_list.append({
        "idx": row[id_col],
        "n_frames": len(image_files),
        "image_files": image_files,
        "query0": prepare_query(
            len(image_files), row[action_col], desc_mode
        ),
    })
    # print_success(f"Completed processing video ID: {row[id_col]}")
    return data_list


def load_csv(csv_file, video_dir, id_col, ext):
    """Load a CSV file into a pandas DataFrame."""
    print_header("Loading and validating data")
    print_progress(f"Loading CSV file from {csv_file}")
    df = pd.read_csv(csv_file)
    print_info(f"Loaded {len(df)} rows from CSV")
    
    print_progress("Validating video file paths...")
    df["video_path"] = df[id_col].apply(
        lambda x: os.path.join(video_dir, f"{x}.{ext}")
    )
    initial_count = len(df)
    df = df[df.video_path.apply(os.path.exists)]
    final_count = len(df)
    
    if final_count < initial_count:
        print_warning(f"Found {initial_count - final_count} missing video files")
    
    print_success(f"Validated {len(df)} video files exist")
    return df


def main(args):
    """Main function."""
    print_header("Video Data Preparation Pipeline")
    print_info(f"Input CSV: {args.input_csv}")
    print_info(f"Video directory: {args.video_dir}")
    print_info(f"Output directory: {args.output_dir}")
    print_info(f"Frame sampling: {args.frame_sampling}")
    print_info(f"Number of frames: {args.n_frames}")
    print_info(f"Description mode: {args.desc_mode}")
    print()
    
    # Load CSV
    df = load_csv(args.input_csv, args.video_dir, args.id_col, args.ext)
    print_info(f"DataFrame shape: {df.shape}")
    print()
    
    # Configure output directory
    df, json_path = configure_output_dir(
        df, args.id_col, args.output_dir, args.debug, args.suffix,
    )
    print()

    # Process each row
    print_header("Processing Videos")
    from tqdm import tqdm
    iterator = tqdm(range(len(df)), desc="Processing rows")
    data_list = []
    for i in iterator:
        row = df.iloc[i].to_dict()
        data_list = process_row(
            row,
            data_list,
            args.id_col,
            args.action_col,
            args.desc_mode,
            args.frame_sampling,
            args.n_frames,
        )
    
    # Save the data to a json file
    print_header("Saving Results")
    print_progress(f"Saving processed data to {json_path}")
    with open(json_path, "w") as f:
        json.dump(data_list, f, indent=4)
    print_success(f"Successfully saved {len(data_list)} processed videos to {json_path}")
    print()
    print_header("Pipeline Complete!")
    print_success(f"Total videos processed: {len(data_list)}")
    print_success(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    args = read_args()
    main(args)