"""Given a CSV file, prepares data as needed for progress captioning."""
import os
import argparse
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import json


def extract_frames_by_time(video_path, timestamps):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second (fps) of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Convert timestamps to frame indices
    indices = [int(ts * fps) for ts in timestamps]
    
    for i in indices:
        if i < 0 or i > total_frames:
            print(f"Warning! providing invalid time index for {video_path}, indice {i}, total frames {total_frames}")
            continue  # Skip if the index is out of bounds
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def extract_and_save_frames(video_file, selected_frame_idx, frame_save_path):
    frames = extract_frames_by_time(video_file, selected_frame_idx)
    os.makedirs(frame_save_path, exist_ok=True)
    image_files = []
    for t, f in zip(selected_frame_idx, frames):
        f.save(f"{frame_save_path}/frame{t:03d}.png")
        image_files.append(f"{frame_save_path}/frame{t:03d}.png")
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
    args = parser.parse_args()
    return args


def configure_output_dir(df, id_col, output_dir):
    """Configure the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a directory to save frames
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # Add frame_dir to df
    df['frame_dir'] = df[id_col].apply(
        lambda x: os.path.join(frame_dir, f"{x}")
    )
    
    # Configure a single json file to save all the data
    # TODO: This may not be feasible for large datasets
    # but I will handle that later
    json_path = os.path.join(output_dir, "input_data.json")
    
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


def process_row(row, data_list, id_col,action_col, desc_mode):
    """
    Process a single row of the dataframe.
    
    1. Select indices of frames to sample
    2. Extract and save frames
    3. Update the data_list with the processed data
    """
    # 1. Select indices of frames to sample
    if row["frame_sampling"] == "uniform":
        selected_frame_idx = np.linspace(
            0, row["n_frames"], row["n_frames"]
        ).astype(int).tolist()
    elif row["frame_sampling"] == "kmeans":
        raise NotImplementedError(
            "K-means sampling is not implemented yet"
        )
    else:
        raise ValueError(
            f"Invalid frame sampling method: {row['frame_sampling']}"
        )
    
    # 2. Extract and save frames
    image_files = extract_and_save_frames(
        row["video_path"], selected_frame_idx, row["frame_dir"]
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
    return data_list


def load_csv(csv_file, video_dir, id_col, ext='mp4'):
    """Load a CSV file into a pandas DataFrame."""
    print(f"Loading CSV file from {csv_file}")
    df = pd.read_csv(csv_file)
    df["video_path"] = df[id_col].apply(
        lambda x: os.path.join(video_dir, f"{x}.{ext}")
    )
    df = df[df.video_path.apply(os.path.exists)]
    return df


def main(args):
    """Main function."""
    
    # Load CSV
    df = load_csv(args.input_csv, args.video_dir, args.id_col)
    print(df.head())
    
    # Configure output directory
    df, json_path = configure_output_dir(df, args.id_col, args.output_dir)

    # Process each row
    from tqdm import tqdm
    iterator = tqdm(range(len(df)), desc="Processing rows")
    data_list = []
    for i in iterator:
        row = df.iloc[i].to_dict()
        data_list = process_row(
            row, data_list, args.id_col, args.action_col, args.desc_mode,
        )
        import ipdb; ipdb.set_trace()
    
    # Save the data to a json file
    with open(json_path, "w") as f:
        json.dump(data_list, f, indent=4)
    print(f"Saved data to {json_path}")


if __name__ == "__main__":
    args = read_args()
    main(args)