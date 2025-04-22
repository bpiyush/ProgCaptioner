import os
import cv2
import json
import argparse
from PIL import Image


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


def prepare_data_json(video_file, output_file, action_name, desc_mode):
    # recommend 2-6 frames, could be uniformly sample / k-means clustering, etc. 
    selected_frame_idx = [2, 26, 40, 41, 54] 
    frames = extract_frames_by_time(video_file, selected_frame_idx)
    frame_save_path = video_file.replace('.mp4', '')
    os.makedirs(frame_save_path, exist_ok=True)
    image_files = []
    for t, f in zip(selected_frame_idx, frames):
        f.save(f"{frame_save_path}/frame{t:03d}.png")
        image_files.append(f"{frame_save_path}/frame{t:03d}.png")
    print(f"Extracted {len(image_files)} frames from {video_file} and saved to {frame_save_path}")
    
    save_list = [{
        "idx": 0,
        "n_frames": len(image_files),
        "image_files": image_files,
        "query0": prepare_query(len(image_files), action_name, desc_mode),
    }]
    with open(output_file, 'w') as f:
        json.dump(save_list, f, indent=4)
    print(f"Saving to {output_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for video captioning')
    parser.add_argument('--video_file', type=str, default="data/example_video.mp4",
                        help='Path to the input video file')
    parser.add_argument('--output_file', type=str, default="data/data_files/input/one_example.json",
                        help='Path to save the output JSON file')
    parser.add_argument('--desc_mode', type=str, default='c',
                        choices=['a', 'b', 'c'],
                        help='Description mode: a=object state, b=body pose, c=action')
    parser.add_argument('--action_name', type=str, default=None,
                        help='Name of the action being performed in the video')
    
    args = parser.parse_args()
    prepare_data_json(args.video_file, args.output_file, args.action_name, args.desc_mode)