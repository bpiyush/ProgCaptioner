"""Given a CSV file, prepares data as needed for progress captioning."""
import os
import argparse

import pandas as pd


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
    df = load_csv(args.input_csv, args.video_dir, args.id_col)
    print(df.head())
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    args = read_args()
    main(args)