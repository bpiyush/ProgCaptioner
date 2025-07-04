"""
Utilities for input-output loading/saving.
"""

from typing import Any, List
import yaml
import pickle
import json
import pandas as pd


class PrettySafeLoader(yaml.SafeLoader):
    """Custom loader for reading YAML files"""
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple
)


def load_yml(path: str, loader_type: str = 'default'):
    """Read params from a yml file.

    Args:
        path (str): path to the .yml file
        loader_type (str, optional): type of loader used to load yml files. Defaults to 'default'.

    Returns:
        Any: object (typically dict) loaded from .yml file
    """
    assert loader_type in ['default', 'safe']

    loader = yaml.Loader if (loader_type == "default") else PrettySafeLoader

    with open(path, 'r') as f:
        data = yaml.load(f, Loader=loader)

    return data


def save_yml(data: dict, path: str):
    """Save params in the given yml file path.

    Args:
        data (dict): data object to save
        path (str): path to .yml file to be saved
    """
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_pkl(path: str, encoding: str = "ascii"):
    """Loads a .pkl file.

    Args:
        path (str): path to the .pkl file
        encoding (str, optional): encoding to use for loading. Defaults to "ascii".

    Returns:
        Any: unpickled object
    """
    return pickle.load(open(path, "rb"), encoding=encoding)


def save_pkl(data: Any, path: str) -> None:
    """Saves given object into .pkl file

    Args:
        data (Any): object to be saved
        path (str): path to the location to be saved at
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_json(path: str) -> dict:
    """Helper to load json file"""
    with open(path, 'rb') as f:
        data = json.load(f)
    return data


def save_json(data: dict, path: str):
    """Helper to save `dict` as .json file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_txt(path: str):
    """Loads lines of a .txt file.

    Args:
        path (str): path to the .txt file

    Returns:
        List: lines of .txt file
    """
    with open(path) as f:
        lines = f.read().splitlines()
    return lines


def save_txt(data: dict, path: str):
    """Writes data (lines) to a txt file.

    Args:
        data (dict): List of strings
        path (str): path to .txt file
    """
    assert isinstance(data, list)

    lines = "\n".join(data)
    with open(path, "w") as f:
        f.write(str(lines))


def read_spreadsheet(sheet_id, gid, url=None, drop_na=True, **kwargs):
    if url is None:
        BASE_URL = 'https://docs.google.com/spreadsheets/d/'
        url = BASE_URL + sheet_id + f'/export?gid={gid}&format=csv'
    df = pd.read_csv(url, **kwargs)
    
    if drop_na:
        # drop all rows which have atleast 1 NaN value
        df = df.dropna(axis=0)

    return df


def load_midi(file, rate=16000):
    import pretty_midi
    assert file.endswith('.mid')
    pm = pretty_midi.PrettyMIDI(file)
    y = pm.synthesize(fs=rate)
    return y, rate


def load_ptz(path):
    import gzip
    import torch
    with gzip.open(path, 'rb') as f:
        data = torch.load(f)
    return data


def save_video(frames, path, fps=30):
    import imageio
    imageio.mimwrite(path, frames, fps=fps)


def read_spreadsheet(sheet_id, gid, gid_key="granularity", **kwargs):
    BASE_URL = 'https://docs.google.com/spreadsheets/d/'
    df = df = pd.read_csv(BASE_URL + sheet_id + f'/export?gid={gid}&format=csv', **kwargs)
    return df


def load_jsonl(file_path: str) -> list:
    """Load data from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        list: List of dictionaries, where each dictionary is a JSON object from the file
        
    Example:
        >>> data = load_jsonl("path/to/file.jsonl")
        >>> print(data[0])  # Print first JSON object
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, file_path: str) -> None:
    """Save data to a JSONL file.
    
    Args:
        data (list): List of dictionaries to save
        file_path (str): Path where to save the JSONL file
        
    Example:
        >>> data = [{"text": "hello"}, {"text": "world"}]
        >>> save_jsonl(data, "output.jsonl")
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
