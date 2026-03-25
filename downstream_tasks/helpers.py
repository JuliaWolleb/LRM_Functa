import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import functional as FV
from torchvision.transforms import InterpolationMode
from pathlib import Path
import pandas as pd
import numpy as np

import argparse
import enum

import ml_collections
from termcolor import colored
from collections import Counter, defaultdict
from typing import Any, Dict, Optional
import random
import yaml
import scipy.stats
import ml_collections
import numpy as np
import torch
from ml_collections import ConfigDict
from rich.console import Console
from rich.table import Table
from termcolor import colored
import matplotlib

def flatten_config_dict(config: ml_collections.ConfigDict):
    fields = []
    for k in config:
        v = config[k]
        if isinstance(v, ml_collections.ConfigDict):
            for sub_k, sub_v in flatten_config_dict(v):
                fields.append((f"{k}.{sub_k}", sub_v))
        else:
            fields.append((k, v))
    return fields


def parse_cli_overides(config: ml_collections.ConfigDict):
    """
    Parse args from CLI and override config dictionary entries
    """
    parser = argparse.ArgumentParser()
    flatten_config = flatten_config_dict(config)
    for key, value in flatten_config:
        parser.add_argument(f"--{key}")
    args = vars(parser.parse_args())

    def print_config_override(key, old_value, new_value, first_config_overide):
        if first_config_overide:
            print(colored("Config overrides:", "red"))
        print(f"     {key:25s} -> {new_value} (instead of {old_value})")

    def cast_argument(old_value, new_value):
        try:
            if new_value is None:
                return None
            if type(old_value) is int:
                return int(new_value)
            if type(old_value) is float:
                return float(new_value)
            if type(old_value) is str:
                return new_value
            if type(old_value) is bool:
                return new_value.lower() in ("yes", "true", "t", "1")
            if type(old_value) in [tuple, list]:
                sequence_constructor = type(old_value)
                old_element = old_value[0]
                return sequence_constructor(
                    cast_argument(old_element, e) for e in new_value.split(",")
                )
            if issubclass(old_value.__class__, enum.Enum):
                return old_value.__class__(new_value)
            if old_value is None:
                return new_value  # assume string
            raise ValueError()
        except Exception:
            raise ValueError(f"Unable to parse config key '{key}' with value '{new_value}'")

    first_config_overide = True
    for key, original_value in flatten_config:
        override_value = cast_argument(original_value, args[key])
        if override_value is not None and override_value != original_value:
            c = config
            for k in key.split(".")[:-1]:
                c = c[k]
            c[key.split(".")[-1]] = override_value
            # setattr(config, key, override_value)
            print_config_override(key, original_value, override_value, first_config_overide)
            first_config_overide = False

    return config

class VideoDataset(Dataset):
    def __init__(self, videos_directory, labels_file, clip=120):
        self.videos_directory = Path(videos_directory)
        self.clip = clip  #length of the clip we load

        if not self.videos_directory.exists():
            raise FileNotFoundError(f"{videos_directory} not found")

        # Load labels
        df = pd.read_csv(labels_file)

        # Map labels
        label_dict = {"Negative": 0, "Positive": 1}  #binary labels for presence/absence of B-lines

        self.samples = []

        for _, row in df.iterrows():
            video_id = str(row["ID"])
            label = label_dict[row["Label"]]

            video_path = self.videos_directory / f"{video_id}.mp4"

            if video_path.exists():
                self.samples.append((video_path, label))
            else:
                print(f"[WARN] Missing: {video_path}")

        print(f"Loaded {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load video
        frames, _, _ = read_video(str(path), pts_unit="sec")

        # ---- Clip sampling ----
        T = frames.shape[0]

        if T >= self.clip:
            start = np.random.randint(0, T - self.clip + 1)
            frames = frames[start:start + self.clip]
        else:
            repeat = (self.clip + T - 1) // T
            frames = frames.repeat((repeat, 1, 1, 1))[:self.clip]

        # ---- Resize + normalize ----
        frames = FV.resize(frames, (112, 112), interpolation=InterpolationMode.BILINEAR)
        frames = frames.float() / 255.0

        return {
            "images": frames,   # (T, H, W, C)
            "label": label
        }


def get_dataset(config):
    return VideoDataset(
        videos_directory=config.videos_directory,
        labels_file=config.labels_file,
        clip=config.clip
    )


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_config(config):
    print("======== CONFIG ========")
    print(yaml.dump(config.to_dict()), end="")
    print("========================")


def show_splits_info(train_indices, test_indices, valid_indices, labels_dict, label_names):
    console = Console()

    table = Table(show_header=True)
    table.add_column("split")
    table.add_column("size", justify="right")
    for label in label_names:
        table.add_column(label, justify="right")
    train_labels = np.array([labels_dict[i] for i in train_indices])
    valid_labels = np.array([labels_dict[i] for i in valid_indices])
    test_labels = np.array([labels_dict[i] for i in test_indices])

    table.add_row("train", str(len(train_labels)),
                  f"{len(train_labels) - train_labels.sum()} ({int(np.round((len(train_labels) - train_labels.sum()) / len(train_labels) * 100, 0))}%)",
                  f"{train_labels.sum()} ({int(np.round((train_labels.sum()) / len(train_labels) * 100, 0))}%)"
                  )
    table.add_row("valid", str(len(valid_labels)),
                  f"{len(valid_labels) - valid_labels.sum()} ({int(np.round((len(valid_labels) - valid_labels.sum()) / len(valid_labels) * 100, 0))}%)",
                  f"{valid_labels.sum()} ({int(np.round((valid_labels.sum()) / len(valid_labels) * 100, 0))}%)"
                  )
    table.add_row("test", str(len(test_labels)),
                  f"{len(test_labels) - test_labels.sum()} ({int(np.round((len(test_labels) - test_labels.sum()) / len(test_labels) * 100, 0))}%)",
                  f"{test_labels.sum()} ({int(np.round((test_labels.sum()) / len(test_labels) * 100, 0))}%)"
                  )

    print("Split infos:")
    console.print(table)

def log_metrics(title: str, metrics: dict, color=None) -> None:
    try:
        print(colored(f"{title}:", color))
        for key, value in metrics.items():
            if isinstance(value, (int, float)):  # Check if the value is a number
                print(colored(f"{key}: {value:.3f}", color))
            elif isinstance(value, list):
                formatted_values = ", ".join(f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in value)
                print(colored(f"{key}: [{formatted_values}]", color))
            else:
                print(colored(f"{key}: {value}", color))
    except:
        print(colored(f"{title}:", color))
        print(metrics)

def prefix_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def split_array_most_equaly(array, num_splits: int):
    """Split array in k arrays of similar sizes."""
    n = len(array)
    split_sizes = np.ones(num_splits, dtype=int) * (n // num_splits)
    split_sizes[: n % num_splits] += 1

    offset = 0
    splits = []
    for size in split_sizes:
        splits.append(array[offset: offset + size])
        offset += size

    return splits


def split_k_folds(indices, labels, k: int, random_state: int = 0):
    """Stratified K-fold of the indices array."""
    # split indices per label
    indices_by_label = defaultdict(lambda: [])
    for index, label in zip(indices, labels):
        indices_by_label[label].append(index)

    # shuffle each with a fixed random key
    np.random.seed(random_state)
    separate_indices = []
    for _, indices in indices_by_label.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        separate_indices.append(indices)

    # split each in k folds
    folds = [[] for _ in range(k)]
    for i, indices in enumerate(separate_indices):
        # Smallest fold first for a greedy strategy to balance the split sizes.
        folds = sorted(folds, key=lambda indices: sum(map(len, indices)))
        current_label_folds = split_array_most_equaly(indices, k)
        for j in range(k):
            folds[j].append(current_label_folds[j])

    folds = [np.concatenate(indices) for indices in folds]

    # Reshuffle
    for f in folds:
        np.random.shuffle(f)

    return folds


def override_config_dict(config: ConfigDict, overrides: Dict[str, Any]):
    for k, v in overrides.items():
        try:
            if "." in k:
                first = k.split(".")[0]
                rest = ".".join(k.split(".")[1:])
                override_config_dict(config[first], {rest: v})
            else:
                config.get_ref(k).set(v)
        except KeyError:
            raise KeyError(f"Cannot override configuration field '{k}'")


def get_label_names(labels_file):
    if "diagnosis" in labels_file:
        return ["negative", "positive"]

    elif "severity" in labels_file or "prognosis" in labels_file:
        # mild = hospital,
        # severe = hospital with O2 or intubated
        return ["mild", "severe"]
    
    elif "tb_rif_genexpert" in labels_file:
        return ["negative", "positive"]

    return None


def exclusive_cumsum(t, dim=-1):
    shape = list(t.shape)
    shape[dim] = 1
    zeros = torch.zeros(shape, dtype=t.dtype, device=t.device)
    return torch.cat(
        [zeros, torch.cumsum(t, dim=dim).narrow(dim=dim, start=0, length=t.shape[dim] - 1)], dim=dim
    )


def pad_dim_with_zeros(t, dim, length):
    if t.shape[dim] == length:
        return t
    t_padded_shape = list(t.shape)
    t_padded_shape[dim] = length
    t_padded = torch.zeros(t_padded_shape, device=t.device, dtype=t.dtype)
    t_padded.narrow(dim=dim, start=0, length=t.shape[dim]).copy_(t)
    return t_padded


def try_parse_exact_bool(b):
    if isinstance(b, str):
        if b.lower() == "true":
            return True
        if b.lower() == "false":
            return False
    return b


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def nice_plot_settings(font_size=18, font_family='STIXGeneral', mathtext_fontset='stix', usetex=True):
    matplotlib.rcParams['mathtext.fontset'] = mathtext_fontset
    matplotlib.rcParams['font.family'] = font_family
    matplotlib.rcParams['text.usetex'] = usetex
    matplotlib.rcParams['font.size'] = font_size


def print_summary_results(metric_folds, title):
    console = Console()
    table = Table(title=title, show_header=True)
    table.add_column("Criterion")
    table.add_column("mean +/- std", justify="right")
    for metric in metric_folds[0].keys():
        if metric in ['false_positive_rate', 'true_positive_rate']:
            continue
        values = [i[metric] for i in metric_folds]
        avg, std = np.mean(values), np.std(values)
        table.add_row(metric, f"{np.round(avg, 2)} +/- {np.round(std, 2)}")
    console.print(table)

