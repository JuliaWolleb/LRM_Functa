#!/usr/bin/env python3
"""
Compute MAE for ED/ES across ALL subfolders inside ./reconstructions_tracking
Each subfolder name must follow: <prefix>_<middledim>_<vidm>
Example: ortho_512_2048, separate_256_1024

Outputs:
- results_ed_es.csv  (summaries from all folders)
- mae_vs_middledim.png (ED MAE vs middle dim, lines = vidm)
"""

import os
import glob
import csv
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional
from scipy.stats import linregress
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import torch
from torch.utils.data import Dataset, DataLoader

# ------------------- USER-PATHS -------------------
TRACING_FILE = "./data/EchoNet-Dynamic/VolumeTracings.csv"
FILELIST_CSV = "./data/EchoNet-Dynamic/FileList.csv"

gc.collect()
gc.set_threshold(0)

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def _ensure_odd_window(w: int, length: int) -> int:
    if w >= length:
        w = length - 1 if (length - 1) % 2 == 1 else length - 2
    if w % 2 == 0:
        w += 1
    if w < 3:
        w = 3
    return int(w)

def _highpass_filter(signal: np.ndarray, fs: float, cutoff: float = 0.5, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high", analog=False)
    return filtfilt(b, a, signal)

def event_mae_closest(pred_indices: List[int], true_index: int) -> float:
    if pred_indices is None or len(pred_indices) == 0:
        return float("nan")
    pred_arr = np.array(pred_indices, dtype=int)
    return float(np.abs(pred_arr - int(true_index)).min())

# ---------------------------------------------------------------------
# Main ED/ES detection
# ---------------------------------------------------------------------
def detect_ed_es(
    coords,
    tmin,
    tmax,
    frames=None,
    fps=30,
    sg_window=15,
    sg_order=2,
    hp_cutoff=0.5,
    lowfreq_pow_thresh=0.1,
    prominence_ratio=0.3,
    plot_and_save=False,
    outname="trajectory_ed_es.png",
):
    coords = np.asarray(coords)
    if coords.ndim != 2:
        raise ValueError("coords must be 2-dimensional (T, D)")

    T_full, D = coords.shape
    tmin = max(0, int(tmin))
    tmax = min(T_full - 1, int(tmax))

    T, D = coords.shape
    true_ed = int(tmin)
    true_es = int(tmax)

    disp = coords[1:] - coords[:-1]
    norms = np.linalg.norm(disp, axis=1, keepdims=True)
    valid = (norms.squeeze() > 1e-8)

    d_normalized = np.zeros_like(disp)
    if np.any(valid):
        d_normalized[valid] = disp[valid] / norms[valid]
    else:
        d_normalized = disp.copy()

    X = d_normalized[:, 0].reshape(-1, 1)
    y = d_normalized[:, 1]

   
    pca = PCA(n_components=1)
    pca.fit(d_normalized)
    v = pca.components_[0]
    
    v_norm = v / (np.linalg.norm(v) + 1e-12)
    mu = coords.mean(axis=0)
    s_raw = (coords - mu).dot(v_norm)

    sg_w = _ensure_odd_window(sg_window, T)
    try:
        s_smooth = savgol_filter(s_raw, window_length=sg_w, polyorder=sg_order)
    except Exception:
        s_smooth = s_raw.copy()

    sig = s_smooth - np.mean(s_smooth)
    N = len(sig)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    psd = np.abs(np.fft.rfft(sig)) ** 2
    lf_ratio = float(psd[freqs <= hp_cutoff].sum() / (psd.sum() + 1e-12))

    if lf_ratio > lowfreq_pow_thresh:
        s_filt = _highpass_filter(s_smooth, fs=fps, cutoff=hp_cutoff)
    else:
        s_filt = s_smooth

    smin = float(np.min(s_filt))
    smax = float(np.max(s_filt))
    prom = float(prominence_ratio * (smax - smin))

    peaks_idx, _ = find_peaks(s_filt, prominence=prom)
    valleys_idx, _ = find_peaks(-s_filt, prominence=prom)

    return {
        "GroupED": valleys_idx.tolist(),
        "GroupES": peaks_idx.tolist(),
        "s_raw": s_raw,
        "projected_filtered": s_filt,
        "lf_ratio": lf_ratio,
        "prominence_used": prom,
    }

# ---------------------------------------------------------------------
# Custom Dataset
# ---------------------------------------------------------------------
class NFDataset(Dataset):
    def __init__(self, root_dir: str, tracing_csv: str = TRACING_FILE, filelist_csv: str = FILELIST_CSV):
        self.root_dir = root_dir

        if not os.path.isdir(root_dir):
            raise ValueError(f"Root directory not found: {root_dir}")

        self.files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".pt")])
        self.tracing_df = pd.read_csv(tracing_csv) if os.path.exists(tracing_csv) else pd.DataFrame()
        self.filelist_df = pd.read_csv(filelist_csv) if os.path.exists(filelist_csv) else pd.DataFrame()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path, weights_only=False)

        if "modulations" not in data:
            raise KeyError(f"'modulations' not found in {path}")

        m = data["modulations"].float()
        nf = data.get("name", None)
        filename = nf[0].split("/")[-1] if isinstance(nf, (list, tuple)) else os.path.basename(path)

        df_t = self.tracing_df[self.tracing_df["FileName"] == filename].sort_values("Frame")
        if df_t.empty:
            T = m.shape[0]
            true_ed, true_es = 0, T - 1
        else:
            frames = df_t["Frame"].values
            true_ed, true_es = int(frames.min()), int(frames.max())

        df_f = self.filelist_df[self.filelist_df["FileName"] == filename.split('.')[0]]
        ef = float(df_f["EF"].values[0]) if ("EF" in df_f.columns and not df_f.empty) else float("nan")

        if true_ed < 10 or true_es < 10:
            return m, filename, true_ed, true_es, ef, torch.tensor(False)

        number_of_frames = int(df_f["NumberOfFrames"].values[0]) if not df_f.empty else m.shape[0]

    

        cut_T = min(m.shape[0], number_of_frames)
        m = m[:number_of_frames]

        if true_ed > cut_T - 10 or true_es > cut_T - 10:
            return m, filename, true_ed, true_es, ef, torch.tensor(False)

        return m, filename, true_ed, true_es, ef, torch.tensor(True)

# ---------------------------------------------------------------------
# Match true and predicted events
# ---------------------------------------------------------------------
def match_true_to_pred(true_ed: int, true_es: int,
                       group_ed: List[int], group_es: List[int]) -> Tuple[float, float, Optional[int], Optional[int]]:

    g_ed = np.array(group_ed) if group_ed is not None else np.array([])
    g_es = np.array(group_es) if group_es is not None else np.array([])

    if len(g_ed) > 0:
        errA_ed = np.min(np.abs(g_ed - true_ed))
    else:
        errA_ed = float("nan")

    if len(g_es) > 0:
        errA_es = np.min(np.abs(g_es - true_es))
    else:
        errA_es = float("nan")

    totalA = errA_ed + errA_es

    if len(g_es) > 0:
        errB_ed = np.min(np.abs(g_es - true_ed))
    else:
        errB_ed = float("nan")

    if len(g_ed) > 0:
        errB_es = np.min(np.abs(g_ed - true_es))
    else:
        errB_es = float("nan")

    totalB = errB_ed + errB_es

    if np.isnan(totalA) and not np.isnan(totalB):
        choose = "swapped"
    elif np.isnan(totalB) and not np.isnan(totalA):
        choose = "normal"
    else:
        choose = "normal" if totalA <= totalB else "swapped"

    if choose == "normal":
        pred_ed = g_ed[np.argmin(np.abs(g_ed - true_ed))] if len(g_ed) > 0 else None
        pred_es = g_es[np.argmin(np.abs(g_es - true_es))] if len(g_es) > 0 else None
        return errA_ed, errA_es, pred_ed, pred_es
    else:
        pred_ed = g_es[np.argmin(np.abs(g_es - true_ed))] if len(g_es) > 0 else None
        pred_es = g_ed[np.argmin(np.abs(g_ed - true_es))] if len(g_ed) > 0 else None
        return errB_ed, errB_es, pred_ed, pred_es

# ---------------------------------------------------------------------
# Evaluation on a single dataset folder
# ---------------------------------------------------------------------
def evaluate_ed_es_on_dataset(dataset_root, batch_size=1):
    ds = NFDataset(dataset_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ed_maes = []
    es_maes = []

    for i, batch in enumerate(loader):
        m, filename, true_ed, true_es, ef, is_valid = batch

        if not is_valid.item():
            continue

        coords = m[0].cpu().numpy()
        det = detect_ed_es(coords, tmin=true_ed.item(), tmax=true_es.item())
      

        mae_ed, mae_es, pred_ed, pred_es = match_true_to_pred(
            true_ed.item(),
            true_es.item(),
            det["GroupED"],
            det["GroupES"],
        )

        ed_maes.append(mae_ed)
        es_maes.append(mae_es)

    mean_ed_mae = float(np.nanmean(ed_maes)) if ed_maes else float("nan")
    std_ed_mae  = float(np.nanstd(ed_maes)) if ed_maes else float("nan")
    mean_es_mae = float(np.nanmean(es_maes)) if es_maes else float("nan")
    std_es_mae  = float(np.nanstd(es_maes)) if es_maes else float("nan")

    return mean_ed_mae, std_ed_mae, mean_es_mae, std_es_mae

def evaluate_ed_es_on_dataset_return_stats(dataset_root):
    return evaluate_ed_es_on_dataset(dataset_root)


# ---------------------------------------------------------------------
# Evaluate all subfolders under ./reconstructions
# ---------------------------------------------------------------------
def evaluate_all_subfolders(root="./reconstructions", outfile="results_ed_es.csv"):

    test_dirs = sorted(glob.glob(os.path.join(root, "**/nfset/test"), recursive=True))
    print(f"Found {len(test_dirs)} test folders.")

    # --------------------------------------------------
    # Load existing CSV if it exists
    # --------------------------------------------------
    if os.path.exists(outfile):
        df_existing = pd.read_csv(outfile)
        done_folders = set(df_existing["folder"].astype(str))
        all_results = df_existing.to_dict("records")
        print(f"Loaded existing results: {len(done_folders)} entries")
    else:
        done_folders = set()
        all_results = []

    # --------------------------------------------------
    # Iterate over subfolders
    # --------------------------------------------------
    for test_dir in test_dirs:
        parent = os.path.basename(os.path.dirname(os.path.dirname(test_dir)))
        # example parent: ortho_512_2048

        if parent in done_folders:
            print(f"✓ Skipping (already exists): {parent}")
            continue

        print(f"\n=== Processing: {parent} ===")

        mean_ed, std_ed, mean_es, std_es = evaluate_ed_es_on_dataset_return_stats(test_dir)
        print("got stats", mean_ed, mean_es)

        parts = parent.split("_")
        if len(parts) < 4:
            print(f"⚠️  Unexpected folder name format: {parent}, skipping")
            continue

        prefix = parts[1]
        middledim = int(parts[2])
        vidm = int(parts[3])

        row = {
            "folder": parent,
            "prefix": prefix,
            "middledim": middledim,
            "vidm": vidm,
            "mean_ed_mae": mean_ed,
            "std_ed_mae": std_ed,
            "mean_es_mae": mean_es,
            "std_es_mae": std_es,
        }

        all_results.append(row)
        done_folders.add(parent)

        # --------------------------------------------------
        # Save incrementally (safe against crashes)
        # --------------------------------------------------
        pd.DataFrame(all_results).to_csv(outfile, index=False)

    print(f"\nSaved results to {outfile}")

    df = pd.DataFrame(all_results)


# ---------------------------------------------------------------------
def main():
     evaluate_all_subfolders(
         root="./reconstructions",
         outfile="results_ed_es.csv"
     )


if __name__ == "__main__":
    main()
