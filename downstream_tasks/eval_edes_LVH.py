#!/usr/bin/env python3
"""
Compute MAE for ED/ES across ALL subfolders inside ./reconstructions_tracking
Each subfolder name must follow: <prefix>_<middledim>_<vidm>
Example: ortho_512_2048, separate_256_1024

Outputs:
- results_ed_es.csv  (summaries from all folders)
- mae_vs_middledim.png (ED MAE vs middle dim, lines = vidm)
"""
from torch.utils.data import Dataset
from pathlib import Path
import os
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
from sklearn.linear_model import LinearRegression, RANSACRegressor

import torch
from torch.utils.data import Dataset, DataLoader

# ------------------- USER-PATHS -------------------
TRACING_FILE = "./data/EchoNet-Dynamic/VolumeTracings.csv"
FILELIST_CSV = "./EchoNet-Dynamic/FileList.csv"

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
    sg_window=5,
    sg_order=2,
    hp_cutoff=0.5,
    lowfreq_pow_thresh=0.1,  #was 0.1
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
    inlier_mask = np.ones(len(d_normalized), dtype=bool)

    if valid.sum() >= 2:
        try:
            ransac = RANSACRegressor(
                estimator=LinearRegression(), residual_threshold=0.2, random_state=0
            )
            ransac.fit(X[valid], y[valid])
            inlier_mask = np.zeros(len(d_normalized), dtype=bool)
            inlier_idx = np.where(valid)[0]
            inlier_mask[inlier_idx[ransac.inlier_mask_]] = True
        except Exception:
            pass

    inlier_vectors = d_normalized[inlier_mask]
    if inlier_vectors.shape[0] < 2:
        inlier_vectors = d_normalized.copy()

    try:
        pca = PCA(n_components=1)
        pca.fit(d_normalized)
        v = pca.components_[0]
    except Exception:
        v = np.array([1.0, 0.0])

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
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

        if not os.path.isdir(root_dir):
            raise ValueError(f"Root directory not found: {root_dir}")

     #   self.files = sorted(
      #      [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".pt")]
     #   )
        nfset_dir = Path(root_dir) / "nfset"

        if not nfset_dir.is_dir():
            raise ValueError(f"nfset directory not found: {nfset_dir}")

        # find all .pt files recursively in nfset and its subfolders
        self.files = sorted(str(p) for p in nfset_dir.rglob("*.pt"))

        print(f"Found {len(self.files)} .pt files.")

        self.csv_root = (
            "./data/POCUS"
    
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path, weights_only=False)

        if "modulations" not in data:
            raise KeyError(f"'modulations' not found in {path}")

        m = data["modulations"].float()

        # -------------------------
        # Filename handling
        # -------------------------
        filename = data["name"][0].split("/")[-1]
        name = filename.split(".")[0]
        
        # -------------------------
        # Load ED / ES from CSV
        # -------------------------
        csv_path = os.path.join(self.csv_root, name + ".csv")
       

        image_nos = []

        with open(csv_path, "r") as f:
            next(f)  # skip header
            for line in f:
                try:
                    image_nos.append(int(line.split(",")[0]))
                except ValueError:
                    pass

        unique, counts = np.unique(image_nos,return_counts=True)
       
        try:

            # ED: appears once
            # ES: appears more than once
            idx_min = unique[counts == 1][0]
            idx_max = unique[counts > 1][0]

        except Exception as e:
            # Safe fallback
            idx_min = 1
            idx_max = 1

        true_ed = int(idx_min)
        true_es = int(idx_max)

        if true_ed == 0 or true_es ==0:
            print('name 0 ed', name)

      

        # -------------------------
        # Validity checks (unchanged)
        # -------------------------

        if true_ed ==1  or true_es == 1:
            return m, filename, true_ed, true_es, float("nan"), torch.tensor(False)


        return m, filename, true_ed, true_es, float("nan"), torch.tensor(True)

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
    print('root', dataset_root)
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

        print('filename', filename, 'mae', mae_ed, mae_es)

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
def evaluate_all_subfolders(root="./reconstructions",
                            outfile="results_ed_es_POCUS.csv"):

    root = Path(root)

    # --------------------------------------------------
    # Find experiment folders (those containing nfset/)
    # --------------------------------------------------
    experiment_dirs = sorted(
        p.parent for p in root.glob("*/nfset") if p.is_dir()
    )

    print(f"Found {len(experiment_dirs)} experiment folders.")

    all_results = []

    for exp_dir in experiment_dirs:
        parent = exp_dir.name  # e.g. ortho_512_2048

        print(f"\n=== Processing: {parent} ===")

        mean_ed, std_ed, mean_es, std_es = \
            evaluate_ed_es_on_dataset_return_stats(str(exp_dir))

        parts = parent.split("_")
        if len(parts) < 3:
            print(f"⚠️ Unexpected folder format: {parent}, skipping")
            continue
        print('parts', parts)
        prefix = parts[0]
        middledim = int(parts[-2])
        vidm = int(parts[-1])

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

        # Save incrementally
        pd.DataFrame(all_results).to_csv(outfile, index=False)

    print(f"\nSaved results to {outfile}")

    df = pd.DataFrame(all_results)

# ---------------------------------------------------------------------
def main():
     evaluate_all_subfolders(
         root="./reconstructions_POCUS",
         outfile="results_ed_es_POCUS.csv"
     )



if __name__ == "__main__":
    main()
