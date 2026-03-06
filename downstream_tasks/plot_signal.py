#!/usr/bin/env python3

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ============================================================
# CONFIG
# ============================================================

FPS = 30

FILENAME_ORTHO = "0X24708E2D3F05391F.avi"
FILENAME_NONORTHO = "0X102CFB07F752AAE6.avi"

TRUE_ED_ORTHO, TRUE_ES_ORTHO = 133, 149
TRUE_ED_NONORTHO, TRUE_ES_NONORTHO = 163, 184

DIR_ORTHO = "./recontructions/cardiac_ortho_2/nfset/test"
DIR_NONORTHO = "./recontructions/cardiac_basic_2/nfset/test"

VIDEO_DIR_ORTHO = "./recontructions/cardiac_ortho_2/videos"
VIDEO_DIR_NONORTHO = "./recontructions/cardiac_basic_2/videos"

OUTPUT_DIR = "./predicted_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def find_file(root_dir, filename):
    for f in os.listdir(root_dir):
        if not f.endswith(".pt"):
            continue
        path = os.path.join(root_dir, f)
        data = torch.load(path, weights_only=False)
        name = os.path.basename(data["name"][0])
        if name == filename:
            return path
    raise FileNotFoundError(filename)

def load_frame(video_dir, filename, frame_idx):
    path = os.path.join(video_dir, filename)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Frame read failed")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def save_frame_png(video_dir, filename, frame_idx, label):
    frame = load_frame(video_dir, filename, frame_idx)
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{filename}_{label}_{frame_idx}.png"
    )
    plt.figure(figsize=(4,4))
    plt.imshow(frame)
    plt.axis("off")
    plt.title(f"{label} frame {frame_idx}")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

def highpass_filter(signal, fps, cutoff=0.5):
    nyq = 0.5 * fps
    b, a = butter(2, cutoff / nyq, btype="high")
    return filtfilt(b, a, signal)

def compute_motion(coords):
    coords = np.asarray(coords)
    pca = PCA(n_components=1)
    pca.fit(coords)
    direction = pca.components_[0]
    mean_point = coords.mean(axis=0)
    proj = np.dot(coords - mean_point, direction)
    filtered = savgol_filter(proj, 15, 2)
    filtered = highpass_filter(filtered, FPS, cutoff=0.1)
    prominence = 0.1 * (filtered.max() - filtered.min())
    peaks, _ = find_peaks(filtered, prominence=prominence)
    valleys, _ = find_peaks(-filtered, prominence=prominence)
    detected_es = peaks
    detected_ed = valleys
    return proj, filtered, direction, mean_point, detected_ed, detected_es

def resample_trajectory_equidistant(coords: np.ndarray, spacing: float = None):
    """
    Resample a trajectory (T,D) to equidistant points along arc length.
    If spacing is None, uses 2 * smallest consecutive distance.
    """
    coords = np.asarray(coords)
    if coords.shape[0] < 2:
        return coords.copy()

    # consecutive distances
    deltas = coords[1:] - coords[:-1]
    step_dists = np.linalg.norm(deltas, axis=1)

    min_step = np.min(step_dists[step_dists > 1e-8])  # avoid zero
    if spacing is None:
        spacing = 20.0 * min_step

    # cumulative arc length
    arc = np.concatenate([[0], np.cumsum(step_dists)])
    total_length = arc[-1]

    if total_length < spacing:
        return coords.copy()

    # new sample positions
    new_arc = np.arange(0, total_length, spacing)

    # interpolate each dimension
    new_coords = np.zeros((len(new_arc), coords.shape[1]))
    for d in range(coords.shape[1]):
        new_coords[:, d] = np.interp(new_arc, arc, coords[:, d])
    print('new coords', new_coords.shape, coords.shape)
    return new_coords
# -----------------------------
# PLOTTING
# -----------------------------
def plot_trajectory(ax, coords, direction, mean_point,
                    detected_ed, detected_es):
    coords = np.asarray(coords)
    #coords = resample_trajectory_equidistant(coords)
    t = np.arange(len(coords))
    
    # Scatter with color by frame index
    sc = ax.scatter(coords[:,0], coords[:,1], c=t, cmap="viridis", s=20)
    
    # Trajectory line
    ax.plot(coords[:,0], coords[:,1], color="gray", alpha=0.3, label = r"Trajectoy $\phi$ ")
    
    # PCA line
    proj = np.dot(coords - mean_point, direction)
    t_vals = np.linspace(proj.min(), proj.max(), 100)
    line = mean_point + np.outer(t_vals, direction)
    ax.plot(line[:,0], line[:,1], color="purple", linewidth=2, label='Principal motion direction p')
    
    # Detected ED/ES points
    ax.scatter(coords[detected_ed,0],
               coords[detected_ed,1],
               color="red",
               marker="v",
               s=120,
               label="Detected ED")
    ax.scatter(coords[detected_es,0],
               coords[detected_es,1],
               color="orange",
               marker="^",
               s=120,
               label="Detected ES")
    
    ax.legend()
    
    # --- Colorbar using make_axes_locatable ---
    divider = make_axes_locatable(ax)
    # 'size' is width of the colorbar (narrower), 'pad' is distance from main plot
    cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Time")

def plot_signal(ax, raw_signal, filtered, detected_ed, detected_es,
                true_ed, true_es):
    # Plot raw signal in gray
    ax.plot(raw_signal, color="gray", linewidth=1.5, alpha=0.7, label=r"Raw signal $s_t$ ")
    
    # Plot filtered signal
    ax.plot(filtered, color="blue", linewidth=2, label=r"Filtered signal $s_{filt}$ ")
    
    # Detected peaks
    ax.scatter(detected_ed,
               filtered[detected_ed],
               color="red",
               marker="v",
               s=120, label="Detected ED")
    ax.scatter(detected_es,
               filtered[detected_es],
               color="orange",
               marker="^",
               s=120, label="Detected ES")
    
    # True ED/ES lines
    ax.axvline(true_ed, color="red", linestyle="--", linewidth=2, label="True ED")
    ax.axvline(true_es, color="orange", linestyle="--", linewidth=2, label="True ES")
    ax.set_xlabel('Time')
    ax.legend(loc='lower left')

    ax.grid(True)
# -----------------------------
# MAIN PROCESSING
# -----------------------------

def process_case(coords, filename, video_dir,
                 true_ed, true_es, ax_traj, ax_sig, title):
    coords = np.asarray(coords)
    
    # Compute motion
    proj, filtered, direction, mean_point, detected_ed, detected_es = compute_motion(coords)
    
    # Trajectory plot with proper colorbar height
    plot_trajectory(ax_traj, coords, direction, mean_point, detected_ed, detected_es)
    
    # Signal plot: raw proj in gray + filtered signal
    plot_signal(ax_sig, raw_signal=proj, filtered=filtered,
                detected_ed=detected_ed, detected_es=detected_es,
                true_ed=true_ed, true_es=true_es)
    
    # Save predicted frames
    for idx in detected_ed:
        save_frame_png(video_dir, filename, idx, "predicted_ED")
    for idx in detected_es:
        save_frame_png(video_dir, filename, idx, "predicted_ES")
# ============================================================
# MAIN
# ============================================================

def main():

    coords_ortho = -torch.load(find_file(DIR_ORTHO, FILENAME_ORTHO), weights_only=False)["modulations"].numpy()
    coords_nonortho = torch.load(find_file(DIR_NONORTHO, FILENAME_NONORTHO), weights_only=False)["modulations"].numpy()

    # Define height for squares and width for 4 columns
    square_size = 5  # inches per square
    fig, axes = plt.subplots(1, 4, figsize=(4*square_size, square_size*0.8))

    # Make all subplots square
   # for ax in axes:
    #    ax.set_box_aspect(1)  # ensures x and y lengths are equal

    # Orthogonal case
    process_case(
        coords_ortho,
        FILENAME_ORTHO,
        VIDEO_DIR_ORTHO,
        TRUE_ED_ORTHO,
        TRUE_ES_ORTHO,
        axes[0],
        axes[1],
        "Orthogonal"
    )

    # Non-Orthogonal case
    process_case(
        coords_nonortho,
        FILENAME_NONORTHO,
        VIDEO_DIR_NONORTHO,
        TRUE_ED_NONORTHO,
        TRUE_ES_NONORTHO,
        axes[2],
        axes[3],
        "Non-Orthogonal"
    )

    plt.tight_layout()
    plt.savefig("trajectories.png", dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================

if __name__ == "__main__":
    main()