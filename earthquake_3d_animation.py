"""
3D Earthquake Aftershock Animation Tool
=======================================
Generates a 3D animated visualization of earthquake aftershock sequences,
showing spatial and depth distribution over time.

Outputs: MP4 video + GIF animation

Usage:
    1. Edit the CONFIG section below to match your data and region.
    2. Run:  python earthquake_3d_animation.py
    
Requirements:
    pip install pandas numpy matplotlib openpyxl
    ffmpeg (must be installed and available in PATH)

Author : Misra Gedik — Boğaziçi University, Kandilli Observatory and Earthquake Research Institute, Geodesy Department
Contact: misra.gedik@yahoo.com
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import os, shutil, time, argparse
import warnings
warnings.filterwarnings('ignore')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONFIG                                         ║
# ║  Edit this section to match YOUR earthquake data and study region.          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CONFIG = {

    # ── Data Files ──────────────────────────────────────────────────────────
    # Path to your earthquake catalog. Supported formats: .xlsx, .csv
    # Required columns: Date, Time, Longitude, Latitude, Depth, Mw
    "earthquake_file": "your_earthquake_data.xlsx",

    # (Optional) Path to fault line file.
    # Expected format: two-column (lon lat) text file with '>' as segment separator.
    # Set to None to skip fault overlay.
    "fault_file": None,

    # ── Study Region ────────────────────────────────────────────────────────
    # Geographic bounding box. Set to "auto" to derive from earthquake data
    # with a margin, or specify manually as [min, max].
    "lon_range": "auto",       # e.g. [40.0, 41.25]
    "lat_range": "auto",       # e.g. [39.0, 39.75]
    "depth_range": [0, 30],    # km — Z-axis limit

    # Margin (in degrees) added around data extent when using "auto" bounds.
    "auto_margin": 0.15,

    # ── Plot Title ──────────────────────────────────────────────────────────
    "title": "3D Aftershock Distribution",

    # ── Magnitude Scaling ───────────────────────────────────────────────────
    # Define how earthquake magnitudes map to marker sizes (in points²).
    # Format: list of (threshold, size) tuples, evaluated top-to-bottom.
    # An earthquake with Mw < first threshold gets the first size, etc.
    # The last entry acts as the catch-all for Mw >= last threshold.
    #
    # Example for a dataset with Mw 1–6:
    #   [(2, 20), (3, 50), (4, 100), (5, 200), (999, 500)]
    #   → Mw<2: 20pt², Mw 2-3: 50pt², Mw 3-4: 100pt², Mw 4-5: 200pt², Mw≥5: 500pt²
    #
    # Set to "auto" to generate bins automatically from your data's Mw range.
    "magnitude_bins": "auto",

    # ── Color Mode ──────────────────────────────────────────────────────────
    # "depth"  → color by depth using a colormap (adds colorbar)
    # "single" → uniform color for all events
    "color_mode": "single",
    "colormap": "jet",               # colormap name (used when color_mode="depth")
    "single_color": "steelblue",     # marker color  (used when color_mode="single")

    # ── Camera ──────────────────────────────────────────────────────────────
    "elevation": 25,      # vertical viewing angle (degrees)
    "azimuth": -60,       # horizontal rotation (degrees)

    # ── Animation ───────────────────────────────────────────────────────────
    "fps": 5,             # frames per second
    "dpi": 100,           # resolution of each frame
    "fig_size": (14, 12), # figure size in inches

    # ── Output ──────────────────────────────────────────────────────────────
    "output_name": "earthquake_3d_animation",   # base filename (no extension)
    "save_mp4": True,
    "save_gif": True,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         END OF CONFIG — code below                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def load_earthquake_data(filepath):
    """Load earthquake catalog from xlsx or csv."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .xlsx or .csv")

    required = {"Date", "Time", "Longitude", "Latitude", "Depth", "Mw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Your columns: {list(df.columns)}\n"
            f"Required: {sorted(required)}"
        )
    print(f"Loaded {len(df)} earthquakes from {filepath}")
    print(f"  Mw range  : {df['Mw'].min():.2f} – {df['Mw'].max():.2f}")
    print(f"  Depth     : {df['Depth'].min():.1f} – {df['Depth'].max():.1f} km")
    print(f"  Lon       : {df['Longitude'].min():.4f} – {df['Longitude'].max():.4f}")
    print(f"  Lat       : {df['Latitude'].min():.4f} – {df['Latitude'].max():.4f}")
    return df


def load_fault_data(filepath, lon_min, lon_max, lat_min, lat_max):
    """Load and clip fault segments from a GMT-style text file."""
    if filepath is None:
        return []

    fault_data = []
    current_segment = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '>':
                if current_segment:
                    fault_data.append(np.array(current_segment))
                    current_segment = []
            elif line:
                parts = line.split()
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    current_segment.append([lon, lat])
        if current_segment:
            fault_data.append(np.array(current_segment))

    # Clip to bounds (matplotlib 3D does NOT auto-clip lines)
    clipped = []
    for segment in fault_data:
        in_bounds = (
            (segment[:, 0] >= lon_min) & (segment[:, 0] <= lon_max) &
            (segment[:, 1] >= lat_min) & (segment[:, 1] <= lat_max)
        )
        if not np.any(in_bounds):
            continue
        current_sub = []
        for i in range(len(segment)):
            if in_bounds[i]:
                current_sub.append(segment[i])
            else:
                if len(current_sub) >= 2:
                    clipped.append(np.array(current_sub))
                current_sub = []
        if len(current_sub) >= 2:
            clipped.append(np.array(current_sub))

    print(f"Loaded fault data: {len(fault_data)} raw segments → {len(clipped)} clipped segments")
    return clipped


def build_magnitude_bins(df, bins_config):
    """Build magnitude → marker size mapping."""
    if bins_config != "auto":
        return bins_config

    mw_min = np.floor(df['Mw'].min())
    mw_max = np.ceil(df['Mw'].max())
    n_bins = int(mw_max - mw_min)
    if n_bins < 1:
        n_bins = 1

    # Scale marker sizes from 20 to 500 across bins
    sizes = np.linspace(20, 500, n_bins + 1).astype(int)
    bins = []
    for i in range(n_bins):
        bins.append((mw_min + i + 1, int(sizes[i])))
    bins.append((999, int(sizes[-1])))  # catch-all

    print(f"Auto magnitude bins (Mw {mw_min:.0f}–{mw_max:.0f}):")
    for thresh, size in bins:
        if thresh == 999:
            print(f"  Mw >= {bins[-2][0]:.0f} : size {size}")
        else:
            print(f"  Mw <  {thresh:.0f} : size {size}")
    return bins


def get_magnitude_size(mw, bins):
    """Map a single Mw value to marker size using the bins list."""
    for threshold, size in bins:
        if mw < threshold:
            return size
    return bins[-1][1]


def compute_bounds(df, config):
    """Compute lon/lat bounds from config or auto-detect."""
    margin = config["auto_margin"]

    if config["lon_range"] == "auto":
        lon_min = df['Longitude'].min() - margin
        lon_max = df['Longitude'].max() + margin
    else:
        lon_min, lon_max = config["lon_range"]

    if config["lat_range"] == "auto":
        lat_min = df['Latitude'].min() - margin
        lat_max = df['Latitude'].max() + margin
    else:
        lat_min, lat_max = config["lat_range"]

    depth_min, depth_max = config["depth_range"]

    print(f"Region bounds:")
    print(f"  Lon   : {lon_min:.4f} – {lon_max:.4f}")
    print(f"  Lat   : {lat_min:.4f} – {lat_max:.4f}")
    print(f"  Depth : {depth_min} – {depth_max} km")
    return lon_min, lon_max, lat_min, lat_max, depth_min, depth_max


def build_legend(bins, color):
    """Create legend handles from magnitude bins."""
    handles = []
    labels_added = []
    for i, (threshold, size) in enumerate(bins):
        if threshold == 999:
            label = f"Mw ≥ {bins[i-1][0]:.0f}"
        else:
            label = f"Mw < {threshold:.0f}" if i == 0 else f"Mw {bins[i-1][0]:.0f}–{threshold:.0f}"
        
        # Avoid duplicate first label
        if i == 0:
            label = f"Mw < {threshold:.0f}"

        handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color if isinstance(color, str) else 'gray',
                       markersize=np.sqrt(size) / 2,
                       markeredgecolor='black', markeredgewidth=0.5,
                       label=label)
        )
    return handles


def run(config):
    """Main animation pipeline."""
    t0 = time.time()

    # ── Load data ───────────────────────────────────────────────────────
    df = load_earthquake_data(config["earthquake_file"])
    mag_bins = build_magnitude_bins(df, config["magnitude_bins"])
    lon_min, lon_max, lat_min, lat_max, depth_min, depth_max = compute_bounds(df, config)
    faults = load_fault_data(config["fault_file"], lon_min, lon_max, lat_min, lat_max)

    # ── Color setup ─────────────────────────────────────────────────────
    use_depth_color = config["color_mode"] == "depth"
    if use_depth_color:
        cmap = cm.get_cmap(config["colormap"])
        norm = Normalize(vmin=df['Depth'].min(), vmax=df['Depth'].max())

    # ── Frame directory ─────────────────────────────────────────────────
    frame_dir = '_frames_3d_temp'
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)

    # ── Figure setup ────────────────────────────────────────────────────
    fig = plt.figure(figsize=config["fig_size"])
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=config["elevation"], azim=config["azimuth"])
    ax.set_box_aspect([1, 1, 1])

    # Draw faults at surface (z=0)
    for segment in faults:
        zs = np.zeros(len(segment))
        ax.plot(segment[:, 0], segment[:, 1], zs, 'k-', linewidth=0.5, alpha=0.6)

    # Axes
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_zlim(depth_max, depth_min)  # Inverted: surface on top
    ax.set_xlabel('Longitude (°)', fontsize=11, labelpad=12)
    ax.set_ylabel('Latitude (°)', fontsize=11, labelpad=12)
    ax.set_zlabel('Depth (km)', fontsize=11, labelpad=10)
    ax.set_title(config["title"], fontsize=13, pad=25)
    ax.tick_params(labelsize=9)

    # Colorbar (depth mode only)
    if use_depth_color:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.55])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Depth (km)', fontsize=11)
        fig.subplots_adjust(right=0.85)

    # Legend
    marker_color = 'gray' if use_depth_color else config["single_color"]
    legend_handles = build_legend(mag_bins, marker_color)
    ax.legend(handles=legend_handles, loc='upper left', title='Magnitude',
              framealpha=0.9, fontsize=9, title_fontsize=10)

    # Subtitle
    subtitle_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10, style='italic')

    # ── Render frames ───────────────────────────────────────────────────
    scatter_obj = None
    total = len(df)
    print(f"\nRendering {total + 1} frames...")

    for frame in range(total + 1):
        if scatter_obj is not None:
            scatter_obj.remove()
            scatter_obj = None

        if frame > 0:
            idx = frame - 1
            current_df = df.iloc[:idx + 1]
            lons = current_df['Longitude'].values
            lats = current_df['Latitude'].values
            depths = current_df['Depth'].values
            sizes = np.array([get_magnitude_size(mw, mag_bins) for mw in current_df['Mw']])

            if use_depth_color:
                scatter_obj = ax.scatter(
                    lons, lats, depths, s=sizes,
                    c=depths, cmap=cmap, norm=norm,
                    alpha=0.8, edgecolors='black', linewidths=0.5,
                    depthshade=False)
            else:
                scatter_obj = ax.scatter(
                    lons, lats, depths, s=sizes,
                    c=config["single_color"], alpha=0.7,
                    edgecolors='black', linewidths=0.5,
                    depthshade=True)

            last = current_df.iloc[-1]
            try:
                last_date = last['Date'].strftime('%Y-%m-%d')
            except AttributeError:
                last_date = str(last['Date'])
            last_time = last['Time']
            last_mw = last['Mw']
            subtitle_text.set_text(
                f"Date: {last_date} {last_time} | Mw: {last_mw:.1f} | "
                f"Total Events: {idx + 1}/{total}")
        else:
            subtitle_text.set_text('')

        fig.savefig(f'{frame_dir}/frame_{frame:04d}.png',
                    dpi=config["dpi"], facecolor='white', edgecolor='none')

        if frame % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {frame}/{total} ({elapsed:.0f}s)")

    plt.close(fig)
    print(f"All frames rendered in {time.time() - t0:.0f}s")

    # ── Combine into video / gif ────────────────────────────────────────
    base = config["output_name"]
    fps = config["fps"]

    if config["save_mp4"]:
        print("Creating MP4...")
        os.system(
            f'ffmpeg -y -framerate {fps} -i {frame_dir}/frame_%04d.png '
            f'-c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast '
            f'"{base}.mp4" 2>/dev/null')

    if config["save_gif"]:
        print("Creating GIF...")
        os.system(
            f'ffmpeg -y -framerate {fps} -i {frame_dir}/frame_%04d.png '
            f'-vf "fps={fps},scale=800:-1:flags=lanczos,'
            f'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" '
            f'"{base}.gif" 2>/dev/null')

    # Cleanup
    shutil.rmtree(frame_dir)
    total_time = time.time() - t0
    print(f"\nDone! Total time: {total_time:.0f}s")
    if config["save_mp4"]:
        print(f"  → {base}.mp4")
    if config["save_gif"]:
        print(f"  → {base}.gif")


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(CONFIG)
