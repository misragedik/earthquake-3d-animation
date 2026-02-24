# 🌍 3D Earthquake Animation

Generate animated 3D visualizations of earthquake sequences showing spatial distribution, depth, and magnitude evolution over time.

![Example Animation](docs/example_preview.png)

## Features

- **3D depth visualization** — see how the mainshock & aftershocks distribute beneath the surface
- **Cumulative animation** — events appear one by one in chronological order
- **Configurable region** — define your own bounding box or let the tool auto-detect
- **Flexible magnitude scaling** — auto-generate bins from your data or define custom size thresholds
- **Depth coloring** — optional jet/viridis/etc. colormap with colorbar, or single-color mode
- **Fault overlay** — optional fault line display on the surface (GMT-format files)
- **Dual output** — exports both MP4 video and GIF

## Requirements

```bash
pip install pandas numpy matplotlib openpyxl
```

[FFmpeg](https://ffmpeg.org/download.html) must be installed and available in your PATH.

## Quick Start

### 1. Prepare your earthquake catalog

Your data file (`.xlsx` or `.csv`) must contain these columns:

| Column      | Type     | Description                       |
|-------------|----------|-----------------------------------|
| `Date`      | datetime | Origin date (e.g. `2023-02-06`)   |
| `Time`      | string   | Origin time (e.g. `01:17:34`)     |
| `Longitude` | float    | Epicenter longitude (°E)          |
| `Latitude`  | float    | Epicenter latitude (°N)           |
| `Depth`     | float    | Hypocentral depth (km)            |
| `Mw`        | float    | Moment magnitude                  |

> **Tip:** The catalog should be sorted chronologically. The animation adds events in the order they appear in the file.

### 2. Edit the CONFIG section

Open `earthquake_3d_animation.py` and modify the `CONFIG` dictionary at the top of the file:

```python
CONFIG = {
    # Your earthquake data file
    "earthquake_file": "my_earthquakes.csv",
    
    # Optional fault file (set None to skip)
    "fault_file": "faults.txt",
    
    # Region bounds — "auto" or [min, max]
    "lon_range": "auto",
    "lat_range": "auto",
    "depth_range": [0, 30],
    
    # Plot title
    "title": "3D Aftershock Distribution — My Earthquake (Mw 7.0)",
    
    # Magnitude bins — "auto" or custom list
    "magnitude_bins": "auto",
    
    # Color mode: "depth" (colormap) or "single" (uniform)
    "color_mode": "single",
    
    # Output
    "output_name": "my_earthquake_3d",
    ...
}
```

### 3. Run

```bash
python earthquake_3d_animation.py
```

Output files will appear in the working directory.

## Configuration Reference

### Region Bounds

```python
# Auto-detect from data with 0.15° margin
"lon_range": "auto",
"lat_range": "auto",

# Or specify manually
"lon_range": [26.0, 38.0],
"lat_range": [36.0, 40.0],
```

### Magnitude Bins

```python
# Auto-generate from your data's Mw range
"magnitude_bins": "auto",

# Or define custom bins: (threshold, marker_size)
# Events with Mw < threshold get that size
"magnitude_bins": [
    (2, 20),      # Mw < 2  → 20 pt²
    (3, 50),      # Mw 2–3  → 50 pt²
    (4, 100),     # Mw 3–4  → 100 pt²
    (5, 200),     # Mw 4–5  → 200 pt²
    (999, 500),   # Mw ≥ 5  → 500 pt²
],
```

### Color Modes

```python
# Uniform color
"color_mode": "single",
"single_color": "steelblue",

# Depth-based coloring with colorbar
"color_mode": "depth",
"colormap": "jet",        # any matplotlib colormap
```

### Camera Angle

```python
"elevation": 25,    # vertical angle (0 = side view, 90 = top-down)
"azimuth": -60,     # horizontal rotation
```

### Fault File Format

The fault file follows GMT multi-segment format — a plain text file where each fault segment is separated by `>`:

```
> 
27.123 38.456
27.234 38.567
27.345 38.678
>
28.111 39.222
28.222 39.333
```

Each line contains `longitude latitude` separated by whitespace. Segments outside the region bounds are automatically clipped.

## Examples

### Minimal — auto everything

```python
CONFIG = {
    "earthquake_file": "catalog.csv",
    "fault_file": None,
    "lon_range": "auto",
    "lat_range": "auto",
    "depth_range": [0, 50],
    "title": "Aftershock Sequence",
    "magnitude_bins": "auto",
    "color_mode": "single",
    "single_color": "steelblue",
    "colormap": "jet",
    "elevation": 25,
    "azimuth": -60,
    "auto_margin": 0.15,
    "fps": 5,
    "dpi": 100,
    "fig_size": (14, 12),
    "output_name": "earthquake_3d",
    "save_mp4": True,
    "save_gif": True,
}
```

### Full control — custom region, bins, depth coloring, faults

```python
CONFIG = {
    "earthquake_file": "2023_turkey_aftershocks.xlsx",
    "fault_file": "turkey_faults.txt",
    "lon_range": [35.5, 38.5],
    "lat_range": [36.5, 38.5],
    "depth_range": [0, 40],
    "title": "2023 Kahramanmaraş Earthquake Aftershocks (Mw 7.8)",
    "magnitude_bins": [
        (3, 10),
        (4, 40),
        (5, 100),
        (6, 250),
        (999, 600),
    ],
    "color_mode": "depth",
    "colormap": "jet",
    "single_color": "steelblue",
    "elevation": 30,
    "azimuth": -45,
    "auto_margin": 0.15,
    "fps": 10,
    "dpi": 120,
    "fig_size": (16, 12),
    "output_name": "kahramanmaras_3d",
    "save_mp4": True,
    "save_gif": True,
}
```

## How It Works

1. **Loads** your earthquake catalog and validates required columns
2. **Computes** region bounds (auto or manual) and magnitude bin mapping
3. **Clips** fault lines to the 3D bounding box (matplotlib 3D does not auto-clip)
4. **Renders** frames one by one — each frame adds the next chronological event
5. **Combines** frames into MP4 (H.264) and/or GIF using FFmpeg

## Notes

- Animation time scales linearly with event count (~0.2s per frame)
- For catalogs with 1000+ events, consider increasing `fps` or subsampling
- The Z-axis is inverted so that the surface (0 km) is at the top
- `set_box_aspect([1,1,1])` ensures a cube-like 3D view

## Author

**Misra Gedik**
Boğaziçi University, Kandilli Observatory and Earthquake Research Institute, Geodesy Department
📧 misra.gedik@yahoo.com

## License

MIT
