import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pyproj import Transformer
import zipfile
import xml.etree.ElementTree as ET
import os
from matplotlib.gridspec import GridSpec

# =========================
# FILE PATHS
# =========================
base_folder = r"file directory"

excel_file = os.path.join(base_folder, "overview data.xlsx")
cycle1_kmz = os.path.join(base_folder, "CYCLE1.kmz")
cycle2_kmz = os.path.join(base_folder, "CYCLE2.kmz")
cycle3_kmz = os.path.join(base_folder, "CYCLE3.kmz")

lat_col = "LATITUDE"
lon_col = "LONGITUDE"
label_col_points = "LOG NR"
cycle_cols = ["CYCLE 1", "CYCLE 2", "CYCLE 3"]

# =========================
# SETTINGS
# =========================
grid_res = 200
power = 1.9
n_levels = 15
padding_fraction = 0.05
cmap = "viridis"

label_fontsize = 9

# Scale bar
scale_bar_length_m = 1000

# Red dot
red_dot_size = 100
red_dot_color = "red"

# Target location for red dot
dot_lat_str = '47° 1\'55.00"N'
dot_lon_str = '104°38\'30.88"W'

# Smaller colorbars
colorbar_shrink = 0.75
colorbar_pad = 0.02
colorbar_fraction = 0.05

# =========================
# PATH ORIENTATION SETTINGS
# =========================
cycle1_path_color = "blue"
cycle1_path_linewidth = 2.5
cycle1_path_length_m = 1200

cycle2_path_color = "black"
cycle2_path_linewidth = 2.5
cycle2_path_length_m = 1200

cycle3_path_color = "magenta"
cycle3_path_linewidth = 2.5
cycle3_path_length_m = 1200

# =========================
# DMS PARSER
# =========================
def dms_to_decimal(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()

    try:
        return float(s)
    except ValueError:
        pass

    s = (
        s.replace("º", "°")
         .replace("’", "'")
         .replace("‘", "'")
         .replace("″", '"')
         .replace("“", '"')
         .replace("”", '"')
    )

    match = re.match(
        r'^\s*(\d+)\s*°\s*(\d+)\s*\'\s*([\d.]+)\s*"\s*([NSEW])\s*$',
        s,
        flags=re.IGNORECASE
    )

    if not match:
        return np.nan

    deg, minutes, seconds, direction = match.groups()
    decimal = float(deg) + float(minutes) / 60 + float(seconds) / 3600

    if direction.upper() in ("S", "W"):
        decimal *= -1

    return decimal

# =========================
# UTM HELPER
# =========================
def get_utm_epsg(lat, lon):
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

# =========================
# IDW INTERPOLATION
# =========================
def idw_interpolation(x, y, z, xi, yi, power):
    xg = xi[..., None]
    yg = yi[..., None]

    dist = np.sqrt((xg - x) ** 2 + (yg - y) ** 2)
    zi = np.empty(xi.shape)

    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            d = dist[i, j]

            if np.any(d == 0):
                zi[i, j] = z[d == 0][0]
            else:
                w = 1 / (d ** power)
                zi[i, j] = np.sum(w * z) / np.sum(w)

    return zi

# =========================
# SCALE BAR
# =========================
def add_scale_bar(ax, length_m, xmin, xmax, ymin, ymax):
    x0 = xmin + 0.08 * (xmax - xmin)
    y0 = ymin + 0.08 * (ymax - ymin)

    ax.plot([x0, x0 + length_m], [y0, y0], color="black", linewidth=3)

    if length_m >= 1000 and length_m % 1000 == 0:
        label = f"{int(length_m / 1000)} km"
    else:
        label = f"{int(length_m)} m"

    ax.text(
        x0 + length_m / 2,
        y0 + 0.02 * (ymax - ymin),
        label,
        ha="center"
    )

# =========================
# KMZ / KML PATH READER
# =========================
def read_kmz_linestrings(kmz_path):
    if not os.path.exists(kmz_path):
        print(f"Warning: KMZ file not found: {kmz_path}")
        return []

    with zipfile.ZipFile(kmz_path, "r") as zf:
        kml_files = [name for name in zf.namelist() if name.lower().endswith(".kml")]
        if not kml_files:
            print(f"Warning: No KML file found inside {kmz_path}")
            return []

        with zf.open(kml_files[0]) as f:
            tree = ET.parse(f)
            root = tree.getroot()

    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    paths = []
    for linestring in root.findall(".//kml:LineString", ns):
        coord_elem = linestring.find("kml:coordinates", ns)
        if coord_elem is None or coord_elem.text is None:
            continue

        coord_text = coord_elem.text.strip()
        coords = []

        for chunk in coord_text.split():
            parts = chunk.split(",")
            if len(parts) < 2:
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            coords.append((lon, lat))

        if len(coords) >= 2:
            paths.append(coords)

    return paths

# =========================
# PATH ORIENTATION PLOTTING
# =========================
def plot_path_orientations(ax, paths_lonlat, transformer, color="black", linewidth=2, length_m=1000):
    for path in paths_lonlat:
        lons = np.array([pt[0] for pt in path])
        lats = np.array([pt[1] for pt in path])

        x, y = transformer.transform(lons, lats)

        x_mid = np.mean(x)
        y_mid = np.mean(y)

        dx = x[-1] - x[0]
        dy = y[-1] - y[0]

        norm = np.sqrt(dx**2 + dy**2)
        if norm == 0:
            continue

        ux = dx / norm
        uy = dy / norm

        half_len = length_m / 2
        x0 = x_mid - ux * half_len
        x1 = x_mid + ux * half_len
        y0 = y_mid - uy * half_len
        y1 = y_mid + uy * half_len

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth, zorder=6)

# =========================
# READ EXCEL DATA
# =========================
df = pd.read_excel(excel_file)

required_cols = [lat_col, lon_col, label_col_points] + cycle_cols
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df[lat_col] = df[lat_col].apply(dms_to_decimal)
df[lon_col] = df[lon_col].apply(dms_to_decimal)

for col in cycle_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df_points = df[[lat_col, lon_col, label_col_points]].dropna()

if len(df_points) == 0:
    raise ValueError("No valid coordinates found.")

# =========================
# PROJECTION
# =========================
mean_lat = df_points[lat_col].mean()
mean_lon = df_points[lon_col].mean()

utm_epsg = get_utm_epsg(mean_lat, mean_lon)
transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)

x_all, y_all = transformer.transform(df_points[lon_col], df_points[lat_col])
df_points["X"] = x_all
df_points["Y"] = y_all

# Dot location
dot_lat = dms_to_decimal(dot_lat_str)
dot_lon = dms_to_decimal(dot_lon_str)
dot_x, dot_y = transformer.transform(dot_lon, dot_lat)

# Extents
xmin, xmax = df_points["X"].min(), df_points["X"].max()
ymin, ymax = df_points["Y"].min(), df_points["Y"].max()

pad_x = (xmax - xmin) * padding_fraction
pad_y = (ymax - ymin) * padding_fraction

xmin -= pad_x
xmax += pad_x
ymin -= pad_y
ymax += pad_y

# Grid
xi = np.linspace(xmin, xmax, grid_res)
yi = np.linspace(ymin, ymax, grid_res)
XI, YI = np.meshgrid(xi, yi)

# =========================
# READ KMZ PATHS
# =========================
cycle1_paths = read_kmz_linestrings(cycle1_kmz)
cycle2_paths = read_kmz_linestrings(cycle2_kmz)
cycle3_paths = read_kmz_linestrings(cycle3_kmz)

print(f"Found {len(cycle1_paths)} paths in CYCLE1.kmz")
print(f"Found {len(cycle2_paths)} paths in CYCLE2.kmz")
print(f"Found {len(cycle3_paths)} paths in CYCLE3.kmz")

# =========================
# FIGURE LAYOUT
# =========================
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(
    nrows=3, ncols=2, figure=fig,
    width_ratios=[1.15, 1.0],
    height_ratios=[1, 1, 1],
    wspace=0.25, hspace=0.28
)

# Left half: top two-thirds
ax_loc = fig.add_subplot(gs[0:2, 0])

# Right half: three stacked maps
ax_c1 = fig.add_subplot(gs[0, 1])
ax_c2 = fig.add_subplot(gs[1, 1])
ax_c3 = fig.add_subplot(gs[2, 1])

# Blank lower-left panel to preserve layout
ax_blank = fig.add_subplot(gs[2, 0])
ax_blank.axis("off")

# =========================
# MAP 1: LOCATIONS
# =========================
ax = ax_loc

ax.scatter(df_points["X"], df_points["Y"], edgecolor="black")

for _, row in df_points.iterrows():
    ax.text(row["X"], row["Y"], str(row[label_col_points]), fontsize=label_fontsize)

ax.scatter(dot_x, dot_y, s=red_dot_size, color=red_dot_color, zorder=7)

add_scale_bar(ax, scale_bar_length_m, xmin, xmax, ymin, ymax)

ax.set_title("Log Locations")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_xlabel(f"Easting (m) - UTM EPSG:{utm_epsg}")
ax.set_ylabel(f"Northing (m) - UTM EPSG:{utm_epsg}")

# =========================
# HELPER TO DRAW CYCLE MAP
# =========================
def draw_cycle_map(ax, cycle_name, kmz_paths=None, path_color="black", path_linewidth=2.5, path_length_m=1200):
    df_cycle = df[[lat_col, lon_col, cycle_name]].dropna()

    if len(df_cycle) < 3:
        ax.set_title(f"{cycle_name} (not enough data)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        add_scale_bar(ax, scale_bar_length_m, xmin, xmax, ymin, ymax)
        ax.scatter(dot_x, dot_y, s=red_dot_size, color=red_dot_color, zorder=7)
        ax.set_xlabel(f"Easting (m) - UTM EPSG:{utm_epsg}")
        ax.set_ylabel(f"Northing (m) - UTM EPSG:{utm_epsg}")
        return

    x, y = transformer.transform(df_cycle[lon_col], df_cycle[lat_col])
    z = df_cycle[cycle_name].values

    ZI = idw_interpolation(x, y, z, XI, YI, power)

    cf = ax.contourf(XI, YI, ZI, levels=n_levels, cmap=cmap)
    ax.contour(XI, YI, ZI, levels=n_levels, colors="black", linewidths=0.5)

    ax.scatter(x, y, edgecolor="black")

    if kmz_paths is not None:
        plot_path_orientations(
            ax,
            kmz_paths,
            transformer,
            color=path_color,
            linewidth=path_linewidth,
            length_m=path_length_m
        )

    ax.scatter(dot_x, dot_y, s=red_dot_size, color=red_dot_color, zorder=7)
    add_scale_bar(ax, scale_bar_length_m, xmin, xmax, ymin, ymax)

    cbar = fig.colorbar(
        cf,
        ax=ax,
        shrink=colorbar_shrink,
        pad=colorbar_pad,
        fraction=colorbar_fraction
    )
    cbar.set_label(cycle_name)

    ax.set_title(cycle_name)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Easting (m) - UTM EPSG:{utm_epsg}")
    ax.set_ylabel(f"Northing (m) - UTM EPSG:{utm_epsg}")

# =========================
# DRAW CYCLE MAPS
# =========================
draw_cycle_map(
    ax_c1,
    "CYCLE 1",
    kmz_paths=cycle1_paths,
    path_color=cycle1_path_color,
    path_linewidth=cycle1_path_linewidth,
    path_length_m=cycle1_path_length_m
)

draw_cycle_map(
    ax_c2,
    "CYCLE 2",
    kmz_paths=cycle2_paths,
    path_color=cycle2_path_color,
    path_linewidth=cycle2_path_linewidth,
    path_length_m=cycle2_path_length_m
)

draw_cycle_map(
    ax_c3,
    "CYCLE 3",
    kmz_paths=cycle3_paths,
    path_color=cycle3_path_color,
    path_linewidth=cycle3_path_linewidth,
    path_length_m=cycle3_path_length_m
)

plt.tight_layout()

#plt.savefig(os.path.join(base_folder, "isopach_maps.svg"), format="svg")

plt.show()