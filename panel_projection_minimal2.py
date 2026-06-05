#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer


BOUNDARY_MARKER = "0"
VERTICAL_ZERO_MARKER = "topkool0"
WGS84_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


@dataclass(frozen=True)
class ProjectionResult:
    dataframe: pd.DataFrame
    centroid_enu: np.ndarray
    rotation_matrix: np.ndarray


def wgs84_to_ecef(latitude: float, longitude: float, height: float) -> np.ndarray:
    x, y, z = WGS84_TO_ECEF.transform(longitude, latitude, height)
    return np.array([x, y, z], dtype=float)


def ecef_to_enu(
    x: float,
    y: float,
    z: float,
    origin_latitude: float,
    origin_longitude: float,
    origin_height: float,
) -> np.ndarray:
    x0, y0, z0 = wgs84_to_ecef(origin_latitude, origin_longitude, origin_height)
    dx, dy, dz = x - x0, y - y0, z - z0

    phi = np.radians(origin_latitude)
    lam = np.radians(origin_longitude)

    rotation_ecef_to_enu = np.array(
        [
            [-np.sin(lam), np.cos(lam), 0.0],
            [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
        ]
    )

    return rotation_ecef_to_enu @ np.array([dx, dy, dz], dtype=float)


def convert_wgs84_points_to_enu(coordinates: np.ndarray) -> np.ndarray:
    origin_latitude, origin_longitude, origin_height = coordinates[0]

    return np.array(
        [
            ecef_to_enu(
                *wgs84_to_ecef(latitude, longitude, height),
                origin_latitude,
                origin_longitude,
                origin_height,
            )
            for latitude, longitude, height in coordinates
        ]
    )


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)

    axis = np.cross(source, target)
    axis_length = np.linalg.norm(axis)

    if axis_length < 1e-10:
        return np.eye(3)

    axis = axis / axis_length
    angle = np.arccos(np.clip(np.dot(source, target), -1.0, 1.0))

    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )

    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def fit_plane_and_flatten(
    points_enu: np.ndarray,
    reference_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_points = points_enu[list(reference_indices)]
    centroid = np.mean(reference_points, axis=0)
    centered_reference_points = reference_points - centroid

    _, _, vh = np.linalg.svd(centered_reference_points)
    plane_normal = vh[-1]

    if plane_normal[2] < 0:
        plane_normal = -plane_normal

    rotation_matrix = rotation_matrix_from_vectors(plane_normal, np.array([0.0, 0.0, 1.0]))
    rotated_points = (rotation_matrix @ (points_enu - centroid).T).T

    return rotated_points, centroid, rotation_matrix


def split_label(raw_label: str) -> tuple[str, str]:
    raw_label = raw_label.strip()

    if " " in raw_label:
        number_part, text_part = raw_label.split(" ", 1)
    elif "(" in raw_label:
        number_part, text_part = raw_label.split("(", 1)
        text_part = "(" + text_part
    else:
        number_part, text_part = raw_label, ""

    return number_part.strip(), text_part.strip()


def read_point_file(file_path: Path, boundary_marker: str = BOUNDARY_MARKER) -> pd.DataFrame:
    rows = []

    with file_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            parts = line.strip().split(",")
            if len(parts) < 4:
                raise ValueError(f"Line {line_number} is malformed: {line.strip()}")

            raw_label = parts[0].strip()
            longitude = float(parts[1].strip())
            latitude = float(parts[2].strip())
            height = float(parts[3].strip())

            number_part, text_part = split_label(raw_label)
            boundary_flag = "BOUNDARY" if boundary_marker in text_part else "NO"

            rows.append([number_part, boundary_flag, longitude, latitude, height, text_part])

    return pd.DataFrame(rows, columns=["NR", "BOUNDARY?", "LONG", "LAT", "HEIGHT", "INFO"])


def add_rotated_coordinates(df: pd.DataFrame) -> ProjectionResult:
    coordinates = df[["LAT", "LONG", "HEIGHT"]].astype(float).values
    boundary_flags = df["BOUNDARY?"].astype(str).str.strip().str.upper()
    reference_indices = df.index[boundary_flags == "BOUNDARY"].tolist()

    if not reference_indices:
        raise ValueError("No reference-plane points were found.")

    points_enu = convert_wgs84_points_to_enu(coordinates)
    rotated_points, centroid, rotation_matrix = fit_plane_and_flatten(points_enu, reference_indices)

    result_df = df.copy()
    result_df["X"] = rotated_points[:, 0]
    result_df["Y"] = rotated_points[:, 1]
    result_df["Z"] = rotated_points[:, 2]

    return ProjectionResult(result_df, centroid, rotation_matrix)


def vertically_normalize_groups(
    df: pd.DataFrame,
    zero_marker: str = VERTICAL_ZERO_MARKER,
) -> tuple[pd.DataFrame, list[str]]:
    normalized_df = df.copy()
    normalized_df["NR_PREFIX"] = normalized_df["NR"].astype(str).str.split(".").str[0]
    adjusted_prefixes: list[str] = []

    for prefix, group in normalized_df.groupby("NR_PREFIX"):
        if prefix.lower() == "nan" or prefix == "":
            continue

        zero_marker_mask = group["INFO"].str.lower().str.contains(zero_marker.lower(), na=False)
        if zero_marker_mask.any():
            z_shift = -group.loc[zero_marker_mask, "Z"].iloc[0]
            normalized_df.loc[normalized_df["NR_PREFIX"] == prefix, "Z"] += z_shift
            adjusted_prefixes.append(prefix)

    return normalized_df, adjusted_prefixes


def filter_plot_points(df: pd.DataFrame, adjusted_prefixes: Iterable[str]) -> pd.DataFrame:
    return df[
        df["NR"].notna()
        & (df["NR"] != "")
        & df["NR_PREFIX"].isin(list(adjusted_prefixes))
    ].copy()


def project_xy_to_section_axis(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    theta_degrees: float,
) -> np.ndarray:
    theta_radians = np.deg2rad(theta_degrees)
    return np.asarray(x) * np.cos(theta_radians) + np.asarray(y) * np.sin(theta_radians)


def make_section_plot(df_plot: pd.DataFrame, theta_degrees: float, output_stem: Path) -> None:
    x_projected = project_xy_to_section_axis(df_plot["X"], df_plot["Y"], theta_degrees)
    z = df_plot["Z"].values
    labels = df_plot["NR"].astype(str).values
    infos = df_plot["INFO"].fillna("").astype(str).values

    fig, ax = plt.subplots(figsize=(70, 30), dpi=100)
    ax.scatter(x_projected, z, c=z, cmap="jet", marker="o")

    for xp, zp, point_label, info_label in zip(x_projected, z, labels, infos):
        ax.text(xp, zp, point_label, fontsize=12, ha="center", va="center")
        if info_label.strip():
            ax.text(xp, zp + 0.5, info_label, fontsize=10, ha="center", va="bottom")

    ax.set_box_aspect(0.24)
    ax.set_xticks(np.arange(np.floor(x_projected.min()), np.ceil(x_projected.max()) + 10, 10))
    ax.set_yticks(np.arange(np.floor(z.min()), np.ceil(z.max()) + 1, 1))
    ax.grid(True, linestyle="-", alpha=0.5)
    ax.set_xlabel(f"Projected X, theta = {theta_degrees:g} degrees")
    ax.set_ylabel("Z")
    ax.set_title("Projected stratigraphic section")

    fig.savefig(output_stem.with_suffix(".svg"), format="svg")
    fig.savefig(output_stem.with_suffix(".pdf"), format="pdf", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_xy_orientation_map(
    df: pd.DataFrame,
    df_plot: pd.DataFrame,
    theta_degrees: float,
    output_path: Path,
) -> None:
    x_all = df["X"].values
    y_all = df["Y"].values

    theta_radians = np.deg2rad(theta_degrees)
    projection_vector = np.array([np.cos(theta_radians), np.sin(theta_radians)])

    points_2d = np.vstack([x_all, y_all]).T
    projected_lengths = points_2d @ projection_vector
    line_start = projection_vector * np.min(projected_lengths)
    line_end = projection_vector * np.max(projected_lengths)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.scatter(x_all, y_all, marker="o", s=20, label="Points")
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        "--",
        linewidth=2,
        label="Projection orientation",
    )

    for _, row in df_plot.iterrows():
        ax.text(row["X"], row["Y"], str(row["NR"]), fontsize=8, ha="center", va="center")
        info_label = str(row["INFO"])
        if info_label.strip():
            ax.text(row["X"], row["Y"] + 0.2, info_label, fontsize=6, ha="center", va="bottom")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Plan-view XY map with section orientation")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def process_file(
    input_file: Path,
    theta_degrees: float,
    output_directory: Path | None = None,
) -> ProjectionResult:
    output_directory = output_directory or input_file.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    basename = input_file.stem

    df = read_point_file(input_file)
    projection_result = add_rotated_coordinates(df)
    normalized_df, adjusted_prefixes = vertically_normalize_groups(projection_result.dataframe)
    df_plot = filter_plot_points(normalized_df, adjusted_prefixes)

    excel_path = output_directory / f"{basename}_rotated.xlsx"
    normalized_df.to_excel(excel_path, index=False)

    if df_plot.empty:
        raise ValueError(f"No points remain for plotting. Check whether INFO labels contain '{VERTICAL_ZERO_MARKER}'.")

    make_section_plot(
        df_plot,
        theta_degrees=theta_degrees,
        output_stem=output_directory / f"{basename}_section_projection",
    )

    make_xy_orientation_map(
        normalized_df,
        df_plot,
        theta_degrees=theta_degrees,
        output_path=output_directory / f"{basename}_XY_map_orientation.pdf",
    )

    print(f"Rotated coordinates written to: {excel_path}")
    print(f"Section plot written to: {output_directory / f'{basename}_section_projection.pdf'}")
    print(f"XY orientation map written to: {output_directory / f'{basename}_XY_map_orientation.pdf'}")

    return ProjectionResult(normalized_df, projection_result.centroid_enu, projection_result.rotation_matrix)


INPUT_FILE = Path("/Volumes/Extreme Pro/12. BC/1.txt")
THETA_DEGREES = 290.0
OUTPUT_DIRECTORY = None


def main() -> None:
    process_file(INPUT_FILE, theta_degrees=THETA_DEGREES, output_directory=OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
