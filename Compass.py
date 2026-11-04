"""Compass: difficulty-aware parameter prediction, exact MILP, and ordered Rescue.

Runtime dependencies: geopandas, matplotlib, networkx, numpy, pandas, scipy,
shapely>=2, joblib, scikit-learn, and xgboost for XGBoost model bundles.
Trained models, static labels, and optional prediction tables are caller inputs.
"""

from __future__ import annotations

import argparse
import heapq
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from shapely import set_precision
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

PROJECT_DIR = Path(__file__).resolve().parent
ZONE_OPTIMIZATION_DATA_ROOT = Path(os.environ.get("COMPASS_DATA_ROOT", str(PROJECT_DIR / "ZoneOptimization_CityData")))
ZONE_OPTIMIZATION_OUTPUT_ROOT = Path(os.environ.get("COMPASS_OUTPUT_ROOT", str(PROJECT_DIR / "Compass_Outputs")))
DEFAULT_CITY = "GA-Atlanta_1304000"
ATL_DATA_DIR = ZONE_OPTIMIZATION_DATA_ROOT / DEFAULT_CITY
ATL_GRAPH_JSON = ATL_DATA_DIR / f"{DEFAULT_CITY}_graph" / f"{DEFAULT_CITY}.json"
ATL_BLOCK_SOURCE = ATL_DATA_DIR / f"{DEFAULT_CITY}_school_info" / "current" / f"{DEFAULT_CITY}_with_school_info.shp"
ATL_PUBLIC_SCHOOLS_CSV = ATL_DATA_DIR / f"{DEFAULT_CITY}_schools" / "current" / f"{DEFAULT_CITY}_public_schools.csv"
DEFAULT_OUTPUT_DIR = ZONE_OPTIMIZATION_OUTPUT_ROOT / DEFAULT_CITY
QUALITY_METRICS = ("CCRPI", "ELA_prof_plus", "Math_prof_plus")
MIN_SHARED_BOUNDARY_METERS = 1e-6
POINT_BRIDGE_MIN_COMPONENT_AREA_M2 = 10_000.0
STUDENT_COMPONENT_VIRTUAL_BRIDGE_MIN_STUDENTS = 1e-9
EMPTY_UNIT_ASSIGNMENT_EPS = 1e-9
EMPTY_UNIT_OBJECTIVE_AREA_WEIGHT_FACTOR = 0.0
ETHNICITY_COUNT_FIELDS = {
    "asian": "enrol_e_A",
    "black": "enrol_e_B",
    "hispanic": "enrol_e_H",
    "white": "enrol_e_W",
    "native_american": "enrol_e_AM",
    "pacific_islander": "enrol_e_HP",
    "multiracial": "enrol_e_TR",
}
ATTENDANCE_COUNT_ALIASES = {
    "enrol_e_A": ("enrol_e_A", "stu_e_A"),
    "enrol_e_B": ("enrol_e_B", "stu_e_B"),
    "enrol_e_H": ("enrol_e_H", "stu_e_H"),
    "enrol_e_W": ("enrol_e_W", "stu_e_W"),
    "enrol_e_AM": ("enrol_e_AM", "stu_e_AM", "stu_e_N"),
    "enrol_e_HP": ("enrol_e_HP", "stu_e_HP", "stu_e_PC"),
    "enrol_e_TR": ("enrol_e_TR", "stu_e_TR", "stu_e_M"),
    "enrol_e_p": ("enrol_e_p", "stu_e_p"),
}
FOCAL_GROUP_LABELS = {
    "asian": "Asian",
    "black": "Black",
    "hispanic": "Hispanic",
    "white": "White",
    "native_american": "Native American",
    "pacific_islander": "Pacific Islander",
    "multiracial": "Multiracial",
}


def standardize_attendance_count_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for output_col, aliases in ATTENDANCE_COUNT_ALIASES.items():
        if output_col in gdf.columns and pd.to_numeric(gdf[output_col], errors="coerce").fillna(0.0).sum() > 0:
            continue
        for alias in aliases:
            if alias in gdf.columns:
                gdf[output_col] = pd.to_numeric(gdf[alias], errors="coerce").fillna(0.0)
                break
    return gdf


@dataclass(frozen=True)
class CityZonePaths:
    city: str
    city_lower: str
    data_dir: Path
    graph_json: Path
    block_source: Path
    school_csv: Path
    output_dir: Path


def available_city_codes(data_root: Path = ZONE_OPTIMIZATION_DATA_ROOT) -> list[str]:
    retained = retained_city_codes(data_root)
    if retained:
        return sorted(code for code in retained if city_data_complete(code, data_root)[0])
    if not data_root.exists():
        return []
    codes = []
    for path in data_root.iterdir():
        if path.is_dir() and not path.name.startswith("_") and city_data_complete(path.name, data_root)[0]:
            codes.append(path.name)
    return sorted(codes)


def retained_city_codes(data_root: Path = ZONE_OPTIMIZATION_DATA_ROOT) -> set[str]:
    city_list = data_root / "city_list.csv"
    if not city_list.exists():
        return set()
    df = pd.read_csv(city_list, dtype=str)
    if "dataset_code" not in df.columns:
        raise ValueError(f"Missing dataset_code column in {city_list}")
    return set(df["dataset_code"].dropna().str.strip()) - {""}


def expected_output_dir(city: str, output_root: Path = ZONE_OPTIMIZATION_OUTPUT_ROOT) -> Path:
    return output_root / city.strip()


def city_output_complete(city: str, output_root: Path = ZONE_OPTIMIZATION_OUTPUT_ROOT) -> bool:
    city_code = city.strip()
    city_lower = city_code.lower()
    output_dir = expected_output_dir(city_code, output_root)
    required = [
        output_dir / f"{city_lower}_zone_assignment.csv",
        output_dir / f"{city_lower}_zone_summary.csv",
        output_dir / f"{city_lower}_zone_diagnostics.json",
        output_dir / f"{city_lower}_zones.png",
    ]
    return all(path.exists() for path in required)


def city_data_subdir(data_dir: Path, city_code: str, suffix: str) -> Path:
    """Return the standardized city-data subdirectory, preserving city-code case."""
    candidates = [
        data_dir / f"{city_code}_{suffix}",
        data_dir / f"{city_code.lower()}_{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(data_dir.glob(f"*_{suffix}"))
    if len(matches) == 1:
        return matches[0]
    return candidates[0]


def find_one_standard_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[0] if len(matches) == 1 else None


def preferred_graph_json_path(data_dir: Path, city_code: str) -> Path:
    graph_dir = city_data_subdir(data_dir, city_code, "graph")
    exact = graph_dir / f"{city_code}.json"
    if exact.exists():
        return exact
    match = find_one_standard_file(graph_dir, "*.json")
    return match if match is not None else exact


def validate_output_dir(city: str, output_dir: Path, output_root: Path = ZONE_OPTIMIZATION_OUTPUT_ROOT) -> None:
    expected = expected_output_dir(city, output_root).resolve()
    actual = output_dir.resolve()
    if actual != expected:
        raise ValueError(
            f"Output directory for {city} must be {expected}, matching the retained "
            f"ZoneOptimization_CityData folder name. Got {actual}."
        )


def preferred_block_source_path(data_dir: Path, city_lower: str, city_code: str) -> Path:
    info_dir = city_data_subdir(data_dir, city_code, "school_info") / "current"
    geojson = info_dir / f"{city_code}_with_school_info.geojson"
    if geojson.exists():
        return geojson
    geojson_match = find_one_standard_file(info_dir, "*_with_school_info.geojson")
    if geojson_match is not None:
        return geojson_match
    shp = info_dir / f"{city_code}_with_school_info.shp"
    if shp.exists():
        return shp
    shp_match = find_one_standard_file(info_dir, "*_with_school_info.shp")
    return shp_match if shp_match is not None else shp




def inferred_attendance_zones_path(block_source: Path, city: str) -> Optional[Path]:
    """Find the attendance-zone geometry paired with a city block source."""
    city_code = city.strip()
    city_lower = city_code.lower()
    try:
        data_dir = block_source.parents[2]
    except IndexError:
        return None
    boundary_dir = city_data_subdir(data_dir, city_code, "attendance_zones")
    if not boundary_dir.exists():
        return None
    for suffix in (".geojson", ".shp"):
        candidate = boundary_dir / f"attendance_zones_{city_code}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(boundary_dir.glob("attendance_zones_*"))
    return matches[0] if matches else None


def load_real_attendance_zone_geometry(
    block_source: Path,
    city: str,
    target_crs: Any,
) -> tuple[Optional[gpd.GeoDataFrame], Optional[Path]]:
    """Load real multi-zone attendance geometry when a city has it.

    Citywide replacement files have one row and should not be treated as an ATL-
    style attendance-zone source.
    """
    attendance_path = inferred_attendance_zones_path(block_source, city)
    if attendance_path is None or not attendance_path.exists():
        return None, None
    zones = gpd.read_file(attendance_path)
    if zones.empty or len(zones) <= 1 or "geometry" not in zones:
        return None, attendance_path
    if zones.crs is None:
        zones = zones.set_crs("EPSG:4326", allow_override=True)
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty].copy()
    if zones.empty or len(zones) <= 1:
        return None, attendance_path
    zones["geometry"] = zones.geometry.buffer(0)
    return zones.to_crs(target_crs), attendance_path








def clean_topology_geometry(geom: Any) -> Any:
    try:
        return set_precision(make_valid(geom), 0.001).buffer(0)
    except Exception:
        try:
            return make_valid(geom).buffer(0)
        except Exception:
            return geom


def safe_union_geometries(geometries: Any) -> Any:
    """Union geometries with a topology-repair fallback for GEOS edge cases."""
    series = gpd.GeoSeries(geometries)
    series = series[[geom is not None and not geom.is_empty for geom in series]]
    if series.empty:
        return None
    try:
        return series.union_all()
    except Exception:
        cleaned = series.apply(clean_topology_geometry)
        cleaned = cleaned[[geom is not None and not geom.is_empty for geom in cleaned]]
        if cleaned.empty:
            return None
        try:
            return cleaned.union_all()
        except Exception:
            return unary_union(list(cleaned))


def safe_dissolve_by_zone(plot_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve zones in projected meters, repairing topology only on fallback."""
    source_crs = plot_df.crs
    projected = plot_df.to_crs(3857)
    rows = []
    for zone, group in projected.groupby("zone"):
        union_geom = safe_union_geometries(group.geometry)
        if union_geom is None or union_geom.is_empty:
            continue
        rows.append({"zone": int(zone), "geometry": union_geom})
    dissolved = gpd.GeoDataFrame(rows, geometry="geometry", crs=projected.crs)
    if source_crs is not None and not dissolved.empty:
        dissolved = dissolved.to_crs(source_crs)
    return dissolved


def exterior_boundary_lines(geom: Any) -> list[Any]:
    """Return only outer exterior rings, never interior holes."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom.exterior]
    if geom.geom_type == "MultiPolygon":
        return [part.exterior for part in geom.geoms if not part.is_empty]
    if geom.geom_type == "GeometryCollection":
        lines: list[Any] = []
        for part in geom.geoms:
            lines.extend(exterior_boundary_lines(part))
        return lines
    return []






def city_data_complete(city: str, data_root: Path = ZONE_OPTIMIZATION_DATA_ROOT) -> tuple[bool, list[str]]:
    city_code = city.strip()
    city_lower = city_code.lower()
    data_dir = data_root / city_code
    messages: list[str] = []
    required = [
        preferred_graph_json_path(data_dir, city_code),
        preferred_block_source_path(data_dir, city_lower, city_code),
    ]
    for path in required:
        if not path.exists():
            messages.append(f"missing {path.relative_to(data_root)}")
    schools_dir = city_data_subdir(data_dir, city_code, "schools") / "current"
    if len(sorted(schools_dir.glob("*_public_schools.csv"))) != 1:
        messages.append(f"expected exactly one school CSV under {schools_dir.relative_to(data_root)}")
    return not messages, messages


def require_one_standard_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one standardized {label} matching {directory / pattern}; found {len(matches)}."
        )
    return matches[0]


def resolve_city_paths(city: str, data_root: Path = ZONE_OPTIMIZATION_DATA_ROOT) -> CityZonePaths:
    city_code = city.strip()
    if not city_code:
        raise ValueError("city code cannot be blank")
    data_dir = data_root / city_code
    if not data_dir.exists():
        known = ", ".join(available_city_codes(data_root))
        raise FileNotFoundError(f"Missing data directory for city {city_code}: {data_dir}. Known cities: {known}")
    retained = retained_city_codes(data_root)
    if retained and city_code not in retained:
        raise ValueError(f"City {city_code} is not listed in {data_root / 'city_list.csv'}")
    complete, messages = city_data_complete(city_code, data_root)
    if not complete:
        raise ValueError(f"City {city_code} is not runnable because its data is incomplete: {'; '.join(messages)}")
    city_lower = city_code.lower()
    schools_dir = city_data_subdir(data_dir, city_code, "schools") / "current"
    block_source = preferred_block_source_path(data_dir, city_lower, city_code)
    return CityZonePaths(
        city=city_code,
        city_lower=city_lower,
        data_dir=data_dir,
        graph_json=preferred_graph_json_path(data_dir, city_code),
        block_source=block_source,
        school_csv=require_one_standard_file(schools_dir, "*_public_schools.csv", "school CSV"),
        output_dir=ZONE_OPTIMIZATION_OUTPUT_ROOT / city_code,
    )


@dataclass
class ExactZoneConfig:
    n_zones: int = 6
    balance_tolerance: float = 0.55
    min_schools_per_zone: Optional[int] = None
    max_schools_per_zone: Optional[int] = None
    cover_distance_miles: Optional[float] = 8.0
    frl_bounds: Optional[tuple[float, float]] = (0.45, 1.65)
    focal_group_bounds: Optional[tuple[float, float]] = (0.60, 1.30)
    diversity_index_bounds: Optional[tuple[float, float]] = (0.45, 1.45)
    school_quality_bounds: Optional[tuple[float, float]] = (0.45, 1.80)
    max_shortage: Optional[float] = None
    objective_distance_weight: float = 0.10
    objective_distance_power: float = 2.0
    objective_diversity_weight: float = 75.0
    objective_quality_weight: float = 1.0
    objective_min_surplus_weight: float = 2.0
    solver_time_limit_seconds: Optional[int] = 30
    min_enrollment_for_unit: float = 0.0


@dataclass(frozen=True)
class ScoreWeights:
    student_cv: float = 1.0
    school_dev: float = 0.5
    frl_spread: float = 5.0
    focal_group_spread: float = 5.0
    diversity_spread: float = 30.0
    quality_spread: float = 0.5
    shortage: float = 1.5
    distance: float = 1.0
    disconnected: float = 20.0


@dataclass(frozen=True)
class PolicyWeights:
    objective_quality_weight: float = 1.0
    objective_distance_weight: float = 0.10
    objective_distance_power: float = 2.0
    objective_diversity_weight: float = 75.0
    objective_min_surplus_weight: float = 2.0
    score_weights: ScoreWeights = ScoreWeights()


@dataclass(frozen=True)
class AnnealingConfig:
    iterations: int = 40
    initial_temperature: float = 1.0
    cooling_rate: float = 0.92
    seed: int = 20260601
    min_zones: Optional[int] = None
    max_zones: Optional[int] = None
    balance_tolerance_min: float = 0.35
    balance_tolerance_max: float = 0.75
    cover_distance_factor_min: float = 0.80
    cover_distance_factor_max: float = 4.00
    max_shortage_fraction_min: float = 0.0
    max_shortage_fraction_max: float = 0.12
    school_slack_below_min: int = 1
    school_slack_below_max: int = 4
    school_slack_above_min: int = 1
    school_slack_above_max: int = 4
    frl_lower_min: float = 0.30
    frl_lower_max: float = 0.80
    frl_upper_min: float = 1.20
    frl_upper_max: float = 2.20
    focal_group_lower_min: float = 0.40
    focal_group_lower_max: float = 0.85
    focal_group_upper_min: float = 1.10
    focal_group_upper_max: float = 1.80
    diversity_index_lower_min: float = 0.35
    diversity_index_lower_max: float = 0.85
    diversity_index_upper_min: float = 1.15
    diversity_index_upper_max: float = 1.90
    school_quality_lower_min: float = 0.35
    school_quality_lower_max: float = 0.85
    school_quality_upper_min: float = 1.20
    school_quality_upper_max: float = 2.20






def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _round_nonnegative_preserve_total(values: pd.Series, target_total: Optional[int] = None) -> pd.Series:
    """Round nonnegative counts to integers while preserving the rounded total."""
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    floors = np.floor(arr).astype(int)
    target = int(round(float(arr.sum()))) if target_total is None else int(target_total)
    deficit = target - int(floors.sum())
    if deficit > 0:
        order = np.argsort(-(arr - floors))
        floors[order[:deficit]] += 1
    elif deficit < 0:
        order = np.argsort(arr - floors)
        removable = [idx for idx in order if floors[idx] > 0]
        for idx in removable[: abs(deficit)]:
            floors[idx] -= 1
    return pd.Series(floors.astype(int), index=values.index)


def _minmax(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    span = float(numeric.max() - numeric.min())
    if not np.isfinite(span) or span <= 0:
        return pd.Series(np.ones(len(numeric)), index=numeric.index)
    return (numeric - float(numeric.min())) / span


def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    if len(numeric) == 0 or float(w.sum()) <= 0:
        return float(np.nan)
    order = np.argsort(numeric)
    numeric = numeric[order]
    w = w[order]
    cumulative = np.cumsum(w)
    cutoff = float(quantile) * float(w.sum())
    return float(numeric[min(int(np.searchsorted(cumulative, cutoff, side="left")), len(numeric) - 1)])


def assign_city_specific_focal_groups(
    gdf: gpd.GeoDataFrame,
    min_student_share: float = 0.03,
    high_poverty_quantile: float = 0.75,
    low_quality_quantile: float = 0.25,
    exposure_ratio_min: float = 1.10,
    quality_gap_min: float = 0.05,
) -> gpd.GeoDataFrame:
    """Identify all groups exposed to higher-poverty or lower-quality schools."""
    out = gdf.copy()
    total_students = float(out["students"].sum())
    if total_students <= 0:
        out["focal_group_count"] = 0.0
        out["focal_group_pct"] = 0.0
        out["diversity_index"] = out["FRL_pct"].fillna(0.0).clip(0.0, 1.0)
        out.attrs.update(gdf.attrs)
        out.attrs["focal_group_definition"] = []
        out.attrs["focal_group_diagnostics"] = []
        return out

    high_poverty_cutoff = weighted_quantile(out["FRL_pct"], out["students"], high_poverty_quantile)
    low_quality_cutoff = weighted_quantile(out["school_quality_index"], out["students"], low_quality_quantile)
    high_poverty_mask = out["FRL_pct"] >= high_poverty_cutoff
    low_quality_mask = out["school_quality_index"] <= low_quality_cutoff
    district_high_poverty_exposure = float(out.loc[high_poverty_mask, "students"].sum() / total_students)
    district_low_quality_exposure = float(out.loc[low_quality_mask, "students"].sum() / total_students)
    district_quality = float(np.average(out["school_quality_index"], weights=np.maximum(out["students"], 1e-9)))

    diagnostics = []
    focal_groups = []
    eligible_groups = []
    for group_name, count_col in ETHNICITY_COUNT_FIELDS.items():
        if count_col not in out.columns:
            continue
        counts = pd.to_numeric(out[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        group_total = float(counts.sum())
        group_share = group_total / total_students if total_students else 0.0
        if group_total <= 0:
            continue
        high_poverty_exposure = float(counts[high_poverty_mask].sum() / group_total)
        low_quality_exposure = float(counts[low_quality_mask].sum() / group_total)
        high_poverty_ratio = high_poverty_exposure / max(district_high_poverty_exposure, 1e-9)
        low_quality_ratio = low_quality_exposure / max(district_low_quality_exposure, 1e-9)
        group_quality = float(np.average(out["school_quality_index"], weights=np.maximum(counts, 1e-9)))
        quality_gap = district_quality - group_quality
        sufficiently_represented = group_share >= float(min_student_share)
        if sufficiently_represented:
            eligible_groups.append(group_name)
        underserved = (
            high_poverty_ratio >= float(exposure_ratio_min)
            or low_quality_ratio >= float(exposure_ratio_min)
            or quality_gap >= float(quality_gap_min)
        )
        selected = bool(sufficiently_represented and underserved)
        diagnostics.append(
            {
                "group": group_name,
                "label": FOCAL_GROUP_LABELS.get(group_name, group_name),
                "student_share": group_share,
                "high_poverty_exposure": high_poverty_exposure,
                "high_poverty_exposure_ratio": high_poverty_ratio,
                "low_quality_exposure": low_quality_exposure,
                "low_quality_exposure_ratio": low_quality_ratio,
                "average_school_quality": group_quality,
                "district_average_school_quality": district_quality,
                "quality_gap": quality_gap,
                "sufficiently_represented": bool(sufficiently_represented),
                "underserved": bool(underserved),
                "selected": selected,
            }
        )
        if selected:
            focal_groups.append(group_name)

    advantaged_groups = [group for group in eligible_groups if group not in focal_groups]
    focal_cols = [ETHNICITY_COUNT_FIELDS[group] for group in focal_groups if group in ETHNICITY_COUNT_FIELDS]
    out["focal_group_count"] = out[focal_cols].sum(axis=1) if focal_cols else 0.0
    out["focal_group_pct"] = (out["focal_group_count"] / out["students"].replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)
    out["diversity_index"] = 0.5 * (out["FRL_pct"] + out["focal_group_pct"])
    out.attrs.update(gdf.attrs)
    out.attrs["focal_group_definition"] = focal_groups
    out.attrs["focal_group_labels"] = [FOCAL_GROUP_LABELS.get(group, group) for group in focal_groups]
    out.attrs["advantaged_group_definition"] = advantaged_groups
    out.attrs["advantaged_group_labels"] = [FOCAL_GROUP_LABELS.get(group, group) for group in advantaged_groups]
    out.attrs["focal_group_diagnostics"] = diagnostics
    out.attrs["focal_group_thresholds"] = {
        "min_student_share": float(min_student_share),
        "high_poverty_quantile": float(high_poverty_quantile),
        "low_quality_quantile": float(low_quality_quantile),
        "exposure_ratio_min": float(exposure_ratio_min),
        "quality_gap_min": float(quality_gap_min),
        "high_poverty_cutoff": high_poverty_cutoff,
        "low_quality_cutoff": low_quality_cutoff,
    }
    return out


def load_elementary_school_quality(path: Path = ATL_PUBLIC_SCHOOLS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing public school quality file: {path}")
    schools = pd.read_csv(path, low_memory=False)

    # Legacy ATL/Georgia files use Type; generated national CCD files are already
    # elementary-serving and use SCHOOL_TYPE_TEXT plus grade-span fields.
    if "Type" not in schools.columns and not {"SCHOOL_TYPE_TEXT", "SCH_NAME", "NCESSCH"}.intersection(schools.columns):
        maybe = pd.read_csv(path, skiprows=1, low_memory=False)
        if "Type" in maybe.columns or {"SCHOOL_TYPE_TEXT", "SCH_NAME", "NCESSCH"}.intersection(maybe.columns):
            schools = maybe

    elementary_mask = pd.Series(False, index=schools.index)
    if "Type" in schools.columns:
        elementary_mask |= schools["Type"].astype(str).str.contains("Elementary|K12", case=False, na=False)
    if "SCHOOL_LEVEL" in schools.columns:
        elementary_mask |= schools["SCHOOL_LEVEL"].astype(str).str.contains("Elementary", case=False, na=False)
    if {"GSLO", "GSHI"}.issubset(schools.columns):
        elementary_grades = {"PK", "KG", "K", "01", "02", "03", "04", "05", "1", "2", "3", "4", "5"}
        gslo = schools["GSLO"].astype(str).str.strip().str.upper()
        gshi = schools["GSHI"].astype(str).str.strip().str.upper()
        elementary_mask |= gslo.isin(elementary_grades) | gshi.isin(elementary_grades)
    if "SCH_NAME" in schools.columns:
        elementary_mask |= schools["SCH_NAME"].astype(str).str.contains(r"\bElementary\b", case=False, na=False)
    schools = schools[elementary_mask].copy()

    if schools.empty:
        raise ValueError(f"No elementary schools found in {path}")

    if "longitude" not in schools.columns and "LONCOD" in schools.columns:
        schools["longitude"] = schools["LONCOD"]
    if "latitude" not in schools.columns and "LATCOD" in schools.columns:
        schools["latitude"] = schools["LATCOD"]

    numeric_cols = [
        "SchoolID",
        "ENROLLMENT_COUNT",
        "TOTAL_ENROLLMENT_COUNT",
        "enrollment",
        "MEMBER",
        "TOTAL",
        "CCRPI",
        "quality",
        "pProfPlusELA3",
        "pProfPlusELA5",
        "pProfPlusMath3",
        "pProfPlusMath5",
        "read_prof",
        "math_prof",
        "longitude",
        "latitude",
    ]
    for col in numeric_cols:
        if col not in schools.columns:
            schools[col] = np.nan
        schools[col] = pd.to_numeric(schools[col], errors="coerce")
    if "SchoolID" in schools.columns and schools["SchoolID"].notna().any():
        schools["school_unique_id"] = schools["SchoolID"].astype("Int64").astype(str)
    elif "NCESSCH" in schools.columns and schools["NCESSCH"].notna().any():
        schools["school_unique_id"] = schools["NCESSCH"].astype(str).str.replace(r"\.0$", "", regex=True)
    elif "SCH_NAME" in schools.columns and schools["SCH_NAME"].notna().any():
        schools["school_unique_id"] = schools["SCH_NAME"].astype(str)
    else:
        schools["school_unique_id"] = schools.index.astype(str)

    capacity_candidates = [
        "TOTAL_ENROLLMENT_COUNT",
        "ENROLLMENT_COUNT",
        "enrollment",
        "MEMBER",
        "TOTAL",
    ]
    capacity_source = pd.Series(0.0, index=schools.index)
    for col in capacity_candidates:
        if col in schools.columns and schools[col].notna().any() and float(schools[col].fillna(0).sum()) > 0:
            capacity_source = schools[col]
            break
    schools["school_capacity"] = capacity_source.fillna(0.0).clip(lower=0.0)

    if schools["CCRPI"].isna().all() and schools["quality"].notna().any():
        schools["CCRPI"] = schools["quality"]
    schools["ELA_prof_plus"] = schools[["pProfPlusELA3", "pProfPlusELA5", "read_prof"]].mean(axis=1, skipna=True)
    schools["Math_prof_plus"] = schools[["pProfPlusMath3", "pProfPlusMath5", "math_prof"]].mean(axis=1, skipna=True)
    for metric in ["ELA_prof_plus", "Math_prof_plus"]:
        if schools[metric].isna().all():
            schools[metric] = schools["CCRPI"]
    weights = schools["school_capacity"].replace(0, np.nan)
    for metric in QUALITY_METRICS:
        valid = schools[metric].notna()
        if valid.any() and weights[valid].notna().any():
            fill_value = float(np.average(schools.loc[valid, metric], weights=weights[valid].fillna(1.0)))
        elif valid.any():
            fill_value = float(schools.loc[valid, metric].mean())
        else:
            fill_value = 0.0
        schools[metric] = schools[metric].fillna(fill_value)
    schools["school_quality_index"] = (
        _minmax(schools["CCRPI"]) + _minmax(schools["ELA_prof_plus"]) + _minmax(schools["Math_prof_plus"])
    ) / 3.0
    return schools

def school_points_from_coordinates(schools: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build school points from either lon/lat degrees or EPSG:3857 meter coordinates."""
    lon = pd.to_numeric(schools["longitude"], errors="coerce")
    lat = pd.to_numeric(schools["latitude"], errors="coerce")
    valid = lon.notna() & lat.notna()
    if not valid.any():
        return gpd.GeoDataFrame(schools.iloc[0:0].copy(), geometry=[], crs="EPSG:4326")

    point_rows = schools.loc[valid].copy()
    lon = lon.loc[valid]
    lat = lat.loc[valid]
    source_crs = "EPSG:3857" if lon.abs().max() > 180 or lat.abs().max() > 90 else "EPSG:4326"
    return gpd.GeoDataFrame(point_rows, geometry=gpd.points_from_xy(lon, lat), crs=source_crs)


def attach_school_quality(
    gdf: gpd.GeoDataFrame,
    school_csv: Path = ATL_PUBLIC_SCHOOLS_CSV,
) -> gpd.GeoDataFrame:
    """Assign elementary schools to units and replace school counts/capacity with point-based values."""
    schools = load_elementary_school_quality(school_csv)
    school_points = school_points_from_coordinates(schools)
    school_points = school_points[school_points.geometry.notna()].to_crs(gdf.crs)
    if school_points.empty:
        raise ValueError(f"No usable elementary school coordinates in {school_csv}")

    school_points_projected = school_points.to_crs(3857)
    units_projected = gdf[["unit_id", "geometry"]].to_crs(3857)
    if "source_idx_u" in gdf.columns:
        units_projected["source_idx_u"] = pd.to_numeric(gdf["source_idx_u"], errors="coerce").astype("Int64")

    if "idx_u" in school_points_projected.columns and "source_idx_u" in units_projected.columns:
        assigned_rows = []
        units_by_source = {
            int(source_idx): group.copy()
            for source_idx, group in units_projected.dropna(subset=["source_idx_u"]).groupby("source_idx_u", sort=False)
        }
        for _, school_row in school_points_projected.iterrows():
            raw_idx = pd.to_numeric(pd.Series([school_row.get("idx_u")]), errors="coerce").iloc[0]
            candidates = units_by_source.get(int(raw_idx), None) if pd.notna(raw_idx) else None
            if candidates is None or candidates.empty:
                assigned_rows.append(None)
                continue
            distances = candidates.geometry.distance(school_row.geometry)
            nearest_pos = distances.idxmin()
            out_row = school_row.copy()
            out_row["unit_id"] = int(candidates.loc[nearest_pos, "unit_id"])
            out_row["nearest_unit_distance_m"] = float(distances.loc[nearest_pos])
            assigned_rows.append(out_row)
        matched = [row for row in assigned_rows if row is not None]
        unmatched = [idx for idx, row in enumerate(assigned_rows) if row is None]
        joined = gpd.GeoDataFrame(matched, geometry="geometry", crs=school_points_projected.crs)
        if unmatched:
            fallback = gpd.sjoin_nearest(
                school_points_projected.iloc[unmatched],
                units_projected[["unit_id", "geometry"]],
                how="left",
                distance_col="nearest_unit_distance_m",
            )
            fallback = fallback.drop_duplicates(subset=["school_unique_id"], keep="first")
            joined = pd.concat([joined, fallback], ignore_index=True)
            joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=school_points_projected.crs)
    else:
        joined = gpd.sjoin_nearest(
            school_points_projected,
            units_projected[["unit_id", "geometry"]],
            how="left",
            distance_col="nearest_unit_distance_m",
        )
        joined = joined.drop_duplicates(subset=["school_unique_id"], keep="first")
    if joined["unit_id"].isna().any():
        missing = int(joined["unit_id"].isna().sum())
        raise ValueError(f"{missing} elementary schools could not be assigned to an optimization unit")
    joined["unit_id"] = joined["unit_id"].astype(int)
    joined["school_capacity"] = _round_nonnegative_preserve_total(joined["school_capacity"], int(round(schools["school_capacity"].sum())))
    for metric in QUALITY_METRICS:
        joined[f"{metric}_weighted_sum"] = joined[metric] * joined["school_capacity"]

    agg_spec = {
        "quality_school_count": ("school_unique_id", "count"),
        "school_quality_weight": ("school_capacity", "sum"),
        "school_quality_index_sum": ("school_quality_index", "sum"),
        "nearest_school_distance_m": ("nearest_unit_distance_m", "max"),
    }
    for metric in QUALITY_METRICS:
        agg_spec[f"{metric}_weighted_sum"] = (f"{metric}_weighted_sum", "sum")
    school_agg = joined.groupby("unit_id").agg(**agg_spec)

    out = gdf.merge(school_agg, left_on="unit_id", right_index=True, how="left")
    out.attrs.update(gdf.attrs)
    fill_zero_cols = [
        "school_quality_weight",
        "quality_school_count",
        "school_quality_index_sum",
        "nearest_school_distance_m",
        *[f"{metric}_weighted_sum" for metric in QUALITY_METRICS],
    ]
    for col in fill_zero_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    weight = out["school_quality_weight"].replace(0, np.nan)
    for metric in QUALITY_METRICS:
        out[metric] = (out[f"{metric}_weighted_sum"] / weight).fillna(0.0)
    out["school_quality_index"] = (
        out["school_quality_index_sum"] / out["quality_school_count"].replace(0, np.nan)
    ).fillna(0.0)
    out["source_ele_school_count"] = out["quality_school_count"].astype(float)
    out["school_count"] = out["quality_school_count"].astype(float)
    out["block_ele_cap"] = out["school_quality_weight"].astype(float)
    out.attrs["elementary_school_count"] = int(joined["school_unique_id"].nunique())
    out.attrs["elementary_quality_capacity_total"] = int(joined["school_capacity"].sum())
    out.attrs["school_count_capacity_source"] = (
        "public_school_points_nearest_attendance_zone_unit; capacity=TOTAL_ENROLLMENT_COUNT if available else ENROLLMENT_COUNT/MEMBER/TOTAL"
    )
    return out


def load_attendance_zone_units(
    block_source: Path,
    school_csv: Path,
    city: str,
    focal_min_student_share: float,
    focal_high_poverty_quantile: float,
    focal_low_quality_quantile: float,
    focal_exposure_ratio_min: float,
    focal_quality_gap_min: float,
) -> gpd.GeoDataFrame:
    """Load ATL-style elementary attendance-zone polygons as optimization units."""
    zones, attendance_path = load_real_attendance_zone_geometry(block_source, city, "EPSG:4326")
    if zones is None or attendance_path is None:
        raise FileNotFoundError(f"No real multi-zone attendance boundary found for {city}")
    gdf = zones.copy().reset_index(drop=True)
    if "idx_u" in gdf.columns:
        source_idx = pd.to_numeric(gdf["idx_u"], errors="coerce")
        gdf["source_idx_u"] = source_idx.fillna(pd.Series(gdf.index, index=gdf.index)).astype(int)
    else:
        gdf["source_idx_u"] = gdf.index.astype(int)
    gdf["unit_id"] = gdf.index.astype(int)
    gdf["parent_unit_id"] = gdf["unit_id"]
    gdf["unit_part"] = 0
    gdf["idx_u"] = gdf["unit_id"]
    if "unit_name" not in gdf.columns:
        if "schnam" in gdf.columns:
            gdf["unit_name"] = gdf["schnam"]
        elif "SrcName" in gdf.columns:
            gdf["unit_name"] = gdf["SrcName"]
        else:
            gdf["unit_name"] = gdf["unit_id"].astype(str)
    gdf["unit_name"] = gdf["unit_name"].fillna(gdf["unit_id"].astype(str)).astype(str)
    if "SrcName" not in gdf.columns:
        gdf["SrcName"] = gdf["unit_name"]
    if "schnam" not in gdf.columns:
        gdf["schnam"] = gdf["unit_name"]

    gdf = standardize_attendance_count_columns(gdf)
    numeric_cols = [
        "enrol_e",
        "enrol_e_A",
        "enrol_e_B",
        "enrol_e_H",
        "enrol_e_W",
        "enrol_e_AM",
        "enrol_e_HP",
        "enrol_e_TR",
        "enrol_e_p",
        "ele_school",
        "ele_cap",
        "kidsPOP",
    ]
    for col in numeric_cols:
        if col not in gdf.columns:
            gdf[col] = 0.0
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    gdf["students"] = gdf["enrol_e"].clip(lower=0.0)
    gdf["source_ele_school_count"] = gdf["ele_school"].clip(lower=0.0)
    gdf["school_count"] = gdf["ele_school"].clip(lower=0.0)
    gdf["block_ele_cap"] = gdf["ele_cap"].clip(lower=0.0)
    denom = gdf["students"].replace(0, np.nan)
    gdf["FRL_pct"] = (gdf["enrol_e_p"] / denom).fillna(0.0).clip(0.0, 1.0)
    gdf["asian_pct"] = (gdf["enrol_e_A"] / denom).fillna(0.0).clip(0.0, 1.0)
    gdf["white_pct"] = (gdf["enrol_e_W"] / denom).fillna(0.0).clip(0.0, 1.0)
    gdf["native_american_pct"] = (gdf["enrol_e_AM"] / denom).fillna(0.0).clip(0.0, 1.0)
    gdf["pacific_islander_pct"] = (gdf["enrol_e_HP"] / denom).fillna(0.0).clip(0.0, 1.0)
    gdf["multiracial_pct"] = (gdf["enrol_e_TR"] / denom).fillna(0.0).clip(0.0, 1.0)
    gdf["geometry"] = gdf.geometry.buffer(0)

    multipart_before = int(gdf.geometry.geom_type.eq("MultiPolygon").sum())
    if multipart_before:
        gdf = explode_multipart_units(gdf)
        gdf.attrs["exploded_multipart_unit_count"] = multipart_before
        gdf.attrs["exploded_unit_count_after"] = int(len(gdf))
    else:
        gdf.attrs["exploded_multipart_unit_count"] = 0
        gdf.attrs["exploded_unit_count_after"] = int(len(gdf))

    gdf = attach_school_quality(gdf, school_csv)
    empty_unit_mask = (
        (gdf["students"] <= 1e-9)
        & (gdf["school_count"] <= 1e-9)
        & (gdf["block_ele_cap"] <= 1e-9)
    )
    gdf.attrs["empty_optimization_units_retained"] = int(empty_unit_mask.sum())
    gdf.attrs["dropped_empty_optimization_units"] = 0
    gdf = assign_city_specific_focal_groups(
        gdf,
        min_student_share=focal_min_student_share,
        high_poverty_quantile=focal_high_poverty_quantile,
        low_quality_quantile=focal_low_quality_quantile,
        exposure_ratio_min=focal_exposure_ratio_min,
        quality_gap_min=focal_quality_gap_min,
    )
    gdf = drop_tiny_zero_student_school_sliver_components(gdf, build_adjacency_graph(gdf))
    gdf.attrs["dropped_no_school_component_count"] = 0
    gdf.attrs["dropped_no_school_component_units"] = []
    gdf.attrs["dropped_no_school_component_students"] = 0.0
    gdf.attrs["dropped_no_school_component_capacity"] = 0.0
    gdf.attrs["city"] = city.strip()
    gdf.attrs["block_source"] = str(block_source)
    gdf.attrs["attendance_zone_source"] = str(attendance_path)
    gdf.attrs["unit_source"] = "attendance_zone_polygon"
    gdf.attrs["school_count_capacity_source"] = "official_school_points_joined_to_attendance_zone_polygon"
    gdf.attrs["zoning_total_students"] = int(round(float(gdf["students"].sum())))
    gdf.attrs["zoning_total_capacity"] = int(round(float(gdf["block_ele_cap"].sum())))
    gdf.attrs["zoning_total_schools"] = int(round(float(gdf["school_count"].sum())))
    return gdf


def load_optimization_units(
    block_source: Path = ATL_BLOCK_SOURCE,
    school_csv: Path = ATL_PUBLIC_SCHOOLS_CSV,
    graph_json: Path = ATL_GRAPH_JSON,
    city: str = DEFAULT_CITY,
    focal_min_student_share: float = 0.03,
    focal_high_poverty_quantile: float = 0.75,
    focal_low_quality_quantile: float = 0.25,
    focal_exposure_ratio_min: float = 1.10,
    focal_quality_gap_min: float = 0.05,
) -> gpd.GeoDataFrame:
    """Load ATL-style attendance-zone polygons as the only MILP units.

    The block source remains an auxiliary path used to resolve the matching
    city data folder and block overlay for output maps. It is not used as the
    optimization unit surface.
    """
    return load_attendance_zone_units(
        block_source=block_source,
        school_csv=school_csv,
        city=city,
        focal_min_student_share=focal_min_student_share,
        focal_high_poverty_quantile=focal_high_poverty_quantile,
        focal_low_quality_quantile=focal_low_quality_quantile,
        focal_exposure_ratio_min=focal_exposure_ratio_min,
        focal_quality_gap_min=focal_quality_gap_min,
    )


def explode_multipart_units(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Split multipart attendance geometries so visual contiguity is enforceable."""
    additive_cols = [
        "students",
        "school_count",
        "source_ele_school_count",
        "enrol_e_A",
        "enrol_e_B",
        "enrol_e_H",
        "enrol_e_W",
        "enrol_e_AM",
        "enrol_e_HP",
        "enrol_e_TR",
        "enrol_e_p",
        "block_ele_cap",
    ]
    present_additive = [col for col in additive_cols if col in gdf.columns]
    projected = gdf.to_crs(3857)
    rows = []
    for original_pos, row in gdf.iterrows():
        geom_projected = projected.geometry.iloc[original_pos]
        parts = list(row.geometry.geoms) if row.geometry.geom_type == "MultiPolygon" else [row.geometry]
        projected_parts = list(geom_projected.geoms) if geom_projected.geom_type == "MultiPolygon" else [geom_projected]
        areas = np.array([max(part.area, 0.0) for part in projected_parts], dtype=float)
        shares = areas / areas.sum() if areas.sum() > 0 else np.full(len(parts), 1.0 / len(parts))
        for part_idx, (part, share) in enumerate(zip(parts, shares)):
            new_row = row.copy()
            new_row["parent_unit_id"] = int(row["unit_id"])
            new_row["unit_part"] = int(part_idx)
            if len(parts) > 1:
                new_row["unit_name"] = f"{row['unit_name']} part {part_idx + 1}"
            for col in present_additive:
                new_row[col] = float(row[col]) * float(share)
            new_row.geometry = part
            rows.append(new_row)
    exploded = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs).reset_index(drop=True)
    exploded["unit_id"] = exploded.index.astype(int)
    exploded["idx_u"] = exploded["unit_id"]
    for count_col in ["students", "school_count", "source_ele_school_count", "block_ele_cap"]:
        if count_col in exploded.columns:
            exploded[count_col] = _round_nonnegative_preserve_total(exploded[count_col])
    denom = exploded["students"].replace(0, np.nan)
    exploded["FRL_pct"] = (exploded["enrol_e_p"] / denom).fillna(0.0).clip(0.0, 1.0)
    exploded["asian_pct"] = (exploded["enrol_e_A"] / denom).fillna(0.0).clip(0.0, 1.0)
    exploded["white_pct"] = (exploded["enrol_e_W"] / denom).fillna(0.0).clip(0.0, 1.0)
    exploded["native_american_pct"] = (exploded["enrol_e_AM"] / denom).fillna(0.0).clip(0.0, 1.0)
    exploded["pacific_islander_pct"] = (exploded["enrol_e_HP"] / denom).fillna(0.0).clip(0.0, 1.0)
    exploded["multiracial_pct"] = (exploded["enrol_e_TR"] / denom).fillna(0.0).clip(0.0, 1.0)
    exploded.attrs.update(gdf.attrs)
    return exploded


def build_adjacency_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    graph = nx.Graph()
    for unit_id in gdf["unit_id"].astype(int):
        graph.add_node(int(unit_id))

    projected = gdf.to_crs(gdf.estimate_utm_crs() or 3857).copy()
    projected["geometry"] = projected.geometry.map(clean_topology_geometry)
    projected = projected[projected.geometry.notna() & ~projected.geometry.is_empty].copy()
    if projected.empty:
        graph.graph["adjacency_source"] = "shared_boundary_with_structural_point_bridges"
        graph.graph["min_shared_boundary_meters"] = MIN_SHARED_BOUNDARY_METERS
        graph.graph["connectivity_distance_meters"] = 0.0
        graph.graph["component_bridge_edges"] = []
        graph.graph["rejected_point_contact_edge_count"] = 0
        return graph

    projected["unit_id"] = projected["unit_id"].astype(int)
    projected["_area_m2"] = projected.geometry.area
    projected["_students"] = (
        pd.to_numeric(projected.get("students", 0.0), errors="coerce").fillna(0.0)
        if "students" in projected.columns
        else 0.0
    )
    area_by_unit = projected.set_index("unit_id")["_area_m2"].to_dict()
    students_by_unit = projected.set_index("unit_id")["_students"].to_dict()

    spatial_index = projected.sindex
    projected_geoms = projected.geometry
    point_contact_candidates: list[tuple[int, int]] = []
    rejected_point_contacts = 0

    for i, geom in enumerate(projected_geoms):
        try:
            candidates = spatial_index.query(geom, predicate="intersects")
        except Exception:
            candidates = spatial_index.query(geom)
        for j in candidates:
            j = int(j)
            if j <= i:
                continue
            other = projected_geoms.iloc[j]
            u = int(projected.at[projected.index[i], "unit_id"])
            v = int(projected.at[projected.index[j], "unit_id"])
            try:
                if not geom.intersects(other):
                    continue
                shared_boundary_m = float(geom.boundary.intersection(other.boundary).length)
            except Exception:
                clean_geom = clean_topology_geometry(geom)
                clean_other = clean_topology_geometry(other)
                try:
                    if not clean_geom.intersects(clean_other):
                        continue
                    shared_boundary_m = float(clean_geom.boundary.intersection(clean_other.boundary).length)
                except Exception:
                    shared_boundary_m = 0.0

            if shared_boundary_m > MIN_SHARED_BOUNDARY_METERS:
                graph.add_edge(u, v, adjacency_type="shared_boundary", shared_boundary_m=shared_boundary_m)
            else:
                point_contact_candidates.append((u, v))

    def component_stats(component: set[int]) -> tuple[float, float]:
        students = sum(float(students_by_unit.get(unit, 0.0)) for unit in component)
        area = sum(float(area_by_unit.get(unit, 0.0)) for unit in component)
        return students, area

    point_bridge_edges = []
    for u, v in point_contact_candidates:
        try:
            if nx.has_path(graph, u, v):
                rejected_point_contacts += 1
                continue
        except nx.NetworkXError:
            rejected_point_contacts += 1
            continue
        component_u = {int(unit) for unit in nx.node_connected_component(graph, u)}
        component_v = {int(unit) for unit in nx.node_connected_component(graph, v)}
        students_u, area_u = component_stats(component_u)
        students_v, area_v = component_stats(component_v)
        is_structural_bridge = (
            students_u > 0.0
            and students_v > 0.0
            and area_u >= POINT_BRIDGE_MIN_COMPONENT_AREA_M2
            and area_v >= POINT_BRIDGE_MIN_COMPONENT_AREA_M2
        )
        if is_structural_bridge:
            graph.add_edge(u, v, adjacency_type="structural_point_bridge", shared_boundary_m=0.0)
            point_bridge_edges.append(
                {
                    "source": int(u),
                    "target": int(v),
                    "source_component_students": float(students_u),
                    "target_component_students": float(students_v),
                    "source_component_area_m2": float(area_u),
                    "target_component_area_m2": float(area_v),
                }
            )
        else:
            rejected_point_contacts += 1

    original_components = [set(map(int, component)) for component in nx.connected_components(graph)]
    student_component_bridge_edges = []
    if len(original_components) > 1:
        def component_area(component: set[int]) -> float:
            return sum(float(area_by_unit.get(unit, 0.0)) for unit in component)

        main_component = max(original_components, key=lambda component: (component_area(component), len(component)))
        projected_by_unit = projected.set_index("unit_id")
        for component in original_components:
            if component == main_component:
                continue
            students, area = component_stats(component)
            if students <= STUDENT_COMPONENT_VIRTUAL_BRIDGE_MIN_STUDENTS:
                continue
            best_pair: Optional[tuple[float, int, int]] = None
            for u in component:
                geom_u = projected_by_unit.at[u, "geometry"]
                for v in main_component:
                    geom_v = projected_by_unit.at[v, "geometry"]
                    try:
                        distance_m = float(geom_u.distance(geom_v))
                    except Exception:
                        distance_m = float(geom_u.centroid.distance(geom_v.centroid))
                    candidate = (distance_m, int(u), int(v))
                    if best_pair is None or candidate < best_pair:
                        best_pair = candidate
            if best_pair is None:
                continue
            distance_m, u, v = best_pair
            graph.add_edge(
                u,
                v,
                adjacency_type="student_component_virtual_bridge",
                shared_boundary_m=0.0,
                virtual_bridge_distance_m=float(distance_m),
            )
            student_component_bridge_edges.append(
                {
                    "source": int(u),
                    "target": int(v),
                    "component_students": float(students),
                    "component_area_m2": float(area),
                    "virtual_bridge_distance_m": float(distance_m),
                }
            )

    graph.graph["adjacency_source"] = "shared_boundary_with_structural_point_and_student_component_virtual_bridges"
    graph.graph["min_shared_boundary_meters"] = MIN_SHARED_BOUNDARY_METERS
    graph.graph["point_bridge_min_component_area_m2"] = POINT_BRIDGE_MIN_COMPONENT_AREA_M2
    graph.graph["student_component_virtual_bridge_min_students"] = STUDENT_COMPONENT_VIRTUAL_BRIDGE_MIN_STUDENTS
    graph.graph["connectivity_distance_meters"] = 0.0
    graph.graph["component_bridge_edges"] = point_bridge_edges
    graph.graph["student_component_virtual_bridge_edges"] = student_component_bridge_edges
    graph.graph["rejected_point_contact_edge_count"] = rejected_point_contacts
    return graph




def drop_tiny_zero_student_school_sliver_components(gdf: gpd.GeoDataFrame, graph: nx.Graph) -> gpd.GeoDataFrame:
    """Drop non-main disconnected components that have no students.

    This catches cut-out attendance-zone fragments/slivers that are not
    connected to the main city optimization surface. If a zero-student unit is
    connected to the main component, it is retained.
    """
    if graph.number_of_nodes() == 0 or "students" not in gdf.columns:
        return gdf

    indexed = gdf.set_index("unit_id")
    students = pd.to_numeric(indexed["students"], errors="coerce").fillna(0.0)
    projected = gdf.to_crs(3857).set_index("unit_id")
    area = projected.geometry.area
    components = [{int(unit) for unit in component} for component in nx.connected_components(graph)]
    if len(components) <= 1:
        gdf.attrs["dropped_disconnected_zero_student_component_count"] = 0
        gdf.attrs["dropped_disconnected_zero_student_component_units"] = []
        return gdf

    main_component = max(
        components,
        key=lambda component: (
            float(students.reindex(list(component)).fillna(0.0).sum()),
            float(area.reindex(list(component)).fillna(0.0).sum()),
            len(component),
        ),
    )
    drop_units: set[int] = set()
    for component in components:
        if component == main_component:
            continue
        component_students = float(students.reindex(list(component)).fillna(0.0).sum())
        if component_students <= 0:
            drop_units.update(component)

    if not drop_units:
        gdf.attrs["dropped_disconnected_zero_student_component_count"] = 0
        gdf.attrs["dropped_disconnected_zero_student_component_units"] = []
        return gdf

    kept = gdf[~gdf["unit_id"].astype(int).isin(drop_units)].copy().reset_index(drop=True)
    kept.attrs.update(gdf.attrs)
    kept.attrs["dropped_disconnected_zero_student_component_count"] = len(drop_units)
    kept.attrs["dropped_disconnected_zero_student_component_units"] = sorted(drop_units)
    kept["original_unit_id"] = kept["unit_id"].astype(int)
    kept["unit_id"] = kept.index.astype(int)
    kept["idx_u"] = kept["unit_id"]
    kept.attrs["zoning_total_students"] = int(round(float(kept["students"].sum())))
    kept.attrs["zoning_total_capacity"] = int(round(float(kept["block_ele_cap"].sum())))
    kept.attrs["zoning_total_schools"] = int(round(float(kept["school_count"].sum())))
    return kept




def centroid_distances_to_centers(gdf: gpd.GeoDataFrame, centers: list[int]) -> dict[tuple[int, int], float]:
    projected = gdf.to_crs(3857)
    centroids = projected.geometry.centroid
    centroid_by_unit = {
        int(unit): centroids.iloc[pos]
        for pos, unit in enumerate(gdf["unit_id"].astype(int).tolist())
    }
    distances = {}
    for unit, unit_centroid in centroid_by_unit.items():
        for center in centers:
            center = int(center)
            distances[(unit, center)] = float(unit_centroid.distance(centroid_by_unit[center]) / 1609.344)
    return distances


def build_milp_cache(gdf: gpd.GeoDataFrame, graph: nx.Graph) -> dict[str, Any]:
    """Precompute city-static data reused across exact MILP attempts."""
    units = tuple(gdf["unit_id"].astype(int).tolist())
    projected = gdf.to_crs(3857)
    centroids = projected.geometry.centroid
    centroid_xy = {
        int(unit): (float(centroids.iloc[pos].x), float(centroids.iloc[pos].y))
        for pos, unit in enumerate(units)
    }
    indexed = gdf.set_index("unit_id")
    area_m2 = {
        int(unit): float(max(projected.geometry.iloc[pos].area, 0.0))
        for pos, unit in enumerate(units)
    }
    students = pd.to_numeric(indexed["students"], errors="coerce").fillna(0.0).clip(lower=0.0)
    capacities = pd.to_numeric(indexed["block_ele_cap"], errors="coerce").fillna(0.0).clip(lower=0.0)
    schools = pd.to_numeric(indexed["school_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    objective_unit_weight = {
        int(unit): float(
            max(
                float(students.get(int(unit), 0.0)),
                float(capacities.get(int(unit), 0.0)),
                1.0,
            )
        )
        for unit in units
    }
    quality_weight = indexed["school_quality_weight"].to_dict()
    quality_sums = {
        metric: indexed[f"{metric}_weighted_sum"].to_dict() for metric in QUALITY_METRICS
    }
    return {
        "units": units,
        "students": students.to_dict(),
        "capacities": capacities.to_dict(),
        "schools": schools.to_dict(),
        "frl": indexed["FRL_pct"].to_dict(),
        "focal_group": indexed["focal_group_pct"].to_dict(),
        "diversity_index": indexed["diversity_index"].to_dict(),
        "area_m2": area_m2,
        "objective_unit_weight": objective_unit_weight,
        "quality_weight": quality_weight,
        "quality_sums": quality_sums,
        "centroid_xy": centroid_xy,
        "neighbors": {int(u): [int(v) for v in graph.neighbors(u)] for u in units},
        "shortest_lengths_by_center": {},
        "geo_dist_by_center": {},
        "closer_neighbors_by_center": {},
    }


def cached_center_distances(cache: dict[str, Any], center: int) -> dict[int, float]:
    center = int(center)
    by_center = cache["geo_dist_by_center"]
    if center in by_center:
        return by_center[center]
    cx, cy = cache["centroid_xy"][center]
    distances = {}
    for unit, (ux, uy) in cache["centroid_xy"].items():
        distances[int(unit)] = float(math.hypot(ux - cx, uy - cy) / 1609.344)
    by_center[center] = distances
    return distances


def cached_shortest_lengths(cache: dict[str, Any], graph: nx.Graph, center: int) -> dict[int, int]:
    center = int(center)
    by_center = cache["shortest_lengths_by_center"]
    if center not in by_center:
        by_center[center] = nx.single_source_shortest_path_length(graph, center)
    return by_center[center]


def cached_closer_neighbors(cache: dict[str, Any], graph: nx.Graph, center: int) -> dict[int, list[int]]:
    center = int(center)
    by_center = cache["closer_neighbors_by_center"]
    if center in by_center:
        return by_center[center]
    lengths = cached_shortest_lengths(cache, graph, center)
    closer_by_unit: dict[int, list[int]] = {}
    for unit in cache["units"]:
        unit = int(unit)
        d_u = lengths.get(unit, math.inf)
        closer_by_unit[unit] = [
            int(v)
            for v in cache["neighbors"].get(unit, [])
            if lengths.get(int(v), math.inf) < d_u
        ]
    by_center[center] = closer_by_unit
    return closer_by_unit



def redistribute_empty_optimization_units(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    assignment: dict[int, int],
    centers: Optional[list[int]] = None,
    milp_cache: Optional[dict[str, Any]] = None,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Attach empty optimization units by simultaneous graph-front propagation.

    Empty units do not drive MILP feasibility. After the MILP assigns non-empty
    units, all non-empty zone fronts expand through the empty-unit subgraph at
    the same time. This avoids greedy order effects where one zone eats a whole
    empty chain before competing zones get a chance to reach it.
    """
    if gdf.empty:
        return assignment, {"empty_unit_reassignment_count": 0}
    indexed = gdf.set_index("unit_id")
    empty_units = {
        int(unit)
        for unit, row in indexed.iterrows()
        if float(row.get("students", 0.0) or 0.0) <= EMPTY_UNIT_ASSIGNMENT_EPS
        and float(row.get("school_count", 0.0) or 0.0) <= EMPTY_UNIT_ASSIGNMENT_EPS
        and float(row.get("block_ele_cap", 0.0) or 0.0) <= EMPTY_UNIT_ASSIGNMENT_EPS
    }
    if not empty_units:
        return assignment, {"empty_unit_reassignment_count": 0}

    new_assignment = {int(unit): int(zone) for unit, zone in assignment.items()}
    original_empty_assignment = {unit: new_assignment.get(unit) for unit in empty_units}

    protected_empty_units: set[int] = set()
    for zone in sorted(set(new_assignment.values())):
        zone_units = {int(unit) for unit, assigned_zone in new_assignment.items() if int(assigned_zone) == int(zone)}
        zone_empty_units = zone_units & empty_units
        if not zone_empty_units:
            continue
        base_components = nx.number_connected_components(graph.subgraph(zone_units)) if zone_units else 0
        for unit in zone_empty_units:
            remaining_units = zone_units - {int(unit)}
            if not remaining_units:
                continue
            if nx.number_connected_components(graph.subgraph(remaining_units)) > base_components:
                protected_empty_units.add(int(unit))

    redistributable_empty_units = empty_units - protected_empty_units
    assignment_sources = {
        int(unit)
        for unit in indexed.index.astype(int)
        if int(unit) not in redistributable_empty_units and int(unit) in new_assignment
    }
    if not assignment_sources:
        return new_assignment, {
            "empty_unit_reassignment_count": 0,
            "empty_unit_reassignment_skipped_reason": "no_assignment_sources",
            "empty_unit_protected_bridge_count": int(len(protected_empty_units)),
            "empty_unit_protected_bridge_units": sorted(protected_empty_units),
        }

    center_distance_by_unit_zone: dict[tuple[int, int], float] = {}
    if centers is not None and milp_cache is not None:
        for zone, center in enumerate(centers):
            try:
                distances = cached_center_distances(milp_cache, int(center))
                for unit, distance in distances.items():
                    center_distance_by_unit_zone[(int(unit), int(zone))] = float(distance)
            except Exception:
                continue

    def edge_cost(u: int, v: int) -> float:
        edge_data = graph.get_edge_data(int(u), int(v), default={}) or {}
        shared_boundary = float(edge_data.get("shared_boundary_m", 0.0) or 0.0)
        if milp_cache is not None and "centroid_xy" in milp_cache:
            ux, uy = milp_cache["centroid_xy"][int(u)]
            vx, vy = milp_cache["centroid_xy"][int(v)]
            centroid_cost = math.hypot(ux - vx, uy - vy) / 1609.344
        else:
            centroid_cost = 1.0
        # Favor broad shared boundaries while keeping a positive graph distance.
        boundary_discount = 1.0 / max(shared_boundary, 1.0)
        return max(centroid_cost, 1e-9) + boundary_discount

    heap: list[tuple[float, float, int, int, int]] = []
    best: dict[int, tuple[float, float, int]] = {}
    source_zone_count_by_empty_component: dict[int, set[int]] = {}
    empty_component_by_unit: dict[int, int] = {}
    empty_subgraph = graph.subgraph(redistributable_empty_units)
    for component_idx, component in enumerate(nx.connected_components(empty_subgraph)):
        component_set = {int(unit) for unit in component}
        source_zones = {
            int(new_assignment[int(neighbor)])
            for unit in component_set
            for neighbor in graph.neighbors(int(unit))
            if int(neighbor) in assignment_sources and int(neighbor) in new_assignment
        }
        source_zone_count_by_empty_component[component_idx] = source_zones
        for unit in component_set:
            empty_component_by_unit[int(unit)] = component_idx

    for source in sorted(assignment_sources):
        zone = int(new_assignment[source])
        for neighbor in graph.neighbors(source):
            neighbor = int(neighbor)
            if neighbor not in redistributable_empty_units:
                continue
            center_distance = center_distance_by_unit_zone.get((neighbor, zone), math.inf)
            cost = edge_cost(source, neighbor)
            candidate = (cost, center_distance, zone)
            if neighbor not in best or candidate < best[neighbor]:
                best[neighbor] = candidate
                heapq.heappush(heap, (cost, center_distance, zone, neighbor, source))

    while heap:
        cost, center_distance, zone, unit, _source = heapq.heappop(heap)
        if best.get(unit) != (cost, center_distance, zone):
            continue
        new_assignment[unit] = int(zone)
        for neighbor in graph.neighbors(unit):
            neighbor = int(neighbor)
            if neighbor not in redistributable_empty_units:
                continue
            next_cost = cost + edge_cost(unit, neighbor)
            next_center_distance = center_distance_by_unit_zone.get((neighbor, zone), math.inf)
            candidate = (next_cost, next_center_distance, zone)
            if neighbor not in best or candidate < best[neighbor]:
                best[neighbor] = candidate
                heapq.heappush(heap, (next_cost, next_center_distance, zone, neighbor, unit))

    fallback_count = 0
    unassigned_empty = redistributable_empty_units - set(best)
    if unassigned_empty:
        if milp_cache is not None and "centroid_xy" in milp_cache:
            assigned_units = [unit for unit in new_assignment if unit not in unassigned_empty]
            for unit in sorted(unassigned_empty):
                ux, uy = milp_cache["centroid_xy"][int(unit)]
                nearest = min(
                    assigned_units,
                    key=lambda other: math.hypot(
                        ux - milp_cache["centroid_xy"][int(other)][0],
                        uy - milp_cache["centroid_xy"][int(other)][1],
                    ),
                )
                new_assignment[unit] = int(new_assignment[int(nearest)])
                fallback_count += 1
        else:
            for unit in sorted(unassigned_empty):
                new_assignment[unit] = int(original_empty_assignment.get(unit, 0) or 0)
            fallback_count += len(unassigned_empty)

    connectivity_repair_count = 0
    for _ in range(max(1, len(redistributable_empty_units))):
        disconnected_zones = {
            int(zone)
            for zone in set(new_assignment.values())
            if nx.number_connected_components(
                graph.subgraph([unit for unit, assigned_zone in new_assignment.items() if int(assigned_zone) == int(zone)])
            )
            > 1
        }
        if not disconnected_zones:
            break
        units_to_revert = [
            int(unit)
            for unit in sorted(redistributable_empty_units)
            if original_empty_assignment.get(int(unit)) in disconnected_zones
            and new_assignment.get(int(unit)) != original_empty_assignment.get(int(unit))
        ]
        if not units_to_revert:
            break
        for unit in units_to_revert:
            new_assignment[unit] = int(original_empty_assignment[unit])
        connectivity_repair_count += len(units_to_revert)

    changed_units = [
        int(unit)
        for unit in sorted(redistributable_empty_units)
        if original_empty_assignment.get(unit) is not None
        and int(original_empty_assignment[unit]) != int(new_assignment[unit])
    ]
    multi_source_components = sum(1 for zones in source_zone_count_by_empty_component.values() if len(zones) > 1)
    return new_assignment, {
        "empty_unit_reassignment_count": len(changed_units),
        "empty_unit_reassigned_units": changed_units,
        "empty_unit_fallback_assignment_count": int(fallback_count),
        "empty_unit_protected_bridge_count": int(len(protected_empty_units)),
        "empty_unit_protected_bridge_units": sorted(protected_empty_units),
        "empty_unit_connectivity_repair_count": int(connectivity_repair_count),
        "empty_unit_component_count": int(len(source_zone_count_by_empty_component)),
        "empty_unit_multi_source_component_count": int(multi_source_components),
        "empty_unit_assignment_rule": "multi_source_graph_voronoi_from_adjacent_non_empty_units",
    }


def solve_exact_milp(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    config: ExactZoneConfig,
    centers: list[int],
    milp_cache: Optional[dict[str, Any]] = None,
) -> tuple[dict[int, int], dict[str, Any]]:
    if milp_cache is None:
        milp_cache = build_milp_cache(gdf, graph)
    units = list(milp_cache["units"])
    z_count = config.n_zones
    n_assignment_vars = len(units) * z_count
    quality_metric_count = len(QUALITY_METRICS)
    n_demographic_deviation_metrics = 3
    n_deviation_vars = (n_demographic_deviation_metrics + quality_metric_count) * z_count
    y_idx = n_assignment_vars + n_deviation_vars
    n_vars = n_assignment_vars + n_deviation_vars + 1
    students = milp_cache["students"]
    capacities = milp_cache["capacities"]
    schools = milp_cache["schools"]
    area_m2 = milp_cache["area_m2"]
    objective_unit_weight = milp_cache["objective_unit_weight"]
    frl = milp_cache["frl"]
    focal_group = milp_cache["focal_group"]
    diversity_index = milp_cache["diversity_index"]
    district_frl = float(np.average(gdf["FRL_pct"], weights=np.maximum(gdf["students"], 1e-9)))
    district_focal_group = float(np.average(gdf["focal_group_pct"], weights=np.maximum(gdf["students"], 1e-9)))
    district_diversity_index = float(np.average(gdf["diversity_index"], weights=np.maximum(gdf["students"], 1e-9)))
    quality_weight = milp_cache["quality_weight"]
    quality_sums = milp_cache["quality_sums"]
    district_quality = {}
    for metric in QUALITY_METRICS:
        total_weight = float(sum(quality_weight.values()))
        district_quality[metric] = (
            float(sum(quality_sums[metric].values()) / total_weight) if total_weight > 0 else 0.0
        )
    total_students = float(sum(students.values()))
    avg_students = total_students / z_count
    balance_B = config.balance_tolerance * avg_students
    total_schools = float(sum(schools.values()))
    avg_schools = total_schools / z_count
    min_schools = int(config.min_schools_per_zone) if config.min_schools_per_zone is not None else None
    max_schools = int(config.max_schools_per_zone) if config.max_schools_per_zone is not None else None
    geo_dist_by_center = {int(center): cached_center_distances(milp_cache, int(center)) for center in centers}
    closer_by_center = {int(center): cached_closer_neighbors(milp_cache, graph, int(center)) for center in centers}

    unit_pos = {unit: i for i, unit in enumerate(units)}
    matrix_row_idx: list[int] = []
    matrix_col_idx: list[int] = []
    matrix_data: list[float] = []
    lows: list[float] = []
    highs: list[float] = []

    def idx(unit: int, zone: int) -> int:
        return unit_pos[unit] * z_count + zone

    def dev_idx(metric_pos: int, zone: int) -> int:
        return n_assignment_vars + metric_pos * z_count + zone

    def add_terms(cols: list[int], coeffs: list[float], low: float = -np.inf, high: float = np.inf) -> None:
        row_idx = len(lows)
        for col_idx, coeff in zip(cols, coeffs):
            if coeff:
                matrix_row_idx.append(row_idx)
                matrix_col_idx.append(int(col_idx))
                matrix_data.append(float(coeff))
        lows.append(float(low))
        highs.append(float(high))

    def add(row: dict[int, float], low: float = -np.inf, high: float = np.inf) -> None:
        add_terms(list(row.keys()), list(row.values()), low, high)

    objective = np.zeros(n_vars)
    for u in units:
        for z in range(z_count):
            objective[idx(u, z)] = (
                config.objective_distance_weight
                * (geo_dist_by_center[int(centers[z])][u] ** config.objective_distance_power)
                * float(objective_unit_weight[u])
            )
    for z in range(z_count):
        objective[dev_idx(0, z)] = config.objective_diversity_weight
        objective[dev_idx(1, z)] = config.objective_diversity_weight
        objective[dev_idx(2, z)] = config.objective_diversity_weight
        for metric_offset in range(quality_metric_count):
            objective[dev_idx(n_demographic_deviation_metrics + metric_offset, z)] = config.objective_quality_weight
    objective[y_idx] = -config.objective_min_surplus_weight

    for u in units:
        add_terms([idx(u, z) for z in range(z_count)], [1.0] * z_count, 1.0, 1.0)

    zone_student_rows = []
    for z, center in enumerate(centers):
        add_terms([idx(center, z)], [1.0], 1.0, 1.0)
        student_row = {idx(u, z): float(students[u]) for u in units}
        zone_student_rows.append(student_row)
        if min_schools is not None or max_schools is not None:
            add(
                {idx(u, z): float(schools[u]) for u in units},
                -np.inf if min_schools is None else min_schools,
                np.inf if max_schools is None else max_schools,
            )
        surplus_row = {idx(u, z): float(capacities[u] - students[u]) for u in units}
        surplus_row[y_idx] = -1.0
        add(surplus_row, 0.0, np.inf)
        if config.max_shortage is not None:
            add({idx(u, z): float(capacities[u] - students[u]) for u in units}, -float(config.max_shortage), np.inf)
        if config.cover_distance_miles is not None:
            add(
                {
                    idx(u, z): float((geo_dist_by_center[int(center)][u] - config.cover_distance_miles) * students[u])
                    for u in units
                },
                -np.inf,
                0.0,
            )
        if config.frl_bounds is not None:
            add(
                {idx(u, z): float((frl[u] - config.frl_bounds[0] * district_frl) * students[u]) for u in units},
                0.0,
                np.inf,
            )
            add(
                {idx(u, z): float((frl[u] - config.frl_bounds[1] * district_frl) * students[u]) for u in units},
                -np.inf,
                0.0,
            )
        if config.focal_group_bounds is not None:
            add(
                {
                    idx(u, z): float((focal_group[u] - config.focal_group_bounds[0] * district_focal_group) * students[u])
                    for u in units
                },
                0.0,
                np.inf,
            )
            add(
                {
                    idx(u, z): float((focal_group[u] - config.focal_group_bounds[1] * district_focal_group) * students[u])
                    for u in units
                },
                -np.inf,
                0.0,
            )
        if config.diversity_index_bounds is not None:
            add(
                {
                    idx(u, z): float(
                        (diversity_index[u] - config.diversity_index_bounds[0] * district_diversity_index)
                        * students[u]
                    )
                    for u in units
                },
                0.0,
                np.inf,
            )
            add(
                {
                    idx(u, z): float(
                        (diversity_index[u] - config.diversity_index_bounds[1] * district_diversity_index)
                        * students[u]
                    )
                    for u in units
                },
                -np.inf,
                0.0,
            )
        if config.school_quality_bounds is not None:
            for metric in QUALITY_METRICS:
                district_metric = district_quality[metric]
                if district_metric <= 0:
                    continue
                lower, upper = config.school_quality_bounds
                add(
                    {
                        idx(u, z): float(quality_sums[metric][u] - lower * district_metric * quality_weight[u])
                        for u in units
                    },
                    0.0,
                    np.inf,
                )
                add(
                    {
                        idx(u, z): float(quality_sums[metric][u] - upper * district_metric * quality_weight[u])
                        for u in units
                    },
                    -np.inf,
                    0.0,
                )
        for metric_pos, values, district_avg in [
            (0, frl, district_frl),
            (1, focal_group, district_focal_group),
            (2, diversity_index, district_diversity_index),
        ]:
            deviation_row = {dev_idx(metric_pos, z): 1.0}
            for u in units:
                deviation_row[idx(u, z)] = float(-(values[u] - district_avg) * students[u])
            add(deviation_row, 0.0, np.inf)
            deviation_row = {dev_idx(metric_pos, z): 1.0}
            for u in units:
                deviation_row[idx(u, z)] = float((values[u] - district_avg) * students[u])
            add(deviation_row, 0.0, np.inf)
        for metric_offset, metric in enumerate(QUALITY_METRICS):
            metric_pos = n_demographic_deviation_metrics + metric_offset
            district_metric = district_quality[metric]
            if district_metric <= 0:
                continue
            scale = max(district_metric, 1.0)
            deviation_row = {dev_idx(metric_pos, z): 1.0}
            for u in units:
                deviation_row[idx(u, z)] = float(
                    -(quality_sums[metric][u] - district_metric * quality_weight[u]) / scale
                )
            add(deviation_row, 0.0, np.inf)
            deviation_row = {dev_idx(metric_pos, z): 1.0}
            for u in units:
                deviation_row[idx(u, z)] = float(
                    (quality_sums[metric][u] - district_metric * quality_weight[u]) / scale
                )
            add(deviation_row, 0.0, np.inf)
        for u in units:
            if u == center:
                continue
            closer = closer_by_center[int(center)].get(int(u), [])
            if closer:
                add_terms(
                    [idx(u, z)] + [idx(int(v), z) for v in closer],
                    [1.0] + [-1.0] * len(closer),
                    -np.inf,
                    0.0,
                )
            else:
                add_terms([idx(u, z)], [1.0], 0.0, 0.0)

    for z in range(z_count):
        for w in range(z + 1, z_count):
            row = dict(zone_student_rows[z])
            for col, coeff in zone_student_rows[w].items():
                row[col] = row.get(col, 0.0) - coeff
            add(row, -balance_B, balance_B)

    matrix = coo_matrix(
        (matrix_data, (matrix_row_idx, matrix_col_idx)),
        shape=(len(lows), n_vars),
        dtype=float,
    ).tocsr()
    solver_threads = max(1, int(os.environ.get("ZONE_OPTIMIZATION_SOLVER_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))))
    solver_options = {"threads": solver_threads}
    if config.solver_time_limit_seconds is not None and config.solver_time_limit_seconds > 0:
        solver_options["time_limit"] = float(config.solver_time_limit_seconds)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        result = milp(
            c=objective,
            integrality=np.concatenate([np.ones(n_assignment_vars), np.zeros(n_deviation_vars + 1)]),
            bounds=Bounds(
                np.concatenate([np.zeros(n_assignment_vars + n_deviation_vars), np.array([-np.inf])]),
                np.concatenate([np.ones(n_assignment_vars), np.full(n_deviation_vars + 1, np.inf)]),
            ),
            constraints=LinearConstraint(matrix.tocsr(), np.array(lows), np.array(highs)),
            options=solver_options,
        )
    if not result.success:
        raise RuntimeError(f"MILP status {result.message}")

    assignment = {}
    for u in units:
        vals = result.x[[idx(u, z) for z in range(z_count)]]
        assignment[u] = int(np.argmax(vals))
    assignment, empty_assignment_diagnostics = redistribute_empty_optimization_units(
        gdf,
        graph,
        assignment,
        centers=centers,
        milp_cache=milp_cache,
    )
    diagnostics = {
        "solver_status": f"SciPy/HiGHS: {result.message}",
        "objective": float(result.fun),
        "centers": centers,
        "avg_students": avg_students,
        "avg_schools": avg_schools,
        "min_schools": min_schools,
        "max_schools": max_schools,
        "district_quality": district_quality,
        "school_quality_bounds": config.school_quality_bounds,
        "balance_tolerance": config.balance_tolerance,
        "balance_B_students": balance_B,
        "cover_distance_miles": config.cover_distance_miles,
        "max_shortage": config.max_shortage,
        "objective_distance_power": config.objective_distance_power,
        "objective_minimum_seat_surplus": float(result.x[y_idx]),
        "variables": int(n_vars),
        "constraints": int(len(lows)),
        "nonzeros": int(matrix.nnz),
        "solver_threads": solver_threads,
    }
    diagnostics.update(empty_assignment_diagnostics)
    return assignment, diagnostics


def _solve_exact_milp_worker(
    queue: Any,
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    config: ExactZoneConfig,
    centers: list[int],
    milp_cache: Optional[dict[str, Any]],
) -> None:
    try:
        assignment, diagnostics = solve_exact_milp(gdf, graph, config, centers, milp_cache=milp_cache)
    except BaseException as exc:
        queue.put(("error", type(exc).__name__, str(exc)))
    else:
        queue.put(("ok", assignment, diagnostics))


def solve_exact_milp_with_wall_timeout(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    config: ExactZoneConfig,
    centers: list[int],
    milp_cache: Optional[dict[str, Any]],
    wall_time_limit_seconds: Optional[int],
) -> tuple[dict[int, int], dict[str, Any]]:
    if wall_time_limit_seconds is None or wall_time_limit_seconds <= 0:
        return solve_exact_milp(gdf, graph, config, centers, milp_cache=milp_cache)
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_solve_exact_milp_worker,
        args=(queue, gdf, graph, config, centers, milp_cache),
    )
    process.start()
    process.join(float(wall_time_limit_seconds))
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise RuntimeError(
            f"MILP status Time limit reached by hard wall clock after {wall_time_limit_seconds} seconds"
        )
    if queue.empty():
        raise RuntimeError(
            f"MILP worker exited without returning a result; exitcode={process.exitcode}"
        )
    status, *payload = queue.get()
    if status == "ok":
        assignment, diagnostics = payload
        return assignment, diagnostics
    exc_type, message = payload
    raise RuntimeError(f"{exc_type}: {message}")


def summarize_solution(
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    assignment: dict[int, int],
    centers: Optional[list[int]] = None,
    distances: Optional[dict[tuple[int, int], float]] = None,
) -> pd.DataFrame:
    df = gdf.drop(columns="geometry").copy()
    df["zone"] = df["unit_id"].map(assignment)
    geom_df = gdf[["unit_id", "geometry"]].copy()
    geom_df["zone"] = geom_df["unit_id"].map(assignment)
    rows = []
    for zone, group in df.groupby("zone"):
        weights = group["students"].replace(0, np.nan).fillna(1.0)
        zone_units = group["unit_id"].astype(int).tolist()
        union_geom = safe_union_geometries(geom_df[geom_df["zone"] == zone].to_crs(3857).geometry)
        geometric_parts = 1 if union_geom.geom_type == "Polygon" else len(union_geom.geoms)
        row = {
            "zone": int(zone),
            "units": int(len(group)),
            "connected_components": int(nx.number_connected_components(graph.subgraph(zone_units))),
            "geometric_parts": int(geometric_parts),
            "students": int(round(float(group["students"].sum()))),
            "capacity": int(round(float(group["block_ele_cap"].sum()))),
            "seat_surplus": int(round(float(group["block_ele_cap"].sum() - group["students"].sum()))),
            "schools": int(round(float(group["school_count"].sum()))),
            "FRL_pct": float(np.average(group["FRL_pct"], weights=weights)),
            "focal_group_count": int(round(float(group["focal_group_count"].sum()))),
            "focal_group_pct": float(np.average(group["focal_group_pct"], weights=weights)),
            "diversity_index": float(np.average(group["diversity_index"], weights=weights)),
            "asian_pct": float(np.average(group["asian_pct"], weights=weights)),
            "white_pct": float(np.average(group["white_pct"], weights=weights)),
            "native_american_pct": float(np.average(group["native_american_pct"], weights=weights)),
            "pacific_islander_pct": float(np.average(group["pacific_islander_pct"], weights=weights)),
            "multiracial_pct": float(np.average(group["multiracial_pct"], weights=weights)),
        }
        school_quality_weight = float(group["school_quality_weight"].sum())
        if school_quality_weight > 0:
            for metric in QUALITY_METRICS:
                row[metric] = float(group[f"{metric}_weighted_sum"].sum() / school_quality_weight)
            row["school_quality_index"] = float(
                np.average(
                    group.loc[group["quality_school_count"] > 0, "school_quality_index"],
                    weights=group.loc[group["quality_school_count"] > 0, "quality_school_count"],
                )
                if (group["quality_school_count"] > 0).any()
                else 0.0
            )
        else:
            for metric in QUALITY_METRICS:
                row[metric] = 0.0
            row["school_quality_index"] = 0.0
        if centers is not None and distances is not None:
            center = centers[int(zone)]
            unit_distances = np.array([distances[(int(u), center)] for u in zone_units], dtype=float)
            student_weights = group["students"].to_numpy(dtype=float)
            safe_weights = np.maximum(student_weights, 1e-9)
            row["center_unit_id"] = int(center)
            row["avg_center_distance_miles"] = float(np.average(unit_distances, weights=safe_weights))
            row["max_center_distance_miles"] = float(unit_distances.max()) if len(unit_distances) else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("zone")


def plot_solution(
    gdf: gpd.GeoDataFrame,
    assignment: dict[int, int],
    out_png: Path,
    city: str = DEFAULT_CITY,
    school_csv: Path = ATL_PUBLIC_SCHOOLS_CSV,
    block_source: Path = ATL_BLOCK_SOURCE,
) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.cm import ScalarMappable

    plot_df = gdf.copy()
    plot_df["zone"] = plot_df["unit_id"].map(assignment)
    zones = sorted(plot_df["zone"].dropna().astype(int).unique())
    colors = [
        "#EE6148",
        "#F89E40",
        "#F0C646",
        "#91D7D3",
        "#4099A0",
        "#385C8C",
        "#DFCCAC",
        "#AB7C58",
        "#FBD1D8",
        "#F5F3EB",
        "#D94A3A",
        "#F47A45",
        "#F6D27A",
        "#7EC5C0",
        "#2F7F89",
        "#2F4E7E",
        "#E6D8B6",
        "#9E6A47",
        "#F7B6C1",
        "#E8E4DA",
        "#C73E2A",
        "#E9643B",
        "#E9B56A",
        "#5FB8B4",
        "#1E6770",
        "#27426B",
        "#D8C29B",
        "#8B5A34",
        "#F29AA7",
        "#D6D1C2",
        "#B3281E",
        "#D95F24",
        "#D9A441",
        "#3EA39F",
        "#14565E",
        "#1E3556",
        "#C3AE86",
        "#744728",
        "#E47D8C",
        "#9E988C",
        "#8C1F16",
        "#C24E1A",
        "#C28A2E",
        "#2C7F7B",
        "#0E3E45",
        "#14263E",
        "#A28F6B",
        "#5A3621",
        "#C85D6A",
        "#4A443D",
    ]
    if len(zones) > len(colors):
        raise ValueError(
            f"Cannot plot {len(zones)} zones with the configured {len(colors)}-color palette; "
            "please extend the palette before rendering this city."
        )
    colors = colors[: len(zones)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(zones) + 0.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10.9))
    fig.subplots_adjust(top=0.90, right=0.86)

    # The optimization units are attendance-zone polygons. Dissolving them by
    # assignment gives the ATL-style final zone surface.
    zone_shells = safe_dissolve_by_zone(plot_df)
    zone_shells.plot(column="zone", cmap=cmap, norm=norm, linewidth=0.22, edgecolor="#f7f7f7", ax=ax)

    # Keep the same fine white attendance-zone linework style as the ATL output.
    try:
        plot_df.boundary.plot(ax=ax, color="white", linewidth=0.08, alpha=0.50)
    except Exception:
        pass

    # ATL-style black separators between final dissolved zones.
    dissolved = zone_shells.set_index("zone")
    exterior_lines = []
    for geom in dissolved.geometry:
        exterior_lines.extend(exterior_boundary_lines(geom))
    if exterior_lines:
        gpd.GeoSeries(exterior_lines, crs=plot_df.crs).plot(ax=ax, color="#444444", linewidth=0.55)

    # External outline follows the effective CityData geometry used by the MILP.
    # Keep interior holes so non-city/non-data enclaves do not look unassigned.
    effective_boundary_geom = safe_union_geometries(zone_shells.to_crs(3857).geometry)
    effective_boundary_geom = gpd.GeoSeries([effective_boundary_geom], crs="EPSG:3857").to_crs(plot_df.crs).iloc[0]
    city_boundary = gpd.GeoDataFrame(
        {"source": ["effective_optimization_units"]},
        geometry=[effective_boundary_geom],
        crs=plot_df.crs,
    )
    if city_boundary is not None and not city_boundary.empty:
        try:
            city_boundary.boundary.plot(ax=ax, color="#111111", linewidth=0.90, alpha=0.95)
        except Exception:
            pass

    # Mark elementary-serving school locations. Include all school points in
    # the viewport even when SABS polygons do not cover the building point.
    school_points_for_bounds = None
    try:
        schools = load_elementary_school_quality(school_csv)
        school_points = school_points_from_coordinates(schools)
        if not school_points.empty:
            school_points = school_points.to_crs(plot_df.crs)
            school_points_for_bounds = school_points
            school_points.plot(
                ax=ax,
                marker="*",
                color="#111111",
                markersize=28,
                alpha=0.95,
                edgecolor="white",
                linewidth=0.35,
                zorder=6,
            )
    except Exception:
        pass

    bounds_sources = []
    if city_boundary is not None and not city_boundary.empty:
        bounds_sources.append(city_boundary)
    else:
        bounds_sources.append(zone_shells)
    if school_points_for_bounds is not None and not school_points_for_bounds.empty:
        bounds_sources.append(school_points_for_bounds)
    try:
        bounds = np.array([source.total_bounds for source in bounds_sources if source is not None and not source.empty])
        minx, miny = bounds[:, [0, 1]].min(axis=0)
        maxx, maxy = bounds[:, [2, 3]].max(axis=0)
        pad_x = (maxx - minx) * 0.06 if maxx > minx else 1000
        pad_y = (maxy - miny) * 0.06 if maxy > miny else 1000
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    except Exception:
        pass

    ax.set_axis_off()
    fig.suptitle(f"{city.upper()} Typology Aware Learning + MILP Zoning", fontsize=20, y=0.965)
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar_ax = fig.add_axes([0.89, 0.14, 0.026, 0.72])
    colorbar = fig.colorbar(mappable, cax=cbar_ax, ticks=zones)
    colorbar.ax.tick_params(labelsize=14)
    colorbar.outline.set_linewidth(1.0)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def score_solution(
    summary: pd.DataFrame,
    avg_students: float,
    avg_schools: float,
    weights: ScoreWeights = ScoreWeights(),
) -> float:
    student_cv = float(summary["students"].std(ddof=0) / max(avg_students, 1.0)) if len(summary) else 0.0
    school_dev = float((summary["schools"] - avg_schools).abs().mean() / max(avg_schools, 1.0))
    frl_spread = float(summary["FRL_pct"].max() - summary["FRL_pct"].min())
    focal_group_spread = float(summary["focal_group_pct"].max() - summary["focal_group_pct"].min())
    diversity_spread = float(summary["diversity_index"].max() - summary["diversity_index"].min())
    quality_spread = 0.0
    for metric in QUALITY_METRICS:
        if metric in summary.columns:
            denom = max(float(summary[metric].mean()), 1.0)
            quality_spread += float((summary[metric].max() - summary[metric].min()) / denom)
    distance_score = 0.0
    if "avg_center_distance_miles" in summary.columns:
        distance_score = float(
            summary["avg_center_distance_miles"].mean() / 10.0
            + summary["avg_center_distance_miles"].max() / 10.0
        )
    if "max_center_distance_miles" in summary.columns:
        distance_score += float(summary["max_center_distance_miles"].max() / 15.0)
    shortage_score = 0.0
    if "seat_surplus" in summary.columns:
        shortage_score = float((-summary["seat_surplus"].clip(upper=0)).max() / max(avg_students, 1.0))
    disconnected = float((summary["connected_components"] - 1).clip(lower=0).sum())
    return (
        weights.student_cv * student_cv
        + weights.school_dev * school_dev
        + weights.frl_spread * frl_spread
        + weights.focal_group_spread * focal_group_spread
        + weights.diversity_spread * diversity_spread
        + weights.quality_spread * quality_spread
        + weights.shortage * shortage_score
        + weights.distance * distance_score
        + weights.disconnected * disconnected
    )


def annealed_zone_bounds(gdf: gpd.GeoDataFrame, config: AnnealingConfig) -> tuple[int, int]:
    total_schools = int(round(float(gdf["school_count"].sum())))
    max_units = int(len(gdf))
    lower = 2 if config.min_zones is None else int(config.min_zones)
    if config.max_zones is None:
        upper = min(max_units, max(2, math.ceil(total_schools / 2)))
    else:
        upper = int(config.max_zones)
    lower = max(1, min(lower, max_units))
    upper = max(lower, min(upper, max_units))
    return lower, upper


def city_distance_scale_miles(gdf: gpd.GeoDataFrame) -> float:
    projected = gdf.to_crs(3857)
    area = float(projected.geometry.area.sum())
    if np.isfinite(area) and area > 0:
        return max(math.sqrt(area / math.pi) / 1609.344, 0.25)
    minx, miny, maxx, maxy = projected.total_bounds
    diagonal = math.hypot(float(maxx - minx), float(maxy - miny)) / 1609.344
    return max(diagonal / 4.0, 0.25)


def center_candidate_pool(gdf: gpd.GeoDataFrame, n_zones: int) -> list[int]:
    school_units = gdf[gdf["school_count"] > 0].copy()
    pool = school_units if len(school_units) >= n_zones else gdf
    if "school_quality_index" in pool.columns and pool["school_quality_index"].notna().any():
        quality_threshold = pool["school_quality_index"].quantile(0.60)
        quality_pool = pool[pool["school_quality_index"] >= quality_threshold].copy()
        if len(quality_pool) >= n_zones:
            pool = quality_pool
    return pool["unit_id"].astype(int).tolist()


def weighted_choice(items: list[int], weights: list[float], rng: random.Random) -> int:
    positive = [max(float(w), 0.0) for w in weights]
    total = sum(positive)
    if total <= 0:
        return int(rng.choice(items))
    threshold = rng.random() * total
    running = 0.0
    for item, weight in zip(items, positive):
        running += weight
        if running >= threshold:
            return int(item)
    return int(items[-1])


def graph_components_by_size(graph: Optional[nx.Graph]) -> list[set[int]]:
    if graph is None or graph.number_of_nodes() == 0:
        return []
    return [
        {int(unit) for unit in component}
        for component in sorted(nx.connected_components(graph), key=len, reverse=True)
    ]


def seed_centers(
    gdf: gpd.GeoDataFrame,
    n_zones: int,
    rng: random.Random,
    graph: Optional[nx.Graph] = None,
) -> list[int]:
    pool = center_candidate_pool(gdf, n_zones)
    if len(pool) < n_zones:
        raise ValueError(f"Need {n_zones} centers but only {len(pool)} candidate units are available")
    projected = gdf.to_crs(3857)
    centroid_by_unit = {
        int(unit): projected.geometry.iloc[pos].centroid
        for pos, unit in enumerate(gdf["unit_id"].astype(int).tolist())
    }
    quality = (
        gdf.set_index("unit_id")["school_quality_index"].fillna(0.0).to_dict()
        if "school_quality_index" in gdf.columns
        else {}
    )
    students = gdf.set_index("unit_id")["students"].fillna(0.0).to_dict()
    def base_weight(unit: int) -> float:
        return (1.0 + float(students.get(unit, 0.0))) * (1.0 + float(quality.get(unit, 0.0)))

    all_units = gdf["unit_id"].astype(int).tolist()
    pool_set = set(pool)
    components = graph_components_by_size(graph)
    if components and len(components) > n_zones:
        raise ValueError(f"Need at least {len(components)} zones for {len(components)} connected components")

    centers = []
    for component in components:
        component_candidates = [unit for unit in pool if unit in component]
        if not component_candidates:
            component_candidates = [unit for unit in all_units if unit in component]
        if component_candidates:
            centers.append(weighted_choice(component_candidates, [base_weight(unit) for unit in component_candidates], rng))

    if not centers:
        centers = [weighted_choice(pool, [base_weight(unit) for unit in pool], rng)]
    while len(centers) < n_zones:
        remaining = [unit for unit in pool if unit not in centers]
        if not remaining:
            remaining = [unit for unit in all_units if unit not in centers]
        weights = []
        for unit in remaining:
            pt = centroid_by_unit[unit]
            min_distance = min(pt.distance(centroid_by_unit[center]) for center in centers)
            component_bonus = 1.0 if unit in pool_set else 0.75
            weights.append(component_bonus * base_weight(unit) * max(min_distance, 1.0))
        centers.append(weighted_choice(remaining, weights, rng))
    return centers


















def write_outputs(
    output_dir: Path,
    gdf: gpd.GeoDataFrame,
    graph: nx.Graph,
    assignment: dict[int, int],
    summary: pd.DataFrame,
    diagnostics: dict[str, Any],
    city: str = DEFAULT_CITY,
    school_csv: Path = ATL_PUBLIC_SCHOOLS_CSV,
    block_source: Path = ATL_BLOCK_SOURCE,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    city_code = city.strip()
    city_lower = city_code.lower()
    assign_df = gdf.drop(columns="geometry").copy()
    assign_df["zone"] = assign_df["unit_id"].map(assignment)
    for count_col in ["students", "school_count", "source_ele_school_count", "block_ele_cap", "focal_group_count"]:
        if count_col in assign_df.columns:
            assign_df[count_col] = assign_df[count_col].round().astype(int)
    assignment_cols = [
        "unit_id",
        "idx_u",
        "parent_unit_id",
        "unit_part",
        "zone",
        "unit_name",
        "GEOID20",
        "SrcName",
        "schnam",
        "students",
        "block_ele_cap",
        "school_count",
        "enrol_e_A",
        "enrol_e_B",
        "enrol_e_H",
        "enrol_e_W",
        "enrol_e_AM",
        "enrol_e_HP",
        "enrol_e_TR",
        "enrol_e_p",
        "FRL_pct",
        "acs_block_group_geoid",
        "acs_poverty_rate",
        "poverty_source",
        "focal_group_count",
        "focal_group_pct",
        "diversity_index",
        "asian_pct",
        "white_pct",
        "native_american_pct",
        "pacific_islander_pct",
        "multiracial_pct",
        "CCRPI",
        "ELA_prof_plus",
        "Math_prof_plus",
        "school_quality_index",
    ]
    for col in assignment_cols:
        if col not in assign_df.columns:
            assign_df[col] = pd.NA
    assign_df[assignment_cols].to_csv(output_dir / f"{city_lower}_zone_assignment.csv", index=False)
    summary.to_csv(output_dir / f"{city_lower}_zone_summary.csv", index=False)
    diagnostics["zone_summary"] = summary.to_dict("records")
    diagnostics["data_sources"] = {
        "city": city_code,
        "unit_source": gdf.attrs.get("unit_source"),
        "block_source": gdf.attrs.get("block_source"),
        "city_boundary_source": gdf.attrs.get("city_boundary_source"),
        "city_boundary_clip_applied": gdf.attrs.get("city_boundary_clip_applied"),
        "city_boundary_clip_reason": gdf.attrs.get("city_boundary_clip_reason"),
        "city_boundary_clip_units_before": gdf.attrs.get("city_boundary_clip_units_before"),
        "city_boundary_clip_units_after": gdf.attrs.get("city_boundary_clip_units_after"),
        "city_boundary_clip_area_before_m2": gdf.attrs.get("city_boundary_clip_area_before_m2"),
        "city_boundary_clip_area_after_m2": gdf.attrs.get("city_boundary_clip_area_after_m2"),
        "exploded_multipart_unit_count": gdf.attrs.get("exploded_multipart_unit_count"),
        "exploded_unit_count_after": gdf.attrs.get("exploded_unit_count_after"),
        "graph_json": gdf.attrs.get("graph_json"),
        "adjacency_source": graph.graph.get("adjacency_source"),
        "poverty_sources": gdf.attrs.get("poverty_sources"),
        "zoning_total_students": gdf.attrs.get("zoning_total_students"),
        "dropped_empty_isolated_units": gdf.attrs.get("dropped_empty_isolated_units", []),
        "dropped_no_school_component_count": gdf.attrs.get("dropped_no_school_component_count", 0),
        "dropped_no_school_component_units": gdf.attrs.get("dropped_no_school_component_units", []),
        "dropped_no_school_component_students": gdf.attrs.get("dropped_no_school_component_students", 0.0),
        "dropped_no_school_component_capacity": gdf.attrs.get("dropped_no_school_component_capacity", 0.0),
        "retained_school_component_count": gdf.attrs.get("retained_school_component_count"),
        "zoning_total_capacity": gdf.attrs.get("zoning_total_capacity"),
        "zoning_total_schools": gdf.attrs.get("zoning_total_schools"),
        "school_quality_csv": str(school_csv),
        "focal_group_definition": gdf.attrs.get("focal_group_definition", []),
        "focal_group_labels": gdf.attrs.get("focal_group_labels", []),
        "focal_group_thresholds": gdf.attrs.get("focal_group_thresholds", {}),
        "focal_group_diagnostics": gdf.attrs.get("focal_group_diagnostics", []),
    }
    diagnostics["graph"] = {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "adjacency_source": graph.graph.get("adjacency_source"),
        "min_shared_boundary_meters": graph.graph.get("min_shared_boundary_meters"),
        "component_bridge_edges": graph.graph.get("component_bridge_edges", []),
        "component_bridge_edge_count": len(graph.graph.get("component_bridge_edges", [])),
        "components": [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)],
        "geometric_multipart_zone_count": int(
            (pd.to_numeric(summary["geometric_parts"], errors="coerce").fillna(1) > 1).sum()
        )
        if "geometric_parts" in summary.columns
        else 0,
        "max_geometric_parts": int(pd.to_numeric(summary["geometric_parts"], errors="coerce").fillna(1).max())
        if "geometric_parts" in summary.columns and len(summary)
        else 1,
    }
    (output_dir / f"{city_lower}_zone_diagnostics.json").write_text(
        json.dumps(_jsonify(diagnostics), indent=2), encoding="utf-8"
    )
    plot_solution(gdf, assignment, output_dir / f"{city_lower}_zones.png", city_code, school_csv, block_source)


def is_exact_milp_success(attempt: dict[str, Any]) -> bool:
    return str(attempt.get("status", "")).startswith("SciPy/HiGHS") and np.isfinite(
        float(attempt.get("score", math.inf))
    )


def is_exact_milp_failure(attempt: dict[str, Any]) -> bool:
    return str(attempt.get("status", "")) == "failed" and "error" in attempt


def exact_attempts_until_selected_best(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_index: Optional[int] = None
    selected_score = math.inf
    for index, attempt in enumerate(attempts):
        if not is_exact_milp_success(attempt):
            continue
        score = float(attempt.get("score", math.inf))
        if score < selected_score:
            selected_score = score
            selected_index = index
    if selected_index is None:
        return []
    return [
        attempt
        for attempt in attempts[: selected_index + 1]
        if is_exact_milp_success(attempt) or is_exact_milp_failure(attempt)
    ]


def build_city_comparison_row(city: str, summary: pd.DataFrame, diagnostics: dict[str, Any]) -> dict[str, Any]:
    attempts = diagnostics.get("search_attempts", []) or []
    exact_attempts_until_success = exact_attempts_until_selected_best(attempts)
    attempt_count = len(exact_attempts_until_success)
    feasible_count = 1 if exact_attempts_until_success and is_exact_milp_success(exact_attempts_until_success[-1]) else 0
    feasible_rate = feasible_count / attempt_count if attempt_count else 0.0
    infeasible_count = sum(1 for attempt in exact_attempts_until_success if is_exact_milp_failure(attempt))
    infeasible_rate = infeasible_count / attempt_count if attempt_count else 0.0
    students = pd.to_numeric(summary["students"], errors="coerce").fillna(0.0)
    capacity = pd.to_numeric(summary["capacity"], errors="coerce").fillna(0.0)
    seat_surplus = pd.to_numeric(summary["seat_surplus"], errors="coerce").fillna(0.0)
    schools = pd.to_numeric(summary["schools"], errors="coerce").fillna(0.0)
    avg_students = float(diagnostics.get("avg_students", students.mean() if len(students) else 0.0))
    avg_schools = float(diagnostics.get("avg_schools", schools.mean() if len(schools) else 0.0))
    min_surplus = float(seat_surplus.min()) if len(seat_surplus) else 0.0
    max_surplus = float(seat_surplus.max()) if len(seat_surplus) else 0.0
    distance_col = "avg_center_distance_miles"
    max_distance_col = "max_center_distance_miles"
    distance_values = pd.to_numeric(summary[distance_col], errors="coerce").fillna(0.0) if distance_col in summary.columns else pd.Series(dtype=float)
    max_distance_values = pd.to_numeric(summary[max_distance_col], errors="coerce").fillna(0.0) if max_distance_col in summary.columns else pd.Series(dtype=float)
    quality_index = pd.to_numeric(summary["school_quality_index"], errors="coerce").fillna(0.0) if "school_quality_index" in summary.columns else pd.Series(dtype=float)
    diversity_values = (
        pd.to_numeric(summary["diversity_index"], errors="coerce").fillna(0.0)
        if "diversity_index" in summary.columns
        else pd.Series(dtype=float)
    )
    if len(diversity_values):
        diversity_weights = students.clip(lower=0.0)
        if float(diversity_weights.sum()) > 0:
            city_diversity_index = float(np.average(diversity_values, weights=diversity_weights))
        else:
            city_diversity_index = float(diversity_values.mean())
        diversity_abs_deviation = (diversity_values - city_diversity_index).abs()
        diversity_relative_deviation = diversity_abs_deviation / max(abs(city_diversity_index), 1e-9)
        diversity_max_abs_deviation = float(diversity_abs_deviation.max())
        diversity_mean_abs_deviation = float(diversity_abs_deviation.mean())
        diversity_max_relative_deviation = float(diversity_relative_deviation.max())
        diversity_mean_relative_deviation = float(diversity_relative_deviation.mean())
        diversity_all_zones_within_10pct = bool((diversity_relative_deviation <= 0.10 + 1e-12).all())
        diversity_all_zones_within_15pct = bool((diversity_relative_deviation <= 0.15 + 1e-12).all())
    else:
        city_diversity_index = 0.0
        diversity_max_abs_deviation = 0.0
        diversity_mean_abs_deviation = 0.0
        diversity_max_relative_deviation = 0.0
        diversity_mean_relative_deviation = 0.0
        diversity_all_zones_within_10pct = False
        diversity_all_zones_within_15pct = False
    if diversity_all_zones_within_10pct:
        diversity_partition_quality = "excellent"
    elif diversity_all_zones_within_15pct:
        diversity_partition_quality = "good"
    else:
        diversity_partition_quality = "not_good"
    centers = diagnostics.get("centers", [])
    return {
        "city": city,
        "selected_score": float(diagnostics.get("selected_score", math.nan)),
        "outer_search": diagnostics.get("outer_search"),
        "time_limit_rescue_attempted": bool(diagnostics.get("time_limit_rescue_attempted", False)),
        "time_limit_rescue_success": bool(diagnostics.get("time_limit_rescue_success", False)),
        "small_city_one_zone_rescue_attempted": bool(diagnostics.get("small_city_one_zone_rescue_attempted", False)),
        "small_city_one_zone_rescue_success": bool(diagnostics.get("small_city_one_zone_rescue_success", False)),
        "moderate_small_school_rescue_attempted": bool(diagnostics.get("moderate_small_school_rescue_attempted", False)),
        "moderate_small_school_hard_rescue_success": bool(diagnostics.get("moderate_small_school_hard_rescue_success", False)),
        "moderate_small_school_extreme_rescue_success": bool(diagnostics.get("moderate_small_school_extreme_rescue_success", False)),
        "low_schools_per_zone_rescue_attempted": bool(diagnostics.get("low_schools_per_zone_rescue_attempted", False)),
        "low_schools_per_zone_rescue_success": bool(diagnostics.get("low_schools_per_zone_rescue_success", False)),
        "time_limit_zone_increase_rescue_attempted": bool(diagnostics.get("time_limit_zone_increase_rescue_attempted", False)),
        "time_limit_zone_increase_rescue_success": bool(diagnostics.get("time_limit_zone_increase_rescue_success", False)),
        "final_grid_rescue_attempted": bool(diagnostics.get("final_grid_rescue_attempted", False)),
        "final_grid_rescue_success": bool(diagnostics.get("final_grid_rescue_success", False)),
        "rescue_strategy": diagnostics.get("rescue_strategy", "none"),
        "initial_predicted_n_zones": diagnostics.get("initial_predicted_n_zones"),
        "small_city_one_zone_rescue_reason": diagnostics.get("small_city_one_zone_rescue_reason"),
        "small_city_one_zone_direct_max_schools": diagnostics.get("small_city_one_zone_direct_max_schools"),
        "small_city_one_zone_cluster_max_distance_miles": diagnostics.get("small_city_one_zone_cluster_max_distance_miles"),
        "moderate_small_school_rescue_reason": diagnostics.get("moderate_small_school_rescue_reason"),
        "moderate_small_school_max_schools": diagnostics.get("moderate_small_school_max_schools"),
        "moderate_small_school_max_schools_per_zone": diagnostics.get("moderate_small_school_max_schools_per_zone"),
        "predicted_constraint_difficulty_label": diagnostics.get("predicted_constraint_difficulty_label"),
        "initial_schools_per_predicted_zone": diagnostics.get("initial_schools_per_predicted_zone"),
        "low_schools_per_zone_rescue_reason": diagnostics.get("low_schools_per_zone_rescue_reason"),
        "low_schools_per_zone_min_schools_per_zone": diagnostics.get("low_schools_per_zone_min_schools_per_zone"),
        "low_schools_per_zone_rescue_n_zones": diagnostics.get("low_schools_per_zone_rescue_n_zones"),
        "time_limit_zone_increase_rescue_seconds": diagnostics.get("time_limit_zone_increase_rescue_seconds"),
        "time_limit_zone_increase_rescue_candidate_n_zones": diagnostics.get("time_limit_zone_increase_rescue_candidate_n_zones"),
        "time_limit_zone_increase_rescue_n_zones": diagnostics.get("time_limit_zone_increase_rescue_n_zones"),
        "final_grid_rescue_stage": diagnostics.get("final_grid_rescue_stage"),
        "final_grid_rescue_profile": diagnostics.get("final_grid_rescue_profile"),
        "final_grid_rescue_attempt_count": diagnostics.get("final_grid_rescue_attempt_count"),
        "initial_solver_time_limit_seconds": diagnostics.get("initial_solver_time_limit_seconds"),
        "final_solver_time_limit_seconds": diagnostics.get("final_solver_time_limit_seconds"),
        "n_zones": int(len(summary)),
        "centers": json.dumps(centers),
        "runtime_seconds": float(diagnostics.get("runtime_seconds", math.nan)),
        "runtime_scope": diagnostics.get("runtime_scope", ""),
        "attempt_count": attempt_count,
        "infeasible_count": infeasible_count,
        "infeasible_rate": infeasible_rate,
        "total_students": int(round(float(students.sum()))),
        "avg_students_per_zone": avg_students,
        "min_students": int(round(float(students.min()))) if len(students) else 0,
        "max_students": int(round(float(students.max()))) if len(students) else 0,
        "student_gap": float(students.max() - students.min()) if len(students) else 0.0,
        "student_cv": float(students.std(ddof=0) / max(avg_students, 1.0)) if len(students) else 0.0,
        "total_capacity": int(round(float(capacity.sum()))),
        "min_seat_surplus": min_surplus,
        "max_seat_surplus": max_surplus,
        "seat_surplus_gap": max_surplus - min_surplus,
        "max_shortage": float((-seat_surplus.clip(upper=0)).max()) if len(seat_surplus) else 0.0,
        "shortage_zone_count": int((seat_surplus < 0).sum()) if len(seat_surplus) else 0,
        "avg_schools_per_zone": avg_schools,
        "min_schools": int(round(float(schools.min()))) if len(schools) else 0,
        "max_schools": int(round(float(schools.max()))) if len(schools) else 0,
        "school_gap": float(schools.max() - schools.min()) if len(schools) else 0.0,
        "frl_gap": float(summary["FRL_pct"].max() - summary["FRL_pct"].min()) if "FRL_pct" in summary.columns else 0.0,
        "focal_group_gap": float(summary["focal_group_pct"].max() - summary["focal_group_pct"].min()) if "focal_group_pct" in summary.columns else 0.0,
        "diversity_gap": float(summary["diversity_index"].max() - summary["diversity_index"].min()) if "diversity_index" in summary.columns else 0.0,
        "quality_index_gap": float(quality_index.max() - quality_index.min()) if len(quality_index) else 0.0,
        "city_diversity_index": city_diversity_index,
        "diversity_max_abs_deviation_from_city": diversity_max_abs_deviation,
        "diversity_mean_abs_deviation_from_city": diversity_mean_abs_deviation,
        "diversity_max_relative_deviation_from_city": diversity_max_relative_deviation,
        "diversity_mean_relative_deviation_from_city": diversity_mean_relative_deviation,
        "diversity_all_zones_within_10pct": diversity_all_zones_within_10pct,
        "diversity_all_zones_within_15pct": diversity_all_zones_within_15pct,
        "is_good_diversity_partition": diversity_all_zones_within_15pct,
        "diversity_partition_quality": diversity_partition_quality,
        "avg_distance_miles": float(distance_values.mean()) if len(distance_values) else 0.0,
        "max_avg_distance_miles": float(distance_values.max()) if len(distance_values) else 0.0,
        "max_center_distance_miles": float(max_distance_values.max()) if len(max_distance_values) else 0.0,
        "distance_gap_miles": float(max_distance_values.max() - distance_values.min()) if len(distance_values) and len(max_distance_values) else 0.0,
        "disconnected_zone_count": int((pd.to_numeric(summary["connected_components"], errors="coerce").fillna(1) > 1).sum())
        if "connected_components" in summary.columns
        else 0,
        "geometric_multipart_zone_count": int((pd.to_numeric(summary["geometric_parts"], errors="coerce").fillna(1) > 1).sum())
        if "geometric_parts" in summary.columns
        else 0,
        "max_geometric_parts": int(pd.to_numeric(summary["geometric_parts"], errors="coerce").fillna(1).max())
        if "geometric_parts" in summary.columns and len(summary)
        else 1,
        "focal_group_labels": ", ".join(diagnostics.get("focal_group_labels", []) or []) or "none",
    }





















# ---------------------------------------------------------------------------
# ML parameter prediction utilities copied into this runner intentionally.
# This runner is self-contained and does not import the baseline SA runner.

import joblib

MODEL_ROOT = PROJECT_DIR / "Models"
SERVICE_DELIVERY_MODEL_ROOT = MODEL_ROOT / "ServiceDeliveryModel"
MODEL_OUTPUT_ROOT = PROJECT_DIR / "Predictions"
LABEL_RESULTS_PATH = Path(os.environ.get("COMPASS_LABELS_CSV", str(PROJECT_DIR / "Labels.csv")))
FEATURE_CACHE_PATH = Path(os.environ.get("COMPASS_FEATURES_CSV", str(ZONE_OPTIMIZATION_DATA_ROOT / "City_Features.csv")))

PARAMETER_OUTPUT_FIELDS = [
    "annealed_n_zones", "annealed_centers", "annealed_balance_tolerance",
    "derived_balance_B_students", "annealed_cover_distance_factor",
    "derived_cover_distance_miles", "city_distance_scale_miles",
    "annealed_max_shortage_fraction", "derived_max_shortage_allowed",
    "annealed_school_slack_below", "annealed_school_slack_above",
    "derived_min_schools_per_zone", "derived_max_schools_per_zone",
    "annealed_frl_lower", "annealed_frl_upper",
    "annealed_focal_group_lower", "annealed_focal_group_upper",
    "annealed_diversity_index_lower", "annealed_diversity_index_upper",
    "annealed_school_quality_lower", "annealed_school_quality_upper",
]
DIRECT_TARGET_FIELDS = [
    "annealed_n_zones", "annealed_balance_tolerance", "annealed_cover_distance_factor",
    "annealed_max_shortage_fraction", "annealed_school_slack_below",
    "annealed_school_slack_above", "annealed_frl_lower", "annealed_frl_upper",
    "annealed_focal_group_lower", "annealed_focal_group_upper",
    "annealed_diversity_index_lower", "annealed_diversity_index_upper",
    "annealed_school_quality_lower", "annealed_school_quality_upper",
]
DIRECT_CATEGORICAL_LABEL_FIELDS = [
    "spatial_service_difficulty",
    "capacity_tightness",
    "school_distribution_imbalance",
    "demographic_constraint_risk",
    "graph_fragmentation_risk",
]
CONSTRAINT_DIFFICULTY_FIELD = "constraint_difficulty_label"
CATEGORICAL_LABEL_FIELDS = [*DIRECT_CATEGORICAL_LABEL_FIELDS, CONSTRAINT_DIFFICULTY_FIELD]

UNDER_PREDICTION_DANGEROUS_TARGETS = {
    "annealed_balance_tolerance",
    "annealed_cover_distance_factor",
    "annealed_max_shortage_fraction",
    "annealed_school_slack_below",
    "annealed_school_slack_above",
    "annealed_frl_upper",
    "annealed_focal_group_upper",
    "annealed_diversity_index_upper",
    "annealed_school_quality_upper",
}
OVER_PREDICTION_DANGEROUS_TARGETS = {
    "annealed_frl_lower",
    "annealed_focal_group_lower",
    "annealed_diversity_index_lower",
    "annealed_school_quality_lower",
}
DIFFICULTY_RELAXATION_RULES = {
    "Easy": {
        "balance_multiplier": 1.00,
        "balance_add": 0.00,
        "distance_multiplier": 1.00,
        "shortage_multiplier": 1.00,
        "shortage_add": 0.00,
        "slack_add": 0.00,
        "lower_multiplier": 1.00,
        "lower_subtract": 0.00,
        "upper_multiplier": 1.00,
        "upper_add": 0.00,
    },
    "Moderate": {
        "balance_multiplier": 1.20,
        "balance_add": 0.08,
        "distance_multiplier": 1.25,
        "shortage_multiplier": 1.35,
        "shortage_add": 0.05,
        "slack_add": 2.00,
        "lower_multiplier": 0.88,
        "lower_subtract": 0.04,
        "upper_multiplier": 1.22,
        "upper_add": 0.08,
    },
    "Hard": {
        "balance_multiplier": 1.85,
        "balance_add": 0.25,
        "distance_multiplier": 1.80,
        "shortage_multiplier": 1.85,
        "shortage_add": 0.18,
        "slack_add": 6.00,
        "lower_multiplier": 0.60,
        "lower_subtract": 0.16,
        "upper_multiplier": 1.80,
        "upper_add": 0.22,
    },
    "Extreme": {
        "balance_multiplier": 2.20,
        "balance_add": 0.40,
        "distance_multiplier": 2.25,
        "shortage_multiplier": 2.20,
        "shortage_add": 0.25,
        "slack_add": 10.00,
        "lower_multiplier": 0.45,
        "lower_subtract": 0.25,
        "upper_multiplier": 2.20,
        "upper_add": 0.35,
    },
}


ASYMMETRIC_DANGEROUS_ERROR_WEIGHT = 4.0

ASYMMETRIC_SAFE_ERROR_WEIGHT = 1.0

class AsymmetricSquaredErrorObjective:
    """XGBoost objective that penalizes feasibility-tightening errors harder."""

    def __init__(
        self,
        dangerous_direction: str,
        dangerous_weight: float = ASYMMETRIC_DANGEROUS_ERROR_WEIGHT,
        safe_weight: float = ASYMMETRIC_SAFE_ERROR_WEIGHT,
    ) -> None:
        if dangerous_direction not in {"under", "over", "symmetric"}:
            raise ValueError(f"unknown dangerous direction: {dangerous_direction}")
        self.dangerous_direction = dangerous_direction
        self.dangerous_weight = float(dangerous_weight)
        self.safe_weight = float(safe_weight)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        error = y_pred - y_true
        if self.dangerous_direction == "under":
            dangerous = error < 0.0
        elif self.dangerous_direction == "over":
            dangerous = error > 0.0
        else:
            dangerous = np.zeros_like(error, dtype=bool)
        weights = np.where(dangerous, self.dangerous_weight, self.safe_weight)
        grad = 2.0 * weights * error
        hess = np.full_like(error, 2.0, dtype=float) * weights
        return grad, hess

def dangerous_error_direction(target_field: str) -> str:
    if target_field in UNDER_PREDICTION_DANGEROUS_TARGETS:
        return "under"
    if target_field in OVER_PREDICTION_DANGEROUS_TARGETS:
        return "over"
    return "symmetric"

class FeasibilityAwareXGBMultiOutputRegressor:
    """Train one target-specific XGBoost model per annealing parameter."""

    def __init__(
        self,
        target_fields: list[str],
        xgb_regressor_type: type,
        xgb_params: dict[str, object],
    ) -> None:
        self.target_fields = list(target_fields)
        self.xgb_regressor_type = xgb_regressor_type
        self.xgb_params = dict(xgb_params)
        self.models_: dict[str, object] = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | np.ndarray) -> "FeasibilityAwareXGBMultiOutputRegressor":
        y_frame = pd.DataFrame(y, columns=self.target_fields, index=getattr(y, "index", None))
        self.models_ = {}
        for target in self.target_fields:
            params = dict(self.xgb_params)
            params["objective"] = AsymmetricSquaredErrorObjective(dangerous_error_direction(target))
            model = self.xgb_regressor_type(**params)
            model.fit(X, y_frame[target])
            self.models_[target] = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("model has not been fitted")
        predictions = [self.models_[target].predict(X) for target in self.target_fields]
        return np.column_stack(predictions)

def load_model_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def read_typology_labels() -> pd.DataFrame:
    labels = pd.read_csv(LABEL_RESULTS_PATH, dtype=str).fillna("")
    labels["city"] = labels["dataset_code"].astype(str).str.strip()
    for col in DIRECT_CATEGORICAL_LABEL_FIELDS:
        if col not in labels.columns:
            labels[col] = "Unknown"
        labels[col] = labels[col].replace("", "Unknown")
    labels["type_id"] = labels[DIRECT_CATEGORICAL_LABEL_FIELDS].agg("__".join, axis=1)
    return labels[["city", *DIRECT_CATEGORICAL_LABEL_FIELDS, "type_id"]].drop_duplicates("city")


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = v.notna() & (w > 0)
    if not mask.any():
        return float(v.mean()) if v.notna().any() else 0.0
    return float(np.average(v[mask], weights=w[mask]))


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    mean = weighted_mean(values, weights)
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = v.notna() & (w > 0)
    if not mask.any():
        return 0.0
    return float(math.sqrt(np.average((v[mask] - mean) ** 2, weights=w[mask])))


def add_numeric_stats(row: dict[str, Any], prefix: str, series: pd.Series, weights: Optional[pd.Series] = None) -> None:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        for suffix in ["sum", "mean", "std", "min", "max", "q25", "q75"]:
            row[f"{prefix}_{suffix}"] = 0.0
        if weights is not None:
            row[f"{prefix}_weighted_mean"] = 0.0
            row[f"{prefix}_weighted_std"] = 0.0
        return
    row[f"{prefix}_sum"] = float(x.sum())
    row[f"{prefix}_mean"] = float(x.mean())
    row[f"{prefix}_std"] = float(x.std(ddof=0))
    row[f"{prefix}_min"] = float(x.min())
    row[f"{prefix}_max"] = float(x.max())
    row[f"{prefix}_q25"] = float(x.quantile(0.25))
    row[f"{prefix}_q75"] = float(x.quantile(0.75))
    if weights is not None:
        row[f"{prefix}_weighted_mean"] = weighted_mean(series, weights)
        row[f"{prefix}_weighted_std"] = weighted_std(series, weights)


def load_city_ml_context(city: str) -> dict[str, Any]:
    paths = resolve_city_paths(city, ZONE_OPTIMIZATION_DATA_ROOT)
    gdf = load_optimization_units(paths.block_source, paths.school_csv, paths.graph_json, city, 0.03, 0.75, 0.25, 1.10, 0.05)
    graph = build_adjacency_graph(gdf)
    milp_cache = build_milp_cache(gdf, graph)
    annealing = AnnealingConfig()
    zone_min, zone_max = annealed_zone_bounds(gdf, annealing)
    school_count = int(round(float(gdf["school_count"].sum())))
    positive_school_units = int((pd.to_numeric(gdf["school_count"], errors="coerce").fillna(0.0) > 0).sum())
    zone_max = min(zone_max, max(1, positive_school_units))
    zone_min = min(max(1, zone_min), zone_max)
    return {"city": city, "paths": paths, "gdf": gdf, "graph": graph, "milp_cache": milp_cache, "zone_min": zone_min, "zone_max": zone_max, "school_count": school_count, "unit_count": int(len(gdf)), "distance_scale_miles": city_distance_scale_miles(gdf)}


def city_feature_row(city: str) -> dict[str, Any]:
    if FEATURE_CACHE_PATH.exists():
        cached = pd.read_csv(FEATURE_CACHE_PATH)
        if "city" in cached.columns:
            match = cached[cached["city"].astype(str).eq(str(city))]
            if not match.empty:
                return match.iloc[0].to_dict()
    context = load_city_ml_context(city)
    gdf = context["gdf"]
    graph = context["graph"]
    milp_cache = context["milp_cache"]
    state, _, city_name_raw = str(city).partition("-")
    students = pd.to_numeric(gdf.get("students", 0.0), errors="coerce").fillna(0.0)
    row: dict[str, Any] = {
        "city": city, "state": state, "city_name": city_name_raw.replace("_", " "),
        "unit_count": context["unit_count"], "graph_node_count": int(graph.number_of_nodes()),
        "graph_edge_count": int(graph.number_of_edges()),
        "graph_component_count": int(nx.number_connected_components(graph)) if graph.number_of_nodes() else 0,
        "graph_density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        "graph_isolate_count": int(len(list(nx.isolates(graph)))) if graph.number_of_nodes() else 0,
        "total_students": float(students.sum()), "total_schools": float(gdf["school_count"].sum()),
        "school_unit_count": int((pd.to_numeric(gdf["school_count"], errors="coerce").fillna(0.0) > 0).sum()),
        "zone_min": context["zone_min"], "zone_max": context["zone_max"],
        "city_distance_scale_miles": context["distance_scale_miles"],
        "milp_unit_count": int(len(milp_cache["units"])),
        "milp_neighbor_link_count": int(sum(len(v) for v in milp_cache["neighbors"].values())),
    }
    add_numeric_stats(row, "graph_degree", pd.Series(dict(graph.degree())).reindex(gdf["unit_id"].astype(int), fill_value=0))
    for col in ["students", "school_count", "block_ele_cap", "FRL_pct", "focal_group_pct", "diversity_index", "school_quality_index", "school_quality_weight"]:
        if col in gdf.columns:
            add_numeric_stats(row, col, gdf[col], students)
    row["capacity_student_ratio"] = float(pd.to_numeric(gdf.get("block_ele_cap", 0.0), errors="coerce").fillna(0.0).sum()) / max(float(students.sum()), 1e-9)
    for key, value in asdict(AnnealingConfig()).items():
        row[f"annealing_config_{key}"] = value if value is not None else -1
    policy = PolicyWeights()
    for key, value in asdict(policy).items():
        if key != "score_weights":
            row[f"policy_weight_{key}"] = value
    for key, value in asdict(policy.score_weights).items():
        row[f"score_weight_{key}"] = value
    return row


def retained_city_codes_for_ml() -> list[str]:
    return available_city_codes(ZONE_OPTIMIZATION_DATA_ROOT)


def predict_constraint_difficulty_label(model_bundle: dict[str, Any], feature_row: dict[str, Any]) -> str:
    classifier = model_bundle.get("difficulty_classifier")
    if classifier is None:
        return str(feature_row.get(CONSTRAINT_DIFFICULTY_FIELD) or "Moderate")
    numeric_cols = list(model_bundle.get("numeric_feature_columns") or [])
    if not numeric_cols:
        return str(feature_row.get(CONSTRAINT_DIFFICULTY_FIELD) or "Moderate")
    row = pd.DataFrame([feature_row])
    for col in numeric_cols:
        if col not in row.columns:
            row[col] = 0.0
    try:
        label = classifier.predict(row[numeric_cols])[0]
    except Exception:
        label = feature_row.get(CONSTRAINT_DIFFICULTY_FIELD) or "Moderate"
    label = str(label or "Moderate")
    return label if label in DIFFICULTY_RELAXATION_RULES else "Moderate"


def apply_difficulty_parameter_relaxation(predicted_params: dict[str, Any], difficulty_label: str) -> dict[str, Any]:
    label = difficulty_label if difficulty_label in DIFFICULTY_RELAXATION_RULES else "Moderate"
    rules = DIFFICULTY_RELAXATION_RULES[label]
    adjusted = dict(predicted_params)

    def number(key: str, default: float = 0.0) -> float:
        try:
            value = float(adjusted.get(key, default))
        except Exception:
            value = default
        return value if np.isfinite(value) else default

    if "annealed_balance_tolerance" in adjusted:
        adjusted["annealed_balance_tolerance"] = min(
            2.5,
            max(0.0, number("annealed_balance_tolerance") * rules["balance_multiplier"] + rules["balance_add"]),
        )
    if "annealed_cover_distance_factor" in adjusted:
        adjusted["annealed_cover_distance_factor"] = max(
            0.0,
            number("annealed_cover_distance_factor") * rules["distance_multiplier"],
        )
    if "annealed_max_shortage_fraction" in adjusted:
        adjusted["annealed_max_shortage_fraction"] = min(
            1.5,
            max(0.0, number("annealed_max_shortage_fraction") * rules["shortage_multiplier"] + rules["shortage_add"]),
        )
    for col in ("annealed_school_slack_below", "annealed_school_slack_above"):
        if col in adjusted:
            adjusted[col] = max(0.0, number(col) + rules["slack_add"])
    for col in OVER_PREDICTION_DANGEROUS_TARGETS:
        if col in adjusted:
            adjusted[col] = max(0.0, number(col) * rules["lower_multiplier"] - rules["lower_subtract"])
    for col in {
        "annealed_frl_upper",
        "annealed_focal_group_upper",
        "annealed_diversity_index_upper",
        "annealed_school_quality_upper",
    }:
        if col in adjusted:
            adjusted[col] = max(0.0, number(col) * rules["upper_multiplier"] + rules["upper_add"])
    return adjusted


_GLOBAL_BASELINE_MODEL_BUNDLE: Optional[dict[str, Any]] = None




def load_global_baseline_model_bundle() -> Optional[dict[str, Any]]:
    global _GLOBAL_BASELINE_MODEL_BUNDLE
    if _GLOBAL_BASELINE_MODEL_BUNDLE is not None:
        return _GLOBAL_BASELINE_MODEL_BUNDLE
    if not GLOBAL_BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Global baseline service model is required for feasibility-safe calibration: {GLOBAL_BASELINE_MODEL_PATH}"
        )
    _GLOBAL_BASELINE_MODEL_BUNDLE = load_model_bundle(GLOBAL_BASELINE_MODEL_PATH)
    return _GLOBAL_BASELINE_MODEL_BUNDLE


def predict_direct_without_typology_calibration(model_bundle: dict[str, Any], feature_row: dict[str, Any]) -> dict[str, Any]:
    model = model_bundle["model"]
    feature_columns = list(model_bundle["feature_columns"])
    target_fields = list(model_bundle.get("target_fields") or DIRECT_TARGET_FIELDS)
    row = pd.DataFrame([feature_row])
    for col in feature_columns:
        if col not in row.columns:
            row[col] = "Unknown" if col in CATEGORICAL_LABEL_FIELDS else 0.0
    pred = model.predict(row[feature_columns])[0]
    return dict(zip(target_fields, pred))


def apply_global_feasibility_safe_calibration(predicted_params: dict[str, Any], global_params: dict[str, Any]) -> dict[str, Any]:
    calibrated = dict(predicted_params)
    for col in DIRECT_TARGET_FIELDS:
        if col not in calibrated or col not in global_params:
            continue
        try:
            predicted_value = float(calibrated[col])
            global_value = float(global_params[col])
        except Exception:
            continue
        if not np.isfinite(predicted_value) or not np.isfinite(global_value):
            continue
        if col == "annealed_n_zones":
            calibrated[col] = global_value
        elif col in UNDER_PREDICTION_DANGEROUS_TARGETS:
            calibrated[col] = max(predicted_value, global_value)
        elif col in OVER_PREDICTION_DANGEROUS_TARGETS:
            calibrated[col] = min(predicted_value, global_value)
    calibrated["feasibility_safe_calibration"] = "global_service_baseline_only_loosen"
    return calibrated


def model_predict_direct(model_bundle: dict[str, Any], feature_row: dict[str, Any]) -> dict[str, Any]:
    model = model_bundle["model"]
    feature_columns = list(model_bundle["feature_columns"])
    target_fields = list(model_bundle.get("target_fields") or DIRECT_TARGET_FIELDS)
    difficulty_label = predict_constraint_difficulty_label(model_bundle, feature_row)
    feature_row[CONSTRAINT_DIFFICULTY_FIELD] = difficulty_label
    for col in CATEGORICAL_LABEL_FIELDS:
        if col not in feature_row or str(feature_row.get(col, "")).strip() == "":
            feature_row[col] = "Unknown" if col != CONSTRAINT_DIFFICULTY_FIELD else difficulty_label
    feature_row["type_id"] = "__".join(str(feature_row.get(col) or "Unknown") for col in CATEGORICAL_LABEL_FIELDS)
    row = pd.DataFrame([feature_row])
    for col in feature_columns:
        if col not in row.columns:
            row[col] = "Unknown" if col in CATEGORICAL_LABEL_FIELDS else 0.0
    pred = model.predict(row[feature_columns])[0]
    predicted = dict(zip(target_fields, pred))
    predicted = apply_difficulty_parameter_relaxation(predicted, difficulty_label)
    global_bundle = model_bundle.get("global_baseline_model_bundle")
    if global_bundle is None:
        if model_bundle.get("fold") is not None:
            raise ValueError("Fold prediction requires its paired training-fold baseline model")
        global_bundle = load_global_baseline_model_bundle()
    elif model_bundle.get("fold") != global_bundle.get("fold"):
        raise ValueError("Typology and calibration baseline folds do not match")
    global_prediction = predict_direct_without_typology_calibration(global_bundle, feature_row)
    predicted = apply_global_feasibility_safe_calibration(predicted, global_prediction)
    predicted["global_baseline_model_path"] = (
        "embedded_paired_baseline" if "global_baseline_model_bundle" in model_bundle
        else str(GLOBAL_BASELINE_MODEL_PATH)
    )
    predicted[CONSTRAINT_DIFFICULTY_FIELD] = difficulty_label
    return predicted


def clamp_predicted_value(value: Any, low: float, high: float) -> float:
    try:
        value = float(value)
    except Exception:
        value = low
    if not np.isfinite(value):
        value = low
    return max(float(low), min(float(high), value))


def safe_int_predicted(value: Any, low: int, high: int) -> int:
    return int(round(clamp_predicted_value(value, low, high)))


def generate_predicted_centers(gdf: gpd.GeoDataFrame, graph: nx.Graph, n_zones: int) -> list[int]:
    try:
        return [int(v) for v in seed_centers(gdf, int(n_zones), random.Random(20260601), graph)]
    except Exception:
        pool = center_candidate_pool(gdf, int(n_zones))
        if len(pool) < int(n_zones):
            pool = gdf["unit_id"].astype(int).tolist()
        return [int(v) for v in pool[: int(n_zones)]]


def postprocess_predicted_params(predicted_params: dict[str, Any], gdf: gpd.GeoDataFrame, graph: nx.Graph, distance_scale: float) -> dict[str, Any]:
    school_count = int(round(float(pd.to_numeric(gdf["school_count"], errors="coerce").fillna(0.0).sum())))
    unit_count = int(len(gdf))
    max_k = max(1, min(school_count if school_count > 0 else unit_count, unit_count))
    n_zones = safe_int_predicted(predicted_params.get("annealed_n_zones", 1), 1, max_k)
    avg_schools = school_count / max(n_zones, 1)
    school_slack_below = safe_int_predicted(predicted_params.get("annealed_school_slack_below", 1), 0, max(20, school_count))
    school_slack_above = safe_int_predicted(predicted_params.get("annealed_school_slack_above", 1), 0, max(20, school_count))
    min_schools = max(1, math.floor(avg_schools) - school_slack_below)
    max_schools = max(min_schools, math.ceil(avg_schools) + school_slack_above)
    students_total = float(pd.to_numeric(gdf["students"], errors="coerce").fillna(0.0).sum())

    def positive_pair(lower_key: str, upper_key: str) -> tuple[float, float]:
        lower = max(0.0, float(predicted_params.get(lower_key, 0.0)))
        upper = max(0.0, float(predicted_params.get(upper_key, lower)))
        return tuple(sorted([lower, upper]))

    frl_lower, frl_upper = positive_pair("annealed_frl_lower", "annealed_frl_upper")
    focal_lower, focal_upper = positive_pair("annealed_focal_group_lower", "annealed_focal_group_upper")
    div_lower, div_upper = positive_pair("annealed_diversity_index_lower", "annealed_diversity_index_upper")
    quality_lower, quality_upper = positive_pair("annealed_school_quality_lower", "annealed_school_quality_upper")
    balance_tolerance = clamp_predicted_value(predicted_params.get("annealed_balance_tolerance", 0.55), 0.0, 2.5)
    cover_distance_factor = max(0.0, float(predicted_params.get("annealed_cover_distance_factor", 1.0)))
    shortage_fraction = clamp_predicted_value(predicted_params.get("annealed_max_shortage_fraction", 0.0), 0.0, 1.5)
    return {
        "annealed_n_zones": n_zones,
        "annealed_centers": json.dumps(generate_predicted_centers(gdf, graph, n_zones)),
        "annealed_balance_tolerance": balance_tolerance,
        "derived_balance_B_students": balance_tolerance * (students_total / max(n_zones, 1)),
        "annealed_cover_distance_factor": cover_distance_factor,
        "derived_cover_distance_miles": cover_distance_factor * float(distance_scale),
        "city_distance_scale_miles": float(distance_scale),
        "annealed_max_shortage_fraction": shortage_fraction,
        "derived_max_shortage_allowed": shortage_fraction * (students_total / max(n_zones, 1)),
        "annealed_school_slack_below": school_slack_below,
        "annealed_school_slack_above": school_slack_above,
        "derived_min_schools_per_zone": min_schools,
        "derived_max_schools_per_zone": max_schools,
        "annealed_frl_lower": frl_lower,
        "annealed_frl_upper": frl_upper,
        "annealed_focal_group_lower": focal_lower,
        "annealed_focal_group_upper": focal_upper,
        "annealed_diversity_index_lower": div_lower,
        "annealed_diversity_index_upper": div_upper,
        "annealed_school_quality_lower": quality_lower,
        "annealed_school_quality_upper": quality_upper,
    }


def exact_config_from_predicted_params(params: dict[str, Any], solver_time_limit_seconds: Optional[int]) -> ExactZoneConfig:
    policy = PolicyWeights()
    return ExactZoneConfig(
        n_zones=int(params["annealed_n_zones"]),
        balance_tolerance=float(params["annealed_balance_tolerance"]),
        min_schools_per_zone=int(params["derived_min_schools_per_zone"]),
        max_schools_per_zone=int(params["derived_max_schools_per_zone"]),
        cover_distance_miles=float(params["derived_cover_distance_miles"]),
        frl_bounds=(float(params["annealed_frl_lower"]), float(params["annealed_frl_upper"])),
        focal_group_bounds=(float(params["annealed_focal_group_lower"]), float(params["annealed_focal_group_upper"])),
        diversity_index_bounds=(float(params["annealed_diversity_index_lower"]), float(params["annealed_diversity_index_upper"])),
        school_quality_bounds=(float(params["annealed_school_quality_lower"]), float(params["annealed_school_quality_upper"])),
        max_shortage=float(params["derived_max_shortage_allowed"]),
        objective_distance_weight=policy.objective_distance_weight,
        objective_distance_power=policy.objective_distance_power,
        objective_diversity_weight=policy.objective_diversity_weight,
        objective_quality_weight=policy.objective_quality_weight,
        objective_min_surplus_weight=policy.objective_min_surplus_weight,
        solver_time_limit_seconds=None if solver_time_limit_seconds is None else int(solver_time_limit_seconds),
    )


def selected_state_from_predicted_params(params: dict[str, Any], centers: list[int]) -> dict[str, Any]:
    return {
        "n_zones": params["annealed_n_zones"], "centers": centers,
        "balance_tolerance": params["annealed_balance_tolerance"], "balance_B_students": params["derived_balance_B_students"],
        "cover_distance_factor": params["annealed_cover_distance_factor"], "cover_distance_miles": params["derived_cover_distance_miles"],
        "max_shortage_fraction": params["annealed_max_shortage_fraction"], "max_shortage": params["derived_max_shortage_allowed"],
        "school_slack_below": params["annealed_school_slack_below"], "school_slack_above": params["annealed_school_slack_above"],
        "min_schools_per_zone": params["derived_min_schools_per_zone"], "max_schools_per_zone": params["derived_max_schools_per_zone"],
        "frl_lower": params["annealed_frl_lower"], "frl_upper": params["annealed_frl_upper"],
        "frl_bounds": [params["annealed_frl_lower"], params["annealed_frl_upper"]],
        "focal_group_lower": params["annealed_focal_group_lower"], "focal_group_upper": params["annealed_focal_group_upper"],
        "focal_group_bounds": [params["annealed_focal_group_lower"], params["annealed_focal_group_upper"]],
        "diversity_index_lower": params["annealed_diversity_index_lower"], "diversity_index_upper": params["annealed_diversity_index_upper"],
        "diversity_index_bounds": [params["annealed_diversity_index_lower"], params["annealed_diversity_index_upper"]],
        "school_quality_lower": params["annealed_school_quality_lower"], "school_quality_upper": params["annealed_school_quality_upper"],
        "school_quality_bounds": [params["annealed_school_quality_lower"], params["annealed_school_quality_upper"]],
        "city_distance_scale_miles": params["city_distance_scale_miles"],
    }


def is_time_limit_milp_error(exc: BaseException) -> bool:
    return "time limit" in str(exc).lower()


def is_infeasible_milp_error(exc: BaseException) -> bool:
    return "infeasible" in str(exc).lower()


SMALL_CITY_ONE_ZONE_DIRECT_MAX_SCHOOLS = 3
SMALL_CITY_ONE_ZONE_CLUSTER_MAX_DISTANCE_MILES = 0.5
MODERATE_SMALL_SCHOOL_RESCUE_MAX_SCHOOLS = 10
MODERATE_SMALL_SCHOOL_RESCUE_MAX_SCHOOLS_PER_ZONE = 3.0
# Rescue 4: if an infeasible predicted split leaves too few schools per zone,
# lower the predicted zone count until the average reaches this threshold.
LOW_SCHOOLS_PER_ZONE_RESCUE_MIN_SCHOOLS_PER_ZONE = 10.0
LOW_SCHOOLS_PER_ZONE_RESCUE_MIN_ZONES = 2
TIME_LIMIT_RESCUE_SECONDS = 600
TIME_LIMIT_ZONE_INCREASE_STEP_FRACTION = 0.10
TIME_LIMIT_ZONE_INCREASE_MAX_FRACTION = 0.50
TIME_LIMIT_ZONE_INCREASE_MAX_CANDIDATES = 5

DIFFICULTY_RESCUE_RELAXATION_RULES = {
    "Hard": {
        "balance_multiplier": 1.85,
        "balance_add": 0.25,
        "distance_multiplier": 1.80,
        "shortage_multiplier": 1.85,
        "shortage_add": 0.18,
        "slack_add": 6.00,
        "lower_multiplier": 0.60,
        "lower_subtract": 0.16,
        "upper_multiplier": 1.80,
        "upper_add": 0.22,
    },
    "Extreme": {
        "balance_multiplier": 2.20,
        "balance_add": 0.40,
        "distance_multiplier": 2.25,
        "shortage_multiplier": 2.20,
        "shortage_add": 0.25,
        "slack_add": 10.00,
        "lower_multiplier": 0.45,
        "lower_subtract": 0.25,
        "upper_multiplier": 2.20,
        "upper_add": 0.35,
    },
}


def school_units_are_clustered_for_one_zone_rescue(context: dict[str, Any]) -> bool:
    gdf = context["gdf"]
    graph = context["graph"]
    school_units = gdf[pd.to_numeric(gdf["school_count"], errors="coerce").fillna(0.0) > 0].copy()
    if len(school_units) <= 1:
        return True
    school_unit_ids = [int(value) for value in school_units["unit_id"].tolist()]
    for component in nx.connected_components(graph):
        if set(school_unit_ids).issubset({int(value) for value in component}):
            break
    else:
        return False
    projected = school_units.to_crs(3857)
    centroids = list(projected.geometry.centroid)
    max_pairwise_distance = 0.0
    for i, left in enumerate(centroids):
        for right in centroids[i + 1 :]:
            max_pairwise_distance = max(max_pairwise_distance, float(left.distance(right) / 1609.344))
    city_scale = max(float(context.get("distance_scale_miles") or 0.0), 0.25)
    distance_threshold = min(
        SMALL_CITY_ONE_ZONE_CLUSTER_MAX_DISTANCE_MILES,
        max(0.25, 0.25 * city_scale),
    )
    return max_pairwise_distance <= distance_threshold


def small_city_one_zone_rescue_reason(context: dict[str, Any], params: dict[str, Any]) -> str:
    school_count = int(context.get("school_count") or 0)
    predicted_n_zones = int(params.get("annealed_n_zones") or 1)
    if predicted_n_zones <= 1:
        return "already_one_zone"
    if school_count < 2:
        return "school_count_too_small"
    if school_count <= SMALL_CITY_ONE_ZONE_DIRECT_MAX_SCHOOLS:
        return "direct_small_school_count"
    if school_units_are_clustered_for_one_zone_rescue(context):
        return "all_schools_tiny_area"
    return "schools_not_tiny_area"


def small_city_one_zone_rescue_eligible(context: dict[str, Any], params: dict[str, Any]) -> bool:
    return small_city_one_zone_rescue_reason(context, params) in {
        "direct_small_school_count",
        "all_schools_tiny_area",
    }


def one_zone_rescue_params(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    rescued = dict(params)
    gdf = context["gdf"]
    graph = context["graph"]
    distance_scale = max(float(context.get("distance_scale_miles") or 0.0), 1e-9)
    school_count = max(1, int(context.get("school_count") or 1))
    students_total = float(pd.to_numeric(gdf["students"], errors="coerce").fillna(0.0).sum())
    capacity_series = gdf["block_ele_cap"] if "block_ele_cap" in gdf.columns else pd.Series(0.0, index=gdf.index)
    capacity_total = float(pd.to_numeric(capacity_series, errors="coerce").fillna(0.0).sum())
    centers = generate_predicted_centers(gdf, graph, 1)
    distances = centroid_distances_to_centers(gdf, centers)
    cover_distance_miles = max(float(rescued.get("derived_cover_distance_miles") or 0.0), max(distances.values()) if distances else 0.0)
    total_shortage = max(0.0, students_total - capacity_total)
    max_shortage = max(float(rescued.get("derived_max_shortage_allowed") or 0.0), total_shortage)

    def include_city_average(lower_key: str, upper_key: str) -> None:
        lower = float(rescued.get(lower_key) or 0.0)
        upper = float(rescued.get(upper_key) or lower)
        rescued[lower_key] = min(lower, 1.0)
        rescued[upper_key] = max(upper, 1.0)

    rescued.update({
        "annealed_n_zones": 1,
        "annealed_centers": json.dumps([int(c) for c in centers]),
        "derived_balance_B_students": max(float(rescued.get("derived_balance_B_students") or 0.0), students_total),
        "annealed_cover_distance_factor": cover_distance_miles / distance_scale,
        "derived_cover_distance_miles": cover_distance_miles,
        "city_distance_scale_miles": float(context.get("distance_scale_miles") or distance_scale),
        "annealed_max_shortage_fraction": max_shortage / max(students_total, 1e-9),
        "derived_max_shortage_allowed": max_shortage,
        "annealed_school_slack_below": max(int(rescued.get("annealed_school_slack_below") or 0), school_count),
        "annealed_school_slack_above": max(int(rescued.get("annealed_school_slack_above") or 0), school_count),
        "derived_min_schools_per_zone": 1,
        "derived_max_schools_per_zone": school_count,
    })
    for lower_key, upper_key in [
        ("annealed_frl_lower", "annealed_frl_upper"),
        ("annealed_focal_group_lower", "annealed_focal_group_upper"),
        ("annealed_diversity_index_lower", "annealed_diversity_index_upper"),
        ("annealed_school_quality_lower", "annealed_school_quality_upper"),
    ]:
        include_city_average(lower_key, upper_key)
    return rescued


def predicted_constraint_difficulty_label(prediction: dict[str, Any]) -> str:
    label = str(prediction.get("constraint_difficulty_label") or "").strip()
    return label if label else "Unknown"


def initial_schools_per_predicted_zone(context: dict[str, Any], params: dict[str, Any]) -> float:
    school_count = float(context.get("school_count") or 0.0)
    predicted_n_zones = max(1, int(params.get("annealed_n_zones") or 1))
    return school_count / predicted_n_zones


def moderate_small_school_rescue_reason(
    context: dict[str, Any],
    params: dict[str, Any],
    prediction: dict[str, Any],
) -> str:
    difficulty_label = predicted_constraint_difficulty_label(prediction)
    school_count = int(context.get("school_count") or 0)
    schools_per_zone = initial_schools_per_predicted_zone(context, params)
    if difficulty_label != "Moderate":
        return "predicted_difficulty_not_moderate"
    if school_count > MODERATE_SMALL_SCHOOL_RESCUE_MAX_SCHOOLS:
        return "school_count_not_small"
    if schools_per_zone > MODERATE_SMALL_SCHOOL_RESCUE_MAX_SCHOOLS_PER_ZONE:
        return "schools_per_zone_not_small"
    return "moderate_small_school_low_schools_per_zone"


def moderate_small_school_rescue_eligible(
    context: dict[str, Any],
    params: dict[str, Any],
    prediction: dict[str, Any],
) -> bool:
    return (
        moderate_small_school_rescue_reason(context, params, prediction)
        == "moderate_small_school_low_schools_per_zone"
    )


def difficulty_relaxed_rescue_params(
    params: dict[str, Any],
    context: dict[str, Any],
    difficulty_label: str,
) -> dict[str, Any]:
    """Relax an already postprocessed parameter set using Hard/Extreme rules."""
    if difficulty_label not in DIFFICULTY_RESCUE_RELAXATION_RULES:
        raise ValueError(f"Unsupported difficulty rescue label: {difficulty_label}")
    rule = DIFFICULTY_RESCUE_RELAXATION_RULES[difficulty_label]
    rescued = dict(params)
    school_count = max(1, int(context.get("school_count") or 1))
    n_zones = max(1, int(rescued.get("annealed_n_zones") or 1))
    students_total = float(pd.to_numeric(context["gdf"]["students"], errors="coerce").fillna(0.0).sum())
    distance_scale = float(context.get("distance_scale_miles") or rescued.get("city_distance_scale_miles") or 0.0)

    rescued["annealed_balance_tolerance"] = clamp_predicted_value(
        float(rescued.get("annealed_balance_tolerance") or 0.0) * rule["balance_multiplier"] + rule["balance_add"],
        0.0,
        2.5,
    )
    rescued["derived_balance_B_students"] = rescued["annealed_balance_tolerance"] * (
        students_total / max(n_zones, 1)
    )
    rescued["annealed_cover_distance_factor"] = max(
        0.0,
        float(rescued.get("annealed_cover_distance_factor") or 0.0) * rule["distance_multiplier"],
    )
    rescued["derived_cover_distance_miles"] = rescued["annealed_cover_distance_factor"] * max(distance_scale, 0.0)
    rescued["annealed_max_shortage_fraction"] = clamp_predicted_value(
        float(rescued.get("annealed_max_shortage_fraction") or 0.0) * rule["shortage_multiplier"] + rule["shortage_add"],
        0.0,
        1.5,
    )
    rescued["derived_max_shortage_allowed"] = rescued["annealed_max_shortage_fraction"] * (
        students_total / max(n_zones, 1)
    )

    slack_cap = max(20, school_count)
    rescued["annealed_school_slack_below"] = int(
        round(clamp_predicted_value(float(rescued.get("annealed_school_slack_below") or 0.0) + rule["slack_add"], 0, slack_cap))
    )
    rescued["annealed_school_slack_above"] = int(
        round(clamp_predicted_value(float(rescued.get("annealed_school_slack_above") or 0.0) + rule["slack_add"], 0, slack_cap))
    )
    avg_schools = school_count / max(n_zones, 1)
    rescued["derived_min_schools_per_zone"] = max(
        1,
        math.floor(avg_schools) - int(rescued["annealed_school_slack_below"]),
    )
    rescued["derived_max_schools_per_zone"] = max(
        int(rescued["derived_min_schools_per_zone"]),
        math.ceil(avg_schools) + int(rescued["annealed_school_slack_above"]),
    )

    for lower_key in [
        "annealed_frl_lower",
        "annealed_focal_group_lower",
        "annealed_diversity_index_lower",
        "annealed_school_quality_lower",
    ]:
        rescued[lower_key] = max(
            0.0,
            float(rescued.get(lower_key) or 0.0) * rule["lower_multiplier"] - rule["lower_subtract"],
        )
    for upper_key in [
        "annealed_frl_upper",
        "annealed_focal_group_upper",
        "annealed_diversity_index_upper",
        "annealed_school_quality_upper",
    ]:
        rescued[upper_key] = max(
            0.0,
            float(rescued.get(upper_key) or 0.0) * rule["upper_multiplier"] + rule["upper_add"],
        )
    for lower_key, upper_key in [
        ("annealed_frl_lower", "annealed_frl_upper"),
        ("annealed_focal_group_lower", "annealed_focal_group_upper"),
        ("annealed_diversity_index_lower", "annealed_diversity_index_upper"),
        ("annealed_school_quality_lower", "annealed_school_quality_upper"),
    ]:
        if rescued[upper_key] < rescued[lower_key]:
            rescued[upper_key] = rescued[lower_key]
    return rescued


def low_schools_per_zone_rescue_target_n_zones(context: dict[str, Any], params: dict[str, Any]) -> Optional[int]:
    candidates = low_schools_per_zone_rescue_candidate_n_zones(context, params)
    return candidates[0] if candidates else None


def low_schools_per_zone_rescue_candidate_n_zones(context: dict[str, Any], params: dict[str, Any]) -> list[int]:
    school_count = int(context.get("school_count") or 0)
    current_n_zones = int(params.get("annealed_n_zones") or 1)
    if current_n_zones <= LOW_SCHOOLS_PER_ZONE_RESCUE_MIN_ZONES:
        return []
    current_schools_per_zone = school_count / max(current_n_zones, 1)
    if current_schools_per_zone >= LOW_SCHOOLS_PER_ZONE_RESCUE_MIN_SCHOOLS_PER_ZONE:
        return []
    strict_target = int(math.floor(school_count / LOW_SCHOOLS_PER_ZONE_RESCUE_MIN_SCHOOLS_PER_ZONE))
    strict_target = max(LOW_SCHOOLS_PER_ZONE_RESCUE_MIN_ZONES, strict_target)
    strict_target = min(strict_target, current_n_zones - 1)
    if strict_target >= current_n_zones:
        return []
    # Try gradual reductions first. A city can become feasible before it reaches
    # the requested schools-per-zone threshold; jumping straight to strict_target
    # can over-merge zones and create new infeasibility.
    return list(range(current_n_zones - 1, strict_target - 1, -1))


def low_schools_per_zone_rescue_reason(context: dict[str, Any], params: dict[str, Any]) -> str:
    school_count = int(context.get("school_count") or 0)
    current_n_zones = int(params.get("annealed_n_zones") or 1)
    current_schools_per_zone = school_count / max(current_n_zones, 1)
    if current_n_zones <= LOW_SCHOOLS_PER_ZONE_RESCUE_MIN_ZONES:
        return "already_at_minimum_zone_count"
    if current_schools_per_zone >= LOW_SCHOOLS_PER_ZONE_RESCUE_MIN_SCHOOLS_PER_ZONE:
        return "schools_per_zone_not_low"
    target = low_schools_per_zone_rescue_target_n_zones(context, params)
    if target is None:
        return "no_valid_reduced_zone_count"
    return "low_schools_per_zone"


def low_schools_per_zone_rescue_eligible(context: dict[str, Any], params: dict[str, Any]) -> bool:
    return bool(low_schools_per_zone_rescue_candidate_n_zones(context, params))


def reduced_zone_count_rescue_params(params: dict[str, Any], context: dict[str, Any], target_n_zones: int) -> dict[str, Any]:
    rescued = dict(params)
    gdf = context["gdf"]
    graph = context["graph"]
    school_count = max(1, int(context.get("school_count") or 1))
    students_total = float(pd.to_numeric(gdf["students"], errors="coerce").fillna(0.0).sum())
    n_zones = int(target_n_zones)
    centers = generate_predicted_centers(gdf, graph, n_zones)
    avg_schools = school_count / max(n_zones, 1)
    slack_below = int(rescued.get("annealed_school_slack_below") or 0)
    slack_above = int(rescued.get("annealed_school_slack_above") or 0)
    rescued.update({
        "annealed_n_zones": n_zones,
        "annealed_centers": json.dumps([int(c) for c in centers]),
        "derived_balance_B_students": float(rescued.get("annealed_balance_tolerance") or 0.0) * (
            students_total / max(n_zones, 1)
        ),
        "derived_max_shortage_allowed": float(rescued.get("annealed_max_shortage_fraction") or 0.0) * (
            students_total / max(n_zones, 1)
        ),
        "derived_min_schools_per_zone": max(1, math.floor(avg_schools) - slack_below),
    })
    rescued["derived_max_schools_per_zone"] = max(
        int(rescued["derived_min_schools_per_zone"]),
        math.ceil(avg_schools) + slack_above,
    )
    return rescued


def time_limit_zone_increase_candidate_n_zones(context: dict[str, Any], params: dict[str, Any]) -> list[int]:
    school_count = int(context.get("school_count") or 0)
    unit_count = int(len(context["gdf"]))
    max_k = max(1, min(school_count if school_count > 0 else unit_count, unit_count))
    current_n_zones = max(1, int(params.get("annealed_n_zones") or 1))
    if current_n_zones >= max_k:
        return []
    candidates: list[int] = []
    step_count = int(math.ceil(TIME_LIMIT_ZONE_INCREASE_MAX_FRACTION / TIME_LIMIT_ZONE_INCREASE_STEP_FRACTION))
    for step_idx in range(1, step_count + 1):
        fraction = TIME_LIMIT_ZONE_INCREASE_STEP_FRACTION * step_idx
        candidate = int(math.ceil(current_n_zones * (1.0 + fraction)))
        candidate = int(max(current_n_zones + 1, min(max_k, candidate)))
        if candidate > current_n_zones and candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= TIME_LIMIT_ZONE_INCREASE_MAX_CANDIDATES:
            break
    return candidates


def increased_zone_count_rescue_params(params: dict[str, Any], context: dict[str, Any], target_n_zones: int) -> dict[str, Any]:
    rescued = dict(params)
    gdf = context["gdf"]
    graph = context["graph"]
    school_count = max(1, int(context.get("school_count") or 1))
    students_total = float(pd.to_numeric(gdf["students"], errors="coerce").fillna(0.0).sum())
    n_zones = int(target_n_zones)
    centers = generate_predicted_centers(gdf, graph, n_zones)
    avg_schools = school_count / max(n_zones, 1)
    slack_below = int(rescued.get("annealed_school_slack_below") or 0)
    slack_above = int(rescued.get("annealed_school_slack_above") or 0)
    rescued.update({
        "annealed_n_zones": n_zones,
        "annealed_centers": json.dumps([int(c) for c in centers]),
        "derived_balance_B_students": float(rescued.get("annealed_balance_tolerance") or 0.0) * (
            students_total / max(n_zones, 1)
        ),
        "derived_max_shortage_allowed": float(rescued.get("annealed_max_shortage_fraction") or 0.0) * (
            students_total / max(n_zones, 1)
        ),
        "derived_min_schools_per_zone": max(1, math.floor(avg_schools) - slack_below),
    })
    rescued["derived_max_schools_per_zone"] = max(
        int(rescued["derived_min_schools_per_zone"]),
        math.ceil(avg_schools) + slack_above,
    )
    return rescued


def _relax_bound_pair(
    params: dict[str, Any],
    lower_key: str,
    upper_key: str,
    level: int,
    *,
    lower_multiplier: float = 0.65,
    lower_subtract: float = 0.10,
    upper_multiplier: float = 1.50,
    upper_add: float = 0.15,
) -> None:
    if level <= 0:
        return
    if level >= 2:
        params[lower_key] = 0.0
        params[upper_key] = 10.0
        return
    lower = float(params.get(lower_key) or 0.0)
    upper = float(params.get(upper_key) or lower)
    params[lower_key] = max(0.0, lower * lower_multiplier - lower_subtract)
    params[upper_key] = max(params[lower_key], upper * upper_multiplier + upper_add)


def final_grid_rescue_params(
    params: dict[str, Any],
    context: dict[str, Any],
    *,
    quality_level: int = 0,
    student_level: int = 0,
    capacity_level: int = 0,
    frl_focal_level: int = 0,
    diversity_level: int = 0,
) -> dict[str, Any]:
    rescued = dict(params)
    gdf = context["gdf"]
    n_zones = max(1, int(rescued.get("annealed_n_zones") or 1))
    students_total = float(pd.to_numeric(gdf["students"], errors="coerce").fillna(0.0).sum())
    avg_students = students_total / max(n_zones, 1)
    distance_scale = float(context.get("distance_scale_miles") or rescued.get("city_distance_scale_miles") or 0.0)

    _relax_bound_pair(
        rescued,
        "annealed_school_quality_lower",
        "annealed_school_quality_upper",
        quality_level,
        lower_multiplier=0.60,
        lower_subtract=0.12,
        upper_multiplier=1.65,
        upper_add=0.18,
    )

    if student_level == 1:
        rescued["annealed_balance_tolerance"] = max(
            float(rescued.get("annealed_balance_tolerance") or 0.0) * 1.75 + 0.25,
            float(rescued.get("annealed_balance_tolerance") or 0.0),
        )
        rescued["annealed_cover_distance_factor"] = max(
            float(rescued.get("annealed_cover_distance_factor") or 0.0) * 1.75,
            float(rescued.get("annealed_cover_distance_factor") or 0.0),
        )
    elif student_level >= 2:
        rescued["annealed_balance_tolerance"] = 10.0
        rescued["annealed_cover_distance_factor"] = max(
            float(rescued.get("annealed_cover_distance_factor") or 0.0) * 3.0,
            50.0,
        )
    rescued["derived_balance_B_students"] = float(rescued.get("annealed_balance_tolerance") or 0.0) * avg_students
    rescued["derived_cover_distance_miles"] = float(rescued.get("annealed_cover_distance_factor") or 0.0) * max(distance_scale, 0.0)

    if capacity_level == 1:
        rescued["annealed_max_shortage_fraction"] = max(
            float(rescued.get("annealed_max_shortage_fraction") or 0.0) * 1.75 + 0.20,
            0.35,
        )
    elif capacity_level >= 2:
        rescued["annealed_max_shortage_fraction"] = max(
            float(rescued.get("annealed_max_shortage_fraction") or 0.0),
            1.50,
        )
    rescued["derived_max_shortage_allowed"] = float(rescued.get("annealed_max_shortage_fraction") or 0.0) * avg_students

    for lower_key, upper_key in [
        ("annealed_frl_lower", "annealed_frl_upper"),
        ("annealed_focal_group_lower", "annealed_focal_group_upper"),
    ]:
        _relax_bound_pair(
            rescued,
            lower_key,
            upper_key,
            frl_focal_level,
            lower_multiplier=0.60,
            lower_subtract=0.12,
            upper_multiplier=1.65,
            upper_add=0.18,
        )

    _relax_bound_pair(
        rescued,
        "annealed_diversity_index_lower",
        "annealed_diversity_index_upper",
        diversity_level,
        lower_multiplier=0.75,
        lower_subtract=0.06,
        upper_multiplier=1.35,
        upper_add=0.10,
    )

    for lower_key, upper_key in [
        ("annealed_frl_lower", "annealed_frl_upper"),
        ("annealed_focal_group_lower", "annealed_focal_group_upper"),
        ("annealed_diversity_index_lower", "annealed_diversity_index_upper"),
        ("annealed_school_quality_lower", "annealed_school_quality_upper"),
    ]:
        if float(rescued.get(upper_key) or 0.0) < float(rescued.get(lower_key) or 0.0):
            rescued[upper_key] = rescued[lower_key]
    return rescued


def final_grid_rescue_param_grid(
    params: dict[str, Any],
    context: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    grid: list[tuple[str, str, dict[str, Any]]] = []
    # Layer 1: school quality only.
    for quality_level in [1, 2]:
        grid.append((
            "school_quality",
            f"quality{quality_level}",
            final_grid_rescue_params(params, context, quality_level=quality_level),
        ))
    # Layer 2: add student balance and distance.
    for quality_level in [1, 2]:
        for student_level in [1, 2]:
            grid.append((
                "school_quality_student_balance_distance",
                f"quality{quality_level}_student{student_level}",
                final_grid_rescue_params(params, context, quality_level=quality_level, student_level=student_level),
            ))
    # Layer 3: add capacity/shortage.
    for quality_level in [1, 2]:
        for student_level in [1, 2]:
            for capacity_level in [1, 2]:
                grid.append((
                    "school_quality_student_balance_distance_capacity",
                    f"quality{quality_level}_student{student_level}_capacity{capacity_level}",
                    final_grid_rescue_params(
                        params,
                        context,
                        quality_level=quality_level,
                        student_level=student_level,
                        capacity_level=capacity_level,
                    ),
                ))
    # Layer 4: add FRL/focal-group bounds. Diversity remains untouched here.
    for quality_level in [1, 2]:
        for student_level in [1, 2]:
            for capacity_level in [1, 2]:
                for frl_focal_level in [1, 2]:
                    grid.append((
                        "school_quality_student_balance_distance_capacity_frl_focal",
                        (
                            f"quality{quality_level}_student{student_level}_capacity{capacity_level}"
                            f"_frlfocal{frl_focal_level}"
                        ),
                        final_grid_rescue_params(
                            params,
                            context,
                            quality_level=quality_level,
                            student_level=student_level,
                            capacity_level=capacity_level,
                            frl_focal_level=frl_focal_level,
                        ),
                    ))
    # Layer 5: last-resort diversity relaxation.
    for quality_level in [1, 2]:
        for student_level in [1, 2]:
            for capacity_level in [1, 2]:
                for frl_focal_level in [1, 2]:
                    for diversity_level in [1, 2]:
                        grid.append((
                            "school_quality_student_balance_distance_capacity_frl_focal_diversity",
                            (
                                f"quality{quality_level}_student{student_level}_capacity{capacity_level}"
                                f"_frlfocal{frl_focal_level}_diversity{diversity_level}"
                            ),
                            final_grid_rescue_params(
                                params,
                                context,
                                quality_level=quality_level,
                                student_level=student_level,
                                capacity_level=capacity_level,
                                frl_focal_level=frl_focal_level,
                                diversity_level=diversity_level,
                            ),
                        ))
    return grid


def run_predicted_milp_city(
    city: str,
    model_bundle: Optional[dict[str, Any]],
    output_root: Path,
    solver_time_limit_seconds: Optional[int],
    outer_search_label: str,
    feature_row: Optional[dict[str, Any]] = None,
    direct_prediction: Optional[dict[str, Any]] = None,
    extra_diagnostics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    runtime_start = time.perf_counter()
    context = load_city_ml_context(city)
    if direct_prediction is None:
        if model_bundle is None:
            raise ValueError("Either model_bundle or direct_prediction is required")
        if feature_row is None:
            feature_row = city_feature_row(city)
        direct_prediction = model_predict_direct(model_bundle, feature_row)
    params = postprocess_predicted_params(direct_prediction, context["gdf"], context["graph"], context["distance_scale_miles"])
    centers = [int(value) for value in json.loads(params["annealed_centers"])]
    config = exact_config_from_predicted_params(params, solver_time_limit_seconds)
    rescue_meta = {
        "time_limit_rescue_attempted": False,
        "time_limit_rescue_success": False,
        "small_city_one_zone_rescue_attempted": False,
        "small_city_one_zone_rescue_success": False,
        "moderate_small_school_rescue_attempted": False,
        "moderate_small_school_hard_rescue_success": False,
        "moderate_small_school_extreme_rescue_success": False,
        "low_schools_per_zone_rescue_attempted": False,
        "low_schools_per_zone_rescue_success": False,
        "time_limit_zone_increase_rescue_attempted": False,
        "time_limit_zone_increase_rescue_success": False,
        "time_limit_zone_increase_rescue_seconds": TIME_LIMIT_RESCUE_SECONDS,
        "time_limit_zone_increase_rescue_candidate_n_zones": None,
        "time_limit_zone_increase_rescue_n_zones": None,
        "final_grid_rescue_attempted": False,
        "final_grid_rescue_success": False,
        "final_grid_rescue_stage": None,
        "final_grid_rescue_profile": None,
        "final_grid_rescue_attempt_count": 0,
        "rescue_strategy": "none",
        "initial_predicted_n_zones": int(params.get("annealed_n_zones", 1)),
        "small_city_one_zone_rescue_reason": small_city_one_zone_rescue_reason(context, params),
        "small_city_one_zone_direct_max_schools": SMALL_CITY_ONE_ZONE_DIRECT_MAX_SCHOOLS,
        "small_city_one_zone_cluster_max_distance_miles": SMALL_CITY_ONE_ZONE_CLUSTER_MAX_DISTANCE_MILES,
        "moderate_small_school_rescue_reason": moderate_small_school_rescue_reason(context, params, direct_prediction),
        "moderate_small_school_max_schools": MODERATE_SMALL_SCHOOL_RESCUE_MAX_SCHOOLS,
        "moderate_small_school_max_schools_per_zone": MODERATE_SMALL_SCHOOL_RESCUE_MAX_SCHOOLS_PER_ZONE,
        "predicted_constraint_difficulty_label": predicted_constraint_difficulty_label(direct_prediction),
        "initial_schools_per_predicted_zone": initial_schools_per_predicted_zone(context, params),
        "low_schools_per_zone_rescue_reason": low_schools_per_zone_rescue_reason(context, params),
        "low_schools_per_zone_min_schools_per_zone": LOW_SCHOOLS_PER_ZONE_RESCUE_MIN_SCHOOLS_PER_ZONE,
        "low_schools_per_zone_rescue_n_zones": low_schools_per_zone_rescue_target_n_zones(context, params),
        "initial_solver_time_limit_seconds": solver_time_limit_seconds,
        "final_solver_time_limit_seconds": solver_time_limit_seconds,
    }
    initial_infeasible_error: Optional[str] = None

    def run_low_schools_per_zone_rescue(
        base_params: dict[str, Any],
        source_exc: BaseException,
    ) -> tuple[dict[str, Any], list[int], dict[int, int], dict[str, Any]]:
        rescue_meta.update({
            "low_schools_per_zone_rescue_attempted": True,
            "initial_infeasible_error": initial_infeasible_error,
            "rescue_strategy": "low_schools_per_zone_reduce_n_zones",
        })
        candidate_n_zones = low_schools_per_zone_rescue_candidate_n_zones(context, base_params)
        if not candidate_n_zones:
            raise RuntimeError(
                "Low-schools-per-zone rescue was requested but no valid reduced zone count exists."
            ) from source_exc
        failed_attempts = []
        for target_n_zones in candidate_n_zones:
            reduced_params = reduced_zone_count_rescue_params(base_params, context, target_n_zones)
            reduced_centers = [int(value) for value in json.loads(reduced_params["annealed_centers"])]
            reduced_config = exact_config_from_predicted_params(reduced_params, solver_time_limit_seconds)
            try:
                reduced_assignment, reduced_diagnostics = solve_exact_milp(
                    context["gdf"],
                    context["graph"],
                    reduced_config,
                    reduced_centers,
                    milp_cache=context["milp_cache"],
                )
            except Exception as reduced_exc:
                failed_attempts.append(f"{target_n_zones}: {type(reduced_exc).__name__}: {reduced_exc}")
                continue
            rescue_meta["low_schools_per_zone_rescue_success"] = True
            rescue_meta["low_schools_per_zone_rescue_n_zones"] = target_n_zones
            rescue_meta["low_schools_per_zone_rescue_candidate_n_zones"] = json.dumps(candidate_n_zones)
            return reduced_params, reduced_centers, reduced_assignment, reduced_diagnostics
        rescue_meta["low_schools_per_zone_rescue_candidate_n_zones"] = json.dumps(candidate_n_zones)
        raise RuntimeError(
            "Initial MILP was infeasible; low-schools-per-zone reduced-zone rescue failed for all candidates: "
            + " | ".join(failed_attempts)
        ) from source_exc

    def run_final_grid_rescue(
        base_params: dict[str, Any],
        source_exc: BaseException,
    ) -> tuple[dict[str, Any], list[int], dict[int, int], dict[str, Any]]:
        rescue_meta.update({
            "final_grid_rescue_attempted": True,
            "initial_infeasible_error": initial_infeasible_error,
            "rescue_strategy": "final_grid",
        })
        failed_attempts = []
        for attempt_idx, (stage, profile, grid_params) in enumerate(
            final_grid_rescue_param_grid(base_params, context),
            start=1,
        ):
            grid_centers = [int(value) for value in json.loads(grid_params["annealed_centers"])]
            grid_config = exact_config_from_predicted_params(grid_params, solver_time_limit_seconds)
            try:
                grid_assignment, grid_diagnostics = solve_exact_milp(
                    context["gdf"],
                    context["graph"],
                    grid_config,
                    grid_centers,
                    milp_cache=context["milp_cache"],
                )
            except Exception as grid_exc:
                failed_attempts.append(f"{stage}/{profile}: {type(grid_exc).__name__}: {grid_exc}")
                continue
            rescue_meta["final_grid_rescue_success"] = True
            rescue_meta["final_grid_rescue_stage"] = stage
            rescue_meta["final_grid_rescue_profile"] = profile
            rescue_meta["final_grid_rescue_attempt_count"] = attempt_idx
            rescue_meta["rescue_strategy"] = f"final_grid_{stage}"
            return grid_params, grid_centers, grid_assignment, grid_diagnostics
        rescue_meta["final_grid_rescue_attempt_count"] = len(failed_attempts)
        rescue_meta["final_grid_rescue_failed_error"] = " | ".join(failed_attempts[-5:])
        raise RuntimeError(
            "Initial MILP was infeasible; final grid rescue failed for all candidates: "
            + " | ".join(failed_attempts[-5:])
        ) from source_exc

    def run_time_limit_zone_increase_rescue(
        base_params: dict[str, Any],
        source_exc: BaseException,
    ) -> tuple[dict[str, Any], list[int], dict[int, int], dict[str, Any]]:
        rescue_meta.update({
            "time_limit_zone_increase_rescue_attempted": True,
            "rescue_strategy": "time_limit_increase_n_zones",
            "final_solver_time_limit_seconds": TIME_LIMIT_RESCUE_SECONDS,
        })
        candidate_n_zones = time_limit_zone_increase_candidate_n_zones(context, base_params)
        rescue_meta["time_limit_zone_increase_rescue_candidate_n_zones"] = json.dumps(candidate_n_zones)
        if not candidate_n_zones:
            raise RuntimeError(
                "Time-limit zone-increase rescue was requested but no valid increased zone count exists."
            ) from source_exc
        failed_attempts = []
        for target_n_zones in candidate_n_zones:
            increased_params = increased_zone_count_rescue_params(base_params, context, target_n_zones)
            increased_centers = [int(value) for value in json.loads(increased_params["annealed_centers"])]
            increased_config = exact_config_from_predicted_params(increased_params, TIME_LIMIT_RESCUE_SECONDS)
            try:
                increased_assignment, increased_diagnostics = solve_exact_milp_with_wall_timeout(
                    context["gdf"],
                    context["graph"],
                    increased_config,
                    increased_centers,
                    context["milp_cache"],
                    TIME_LIMIT_RESCUE_SECONDS,
                )
            except Exception as increased_exc:
                failed_attempts.append(f"{target_n_zones}: {type(increased_exc).__name__}: {increased_exc}")
                continue
            rescue_meta["time_limit_zone_increase_rescue_success"] = True
            rescue_meta["time_limit_zone_increase_rescue_n_zones"] = target_n_zones
            rescue_meta["rescue_strategy"] = "time_limit_increase_n_zones"
            return increased_params, increased_centers, increased_assignment, increased_diagnostics
        raise RuntimeError(
            "Initial MILP reached the solver time limit; 10-minute same-zone rescue also timed out; "
            "increased-zone rescue failed for all candidates: "
            + " | ".join(failed_attempts)
        ) from source_exc

    try:
        assignment, diagnostics = solve_exact_milp(context["gdf"], context["graph"], config, centers, milp_cache=context["milp_cache"])
    except RuntimeError as exc:
        initial_infeasible_error = f"{type(exc).__name__}: {exc}" if is_infeasible_milp_error(exc) else None
        if is_time_limit_milp_error(exc) and solver_time_limit_seconds not in (None, 0):
            rescue_meta.update({
                "time_limit_rescue_attempted": True,
                "initial_time_limit_error": f"{type(exc).__name__}: {exc}",
                "final_solver_time_limit_seconds": TIME_LIMIT_RESCUE_SECONDS,
                "rescue_strategy": "time_limit_10min_same_n_zones",
            })
            rescue_config = exact_config_from_predicted_params(params, TIME_LIMIT_RESCUE_SECONDS)
            try:
                assignment, diagnostics = solve_exact_milp_with_wall_timeout(
                    context["gdf"],
                    context["graph"],
                    rescue_config,
                    centers,
                    context["milp_cache"],
                    TIME_LIMIT_RESCUE_SECONDS,
                )
            except Exception as rescue_exc:
                rescue_meta["time_limit_10min_same_n_zones_failed_error"] = (
                    f"{type(rescue_exc).__name__}: {rescue_exc}"
                )
                if is_time_limit_milp_error(rescue_exc):
                    params, centers, assignment, diagnostics = run_time_limit_zone_increase_rescue(params, rescue_exc)
                else:
                    raise RuntimeError(
                        "Initial MILP reached the solver time limit; 10-minute same-zone time-limit rescue failed: "
                        f"{type(rescue_exc).__name__}: {rescue_exc}"
                    ) from rescue_exc
            rescue_meta["time_limit_rescue_success"] = True
        elif is_infeasible_milp_error(exc) and small_city_one_zone_rescue_eligible(context, params):
            rescue_meta.update({
                "small_city_one_zone_rescue_attempted": True,
                "initial_infeasible_error": initial_infeasible_error,
                "rescue_strategy": "small_city_one_zone",
            })
            rescue_params = one_zone_rescue_params(params, context)
            rescue_centers = [int(value) for value in json.loads(rescue_params["annealed_centers"])]
            rescue_config = exact_config_from_predicted_params(rescue_params, solver_time_limit_seconds)
            try:
                assignment, diagnostics = solve_exact_milp(
                    context["gdf"],
                    context["graph"],
                    rescue_config,
                    rescue_centers,
                    milp_cache=context["milp_cache"],
                )
            except Exception as rescue_exc:
                raise RuntimeError(
                    "Initial MILP was infeasible; small-city one-zone rescue failed: "
                    f"{type(rescue_exc).__name__}: {rescue_exc}"
                ) from rescue_exc
            params = rescue_params
            centers = rescue_centers
            rescue_meta["small_city_one_zone_rescue_success"] = True
        elif is_infeasible_milp_error(exc) and moderate_small_school_rescue_eligible(context, params, direct_prediction):
            rescue_meta.update({
                "moderate_small_school_rescue_attempted": True,
                "initial_infeasible_error": initial_infeasible_error,
                "rescue_strategy": "moderate_small_school_hard_then_extreme",
            })
            hard_params = difficulty_relaxed_rescue_params(params, context, "Hard")
            hard_centers = [int(value) for value in json.loads(hard_params["annealed_centers"])]
            hard_config = exact_config_from_predicted_params(hard_params, solver_time_limit_seconds)
            try:
                assignment, diagnostics = solve_exact_milp(
                    context["gdf"],
                    context["graph"],
                    hard_config,
                    hard_centers,
                    milp_cache=context["milp_cache"],
                )
                params = hard_params
                centers = hard_centers
                rescue_meta["moderate_small_school_hard_rescue_success"] = True
                rescue_meta["rescue_strategy"] = "moderate_small_school_hard"
            except RuntimeError as hard_exc:
                if not is_infeasible_milp_error(hard_exc):
                    raise RuntimeError(
                        "Initial MILP was infeasible; Moderate small-school Hard rescue failed without "
                        f"an infeasible status: {type(hard_exc).__name__}: {hard_exc}"
                    ) from hard_exc
                rescue_meta["hard_rescue_infeasible_error"] = f"{type(hard_exc).__name__}: {hard_exc}"
                extreme_params = difficulty_relaxed_rescue_params(params, context, "Extreme")
                extreme_centers = [int(value) for value in json.loads(extreme_params["annealed_centers"])]
                extreme_config = exact_config_from_predicted_params(extreme_params, solver_time_limit_seconds)
                try:
                    assignment, diagnostics = solve_exact_milp(
                        context["gdf"],
                        context["graph"],
                        extreme_config,
                        extreme_centers,
                        milp_cache=context["milp_cache"],
                    )
                except Exception as extreme_exc:
                    rescue_meta["extreme_rescue_failed_error"] = f"{type(extreme_exc).__name__}: {extreme_exc}"
                    if low_schools_per_zone_rescue_eligible(context, params):
                        try:
                            params, centers, assignment, diagnostics = run_low_schools_per_zone_rescue(params, extreme_exc)
                        except Exception as low_zone_exc:
                            rescue_meta["low_schools_per_zone_rescue_failed_error"] = (
                                f"{type(low_zone_exc).__name__}: {low_zone_exc}"
                            )
                            params, centers, assignment, diagnostics = run_final_grid_rescue(params, low_zone_exc)
                    else:
                        params, centers, assignment, diagnostics = run_final_grid_rescue(params, extreme_exc)
                else:
                    params = extreme_params
                    centers = extreme_centers
                    rescue_meta["moderate_small_school_extreme_rescue_success"] = True
                    rescue_meta["rescue_strategy"] = "moderate_small_school_extreme"
            if rescue_meta["low_schools_per_zone_rescue_success"]:
                pass
            elif rescue_meta["moderate_small_school_hard_rescue_success"]:
                pass
            elif rescue_meta["moderate_small_school_extreme_rescue_success"]:
                pass
            elif rescue_meta["final_grid_rescue_success"]:
                pass
            else:
                # Defensive guard: the branch should either succeed at Hard,
                # Extreme, the ordered low-schools-per-zone fallback, or the
                # final layered grid fallback.
                raise RuntimeError("Moderate small-school rescue did not produce a solution.")
        elif is_infeasible_milp_error(exc) and low_schools_per_zone_rescue_eligible(context, params):
            try:
                params, centers, assignment, diagnostics = run_low_schools_per_zone_rescue(params, exc)
            except Exception as low_zone_exc:
                rescue_meta["low_schools_per_zone_rescue_failed_error"] = (
                    f"{type(low_zone_exc).__name__}: {low_zone_exc}"
                )
                params, centers, assignment, diagnostics = run_final_grid_rescue(params, low_zone_exc)
        elif is_infeasible_milp_error(exc):
            params, centers, assignment, diagnostics = run_final_grid_rescue(params, exc)
        else:
            raise
    distances = centroid_distances_to_centers(context["gdf"], centers)
    summary = summarize_solution(context["gdf"], context["graph"], assignment, centers, distances)
    diagnostics.update({
        "outer_search": outer_search_label,
        "predicted_parameter_schema": PARAMETER_OUTPUT_FIELDS,
        "predicted_parameters": params,
        "direct_prediction_feasibility_safe_calibration": direct_prediction.get("feasibility_safe_calibration"),
        "direct_prediction_global_baseline_model_path": direct_prediction.get("global_baseline_model_path"),
        "selected_annealing_state": selected_state_from_predicted_params(params, centers),
        "score": score_solution(summary, diagnostics.get("avg_students", 0.0), diagnostics.get("avg_schools", 0.0), PolicyWeights().score_weights),
    })
    diagnostics.update(rescue_meta)
    diagnostics["runtime_seconds"] = float(time.perf_counter() - runtime_start)
    diagnostics["runtime_scope"] = (
        "city preparation through initial predicted-parameter MILP plus conditional rescue; "
        "model inference excluded for OOF/direct-prediction runs"
    )
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    output_dir = output_root / city
    validate_output_dir(city, output_dir, output_root=output_root)
    paths = context["paths"]
    write_outputs(output_dir, context["gdf"], context["graph"], assignment, summary, diagnostics, city, paths.school_csv, paths.block_source)
    (output_dir / f"{city.lower()}_zone_diagnostics.json").write_text(json.dumps(_jsonify(diagnostics), indent=2), encoding="utf-8")
    return {"city": city, "comparison_row": build_city_comparison_row(city, summary, diagnostics), "parameter_row": {"city": city, **{field: params.get(field) for field in PARAMETER_OUTPUT_FIELDS}}}


def write_predicted_run_tables(results: list[dict[str, Any]], output_root: Path, prefix: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if results:
        pd.DataFrame([r["comparison_row"] for r in results]).to_csv(output_root / f"{prefix}_milp_results.csv", index=False)
        pd.DataFrame([r["parameter_row"] for r in results]).to_csv(output_root / f"{prefix}_predicted_parameters.csv", index=False)


PREDICTED_OUTPUT_ROOT = ZONE_OPTIMIZATION_OUTPUT_ROOT
PREDICTED_MODEL_PATH = SERVICE_DELIVERY_MODEL_ROOT / "TypologyAware_XGBoost-Type.pkl"
GLOBAL_BASELINE_MODEL_PATH = Path(os.environ.get("COMPASS_GLOBAL_MODEL", str(SERVICE_DELIVERY_MODEL_ROOT / "Global_XGBoost.pkl")))
PREDICTED_FOLD_MODEL_DIR = MODEL_ROOT / "5Fold_TypologyAware_XGBoost-Type"
PREDICTED_FOLD_OUTPUT_ROOT = PREDICTED_OUTPUT_ROOT
PREDICTED_FOLD_OOF_PATH = MODEL_OUTPUT_ROOT / "TypologyAware_oof_Predictions.csv"


def direct_prediction_from_oof_row(row: pd.Series) -> dict[str, Any]:
    prediction = {field: row[field] for field in DIRECT_TARGET_FIELDS if field in row.index}
    for field in [
        "constraint_difficulty_label",
        "constraint_difficulty_oracle_label",
        "difficulty_parameter_relaxation",
        "feasibility_safe_calibration",
    ]:
        if field in row.index:
            prediction[field] = row[field]
    return prediction


def run_typology_aware_oof_predictions(
    oof_path: Path,
    output_root: Path,
    solver_time_limit_seconds: Optional[int],
    skip_existing_outputs: bool,
    fold_filter: int | None = None,
    table_output_root: Path | None = None,
) -> dict[str, Any]:
    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    oof = pd.read_csv(oof_path)
    required = {"city", "fold", *DIRECT_TARGET_FIELDS}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"OOF prediction table is missing columns: {missing}")
    if fold_filter is not None:
        oof = oof[oof["fold"].astype(int).eq(int(fold_filter))].copy()
        if oof.empty:
            raise ValueError(f"No OOF rows found for fold={fold_filter} in {oof_path}")
    table_root = table_output_root or output_root
    results = []
    failures = []
    total = len(oof)
    for completed, (_, row) in enumerate(oof.iterrows(), start=1):
        city = str(row["city"]).strip()
        fold = int(row["fold"])
        if skip_existing_outputs and city_output_complete(city, output_root):
            print(f"[{completed}/{total}] skip existing fold={fold} {city}")
            continue
        feature_row = add_typology_features(city_feature_row(city), city)
        print(f"[{completed}/{total}] typology-aware 5-fold OOF MILP fold={fold} {city}", flush=True)
        attempt_start = time.perf_counter()
        try:
            result = run_predicted_milp_city(
                city,
                None,
                output_root,
                solver_time_limit_seconds,
                "typology_aware_xgboost_5fold_oof_parameter_prediction",
                feature_row=feature_row,
                direct_prediction=direct_prediction_from_oof_row(row),
                extra_diagnostics={
                    "fold": fold,
                    "fold_oof_path": str(oof_path),
                    "typology_features": {col: feature_row.get(col) for col in CATEGORICAL_LABEL_FIELDS + ["type_id"]},
                },
            )
            result["comparison_row"]["fold"] = fold
            result["comparison_row"]["fold_oof_path"] = str(oof_path)
            result["parameter_row"]["fold"] = fold
            result["parameter_row"]["fold_oof_path"] = str(oof_path)
            for col in CATEGORICAL_LABEL_FIELDS + ["type_id"]:
                result["parameter_row"][col] = feature_row.get(col)
            results.append(result)
        except Exception as exc:
            failures.append({
                "city": city,
                "fold": fold,
                "fold_oof_path": str(oof_path),
                "runtime_seconds": float(time.perf_counter() - attempt_start),
                "runtime_scope": (
                    "city preparation through failed initial predicted-parameter MILP plus conditional rescue attempt; "
                    "model inference excluded for OOF/direct-prediction runs"
                ),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{completed}/{total}] failed fold={fold} {city}: {type(exc).__name__}: {exc}", flush=True)
    write_predicted_run_tables(results, table_root, "typology_aware_5fold")
    if failures:
        pd.DataFrame(failures).to_csv(table_root / "typology_aware_5fold_failures.csv", index=False)
    manifest = {
        "source": str(oof_path),
        "fold": fold_filter,
        "result_count": len(results),
        "failure_count": len(failures),
    }
    (table_root / "typology_aware_5fold_manifest.json").write_text(
        json.dumps(_jsonify(manifest), indent=2),
        encoding="utf-8",
    )
    return manifest


def _run_typology_aware_oof_fold_worker(payload: tuple[int, str, str, int, bool, str]) -> dict[str, Any]:
    fold, oof_path, output_root, solver_time_limit_seconds, skip_existing_outputs, table_output_root = payload
    return run_typology_aware_oof_predictions(
        Path(oof_path),
        Path(output_root),
        solver_time_limit_seconds,
        skip_existing_outputs,
        fold_filter=fold,
        table_output_root=Path(table_output_root),
    )


def merge_typology_aware_oof_fold_tables(output_root: Path, fold_table_root: Path, fold_manifests: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result_frames = []
    parameter_frames = []
    failure_frames = []
    for fold_dir in sorted(fold_table_root.glob("fold_*")):
        result_path = fold_dir / "typology_aware_5fold_milp_results.csv"
        parameter_path = fold_dir / "typology_aware_5fold_predicted_parameters.csv"
        failure_path = fold_dir / "typology_aware_5fold_failures.csv"
        if result_path.exists():
            result_frames.append(pd.read_csv(result_path))
        if parameter_path.exists():
            parameter_frames.append(pd.read_csv(parameter_path))
        if failure_path.exists():
            failure_frames.append(pd.read_csv(failure_path))
    if result_frames:
        pd.concat(result_frames, ignore_index=True).to_csv(output_root / "typology_aware_5fold_milp_results.csv", index=False)
    if parameter_frames:
        pd.concat(parameter_frames, ignore_index=True).to_csv(output_root / "typology_aware_5fold_predicted_parameters.csv", index=False)
    if failure_frames:
        pd.concat(failure_frames, ignore_index=True).to_csv(output_root / "typology_aware_5fold_failures.csv", index=False)
    else:
        failure_path = output_root / "typology_aware_5fold_failures.csv"
        if failure_path.exists():
            failure_path.unlink()
    result_count = sum(int(item.get("result_count", 0)) for item in fold_manifests)
    failure_count = sum(int(item.get("failure_count", 0)) for item in fold_manifests)
    (output_root / "typology_aware_5fold_manifest.json").write_text(
        json.dumps(
            _jsonify({
                "parallel_folds": sorted(int(item["fold"]) for item in fold_manifests if item.get("fold") is not None),
                "result_count": result_count,
                "failure_count": failure_count,
                "fold_manifests": fold_manifests,
            }),
            indent=2,
        ),
        encoding="utf-8",
    )


def run_typology_aware_oof_predictions_parallel(
    oof_path: Path,
    output_root: Path,
    solver_time_limit_seconds: Optional[int],
    skip_existing_outputs: bool,
    max_workers: int = 5,
) -> None:
    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    oof = pd.read_csv(oof_path, usecols=["fold"])
    folds = sorted(int(fold) for fold in oof["fold"].dropna().unique())
    if not folds:
        raise ValueError(f"No folds found in {oof_path}")
    fold_table_root = output_root / "_fold_tables" / "typology_aware_5fold"
    if fold_table_root.exists():
        shutil.rmtree(fold_table_root)
    fold_table_root.mkdir(parents=True, exist_ok=True)
    payloads = [
        (
            fold,
            str(oof_path),
            str(output_root),
            solver_time_limit_seconds,
            skip_existing_outputs,
            str(fold_table_root / f"fold_{fold}"),
        )
        for fold in folds
    ]
    manifests = []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(payloads))) as executor:
        future_to_fold = {executor.submit(_run_typology_aware_oof_fold_worker, payload): payload[0] for payload in payloads}
        for future in as_completed(future_to_fold):
            fold = future_to_fold[future]
            try:
                manifest = future.result()
                manifests.append(manifest)
                print(f"[fold {fold}] typology-aware OOF worker finished: {manifest}", flush=True)
            except Exception as exc:
                print(f"[fold {fold}] typology-aware OOF worker failed: {type(exc).__name__}: {exc}", flush=True)
                raise
    merge_typology_aware_oof_fold_tables(output_root, fold_table_root, manifests)


def fold_number_from_bundle_or_path(model_bundle: dict[str, Any], model_path: Path) -> int:
    if model_bundle.get("fold") is not None:
        return int(model_bundle["fold"])
    match = re.search(r"fold_(\d+)", model_path.stem)
    if match:
        return int(match.group(1))
    match = re.search(r"fold_(\d+)", str(model_path.parent))
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer fold number from {model_path}")


def test_cities_from_fold_bundle(model_bundle: dict[str, Any], model_path: Path) -> list[str]:
    cities = model_bundle.get("test_cities")
    if not cities:
        raise ValueError(f"Fold model does not contain test_cities: {model_path}")
    return [str(city) for city in cities]


def add_typology_features(feature_row: dict[str, Any], city: str) -> dict[str, Any]:
    labels = read_typology_labels()
    match = labels[labels["city"].astype(str).eq(str(city))]
    if match.empty:
        for col in DIRECT_CATEGORICAL_LABEL_FIELDS:
            feature_row[col] = "Unknown"
        feature_row[CONSTRAINT_DIFFICULTY_FIELD] = "Moderate"
        feature_row["type_id"] = "Unknown"
        return feature_row
    for col in DIRECT_CATEGORICAL_LABEL_FIELDS + ["type_id"]:
        feature_row[col] = str(match[col].iloc[0])
    feature_row[CONSTRAINT_DIFFICULTY_FIELD] = str(feature_row.get(CONSTRAINT_DIFFICULTY_FIELD) or "Moderate")
    return feature_row


def run_typology_aware_fold_models(
    fold_model_dir: Path,
    output_root: Path,
    solver_time_limit_seconds: Optional[int],
    skip_existing_outputs: bool,
) -> None:
    model_paths = sorted(fold_model_dir.glob("fold_*.pkl"))
    if not model_paths:
        raise FileNotFoundError(f"No fold models found under {fold_model_dir}")
    results = []
    failures = []
    total = sum(len(test_cities_from_fold_bundle(load_model_bundle(path), path)) for path in model_paths)
    completed = 0
    for model_path in model_paths:
        model_bundle = load_model_bundle(model_path)
        fold = fold_number_from_bundle_or_path(model_bundle, model_path)
        cities = test_cities_from_fold_bundle(model_bundle, model_path)
        for city in cities:
            completed += 1
            if skip_existing_outputs and city_output_complete(city, output_root):
                print(f"[{completed}/{total}] skip existing fold={fold} {city}")
                continue
            feature_row = add_typology_features(city_feature_row(city), city)
            print(f"[{completed}/{total}] typology-aware 5-fold predicted MILP fold={fold} {city}", flush=True)
            try:
                result = run_predicted_milp_city(
                    city,
                    model_bundle,
                    output_root,
                    solver_time_limit_seconds,
                    "typology_aware_xgboost_5fold_parameter_prediction",
                    feature_row=feature_row,
                    extra_diagnostics={
                        "fold": fold,
                        "fold_model_path": str(model_path),
                        "typology_features": {col: feature_row.get(col) for col in CATEGORICAL_LABEL_FIELDS + ["type_id"]},
                    },
                )
                result["comparison_row"]["fold"] = fold
                result["comparison_row"]["fold_model_path"] = str(model_path)
                result["parameter_row"]["fold"] = fold
                result["parameter_row"]["fold_model_path"] = str(model_path)
                for col in CATEGORICAL_LABEL_FIELDS + ["type_id"]:
                    result["parameter_row"][col] = feature_row.get(col)
                results.append(result)
            except Exception as exc:
                failures.append({"city": city, "fold": fold, "fold_model_path": str(model_path), "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{completed}/{total}] failed fold={fold} {city}: {type(exc).__name__}: {exc}", flush=True)
    write_predicted_run_tables(results, output_root, "typology_aware_5fold")
    if failures:
        pd.DataFrame(failures).to_csv(output_root / "typology_aware_5fold_failures.csv", index=False)
    (output_root / "typology_aware_5fold_manifest.json").write_text(
        json.dumps(_jsonify({"model_count": len(model_paths), "result_count": len(results), "failure_count": len(failures)}), indent=2),
        encoding="utf-8",
    )


def main() -> int:
    global LABEL_RESULTS_PATH, FEATURE_CACHE_PATH, GLOBAL_BASELINE_MODEL_PATH
    global _GLOBAL_BASELINE_MODEL_BUNDLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--all-cities", action="store_true")
    parser.add_argument("--skip-existing-outputs", action="store_true")
    parser.add_argument(
        "--solver-time-limit-seconds",
        type=int,
        default=30,
        help="Optional MILP solver time limit in seconds. Default is 30 seconds. Use 0 to disable.",
    )
    parser.add_argument("--model-path", type=Path, default=PREDICTED_MODEL_PATH)
    parser.add_argument("--global-baseline-model-path", type=Path, default=GLOBAL_BASELINE_MODEL_PATH)
    parser.add_argument("--labels-csv", type=Path, default=LABEL_RESULTS_PATH)
    parser.add_argument("--features-csv", type=Path, default=FEATURE_CACHE_PATH)
    parser.add_argument("--evaluate-fold-models", action="store_true")
    parser.add_argument("--fold-model-dir", type=Path, default=PREDICTED_FOLD_MODEL_DIR)
    parser.add_argument("--fold-output-root", type=Path, default=PREDICTED_FOLD_OUTPUT_ROOT)
    parser.add_argument("--fold-oof-path", type=Path, default=PREDICTED_FOLD_OOF_PATH)
    parser.add_argument("--use-fold-pkl-prediction", action="store_true")
    parser.add_argument("--fold", type=int, help="Run only one OOF fold. Intended for debugging or external parallel orchestration.")
    parser.add_argument("--parallel-folds", action="store_true", help="Run the 5 OOF folds in separate worker processes and merge their tables.")
    parser.add_argument("--fold-workers", type=int, default=5)
    args = parser.parse_args()
    LABEL_RESULTS_PATH = args.labels_csv.resolve()
    FEATURE_CACHE_PATH = args.features_csv.resolve()
    GLOBAL_BASELINE_MODEL_PATH = args.global_baseline_model_path.resolve()
    _GLOBAL_BASELINE_MODEL_BUNDLE = None
    os.environ["COMPASS_LABELS_CSV"] = str(LABEL_RESULTS_PATH)
    os.environ["COMPASS_FEATURES_CSV"] = str(FEATURE_CACHE_PATH)
    os.environ["COMPASS_GLOBAL_MODEL"] = str(GLOBAL_BASELINE_MODEL_PATH)
    if not LABEL_RESULTS_PATH.is_file():
        parser.error(f"Supply your static-label table with --labels-csv: {LABEL_RESULTS_PATH}")
    if not args.evaluate_fold_models:
        for path in (args.model_path,):
            if not path.is_file():
                parser.error(f"Supply a compatible trained model bundle: {path}")
    if args.evaluate_fold_models:
        if args.use_fold_pkl_prediction:
            run_typology_aware_fold_models(
                args.fold_model_dir,
                args.fold_output_root,
                args.solver_time_limit_seconds,
                args.skip_existing_outputs,
            )
        else:
            if args.parallel_folds:
                run_typology_aware_oof_predictions_parallel(
                    args.fold_oof_path,
                    args.fold_output_root,
                    args.solver_time_limit_seconds,
                    args.skip_existing_outputs,
                    max_workers=args.fold_workers,
                )
            else:
                run_typology_aware_oof_predictions(
                    args.fold_oof_path,
                    args.fold_output_root,
                    args.solver_time_limit_seconds,
                    args.skip_existing_outputs,
                    fold_filter=args.fold,
                )
        return 0
    model_bundle = load_model_bundle(args.model_path)
    cities = retained_city_codes_for_ml() if args.all_cities else [args.city]
    results = []
    for index, city in enumerate(cities, start=1):
        if args.skip_existing_outputs and city_output_complete(city, PREDICTED_OUTPUT_ROOT):
            print(f"[{index}/{len(cities)}] skip existing {city}")
            continue
        feature_row = add_typology_features(city_feature_row(city), city)
        print(f"[{index}/{len(cities)}] rescue predicted MILP {city}", flush=True)
        result = run_predicted_milp_city(city, model_bundle, PREDICTED_OUTPUT_ROOT, args.solver_time_limit_seconds, "rescue_xgboost_parameter_prediction", feature_row=feature_row, extra_diagnostics={"typology_features": {col: feature_row.get(col) for col in CATEGORICAL_LABEL_FIELDS + ["type_id"]}})
        for col in CATEGORICAL_LABEL_FIELDS + ["type_id"]:
            result["parameter_row"][col] = feature_row.get(col)
        results.append(result)
    write_predicted_run_tables(results, PREDICTED_OUTPUT_ROOT, "rescue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
