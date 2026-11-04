"""Train Compass models from processed city data and supplied teacher targets.

python Compass_Train.py
python Compass.py --evaluate-fold-models --parallel-folds
python Compass.py --city GA-Atlanta_1304000

Training creates five held-out fold bundles, OOF parameters, Labels.csv, and
full-data service models. It does not run the MILP or publish pretrained weights.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

import Compass as compass

ROOT = Path(__file__).resolve().parent
DIRECT_TARGET_FIELDS = compass.DIRECT_TARGET_FIELDS
DIRECT_LABEL_FIELDS = compass.DIRECT_CATEGORICAL_LABEL_FIELDS
CATEGORICAL_FIELDS = compass.CATEGORICAL_LABEL_FIELDS
DIFFICULTY_FIELD = compass.CONSTRAINT_DIFFICULTY_FIELD
CONSTRAINT_DIFFICULTY_CLASSES = ["Easy", "Moderate", "Hard", "Extreme"]
CONSTRAINT_DIFFICULTY_THRESHOLD_POLICY = "signed_log_jenks_4class"
CLASS_WEIGHTS = {"Easy": 1.0, "Moderate": 2.0, "Hard": 4.0, "Extreme": 8.0}
TARGET_COLUMNS = ["city", *DIRECT_TARGET_FIELDS, "infeasible_count", "infeasible_rate"]


def require_numeric(frame: pd.DataFrame, columns: list[str], context: str) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {sorted(missing)}")
    values = frame[columns].apply(pd.to_numeric, errors="raise").astype(float)
    bad = ~np.isfinite(values.to_numpy())
    if bad.any():
        row, col = np.argwhere(bad)[0]
        city = frame.iloc[row].get("city", frame.index[row])
        raise ValueError(f"{context}: missing/nonfinite {columns[col]} for {city}")
    return values


def require_city_keys(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    if "city" not in frame.columns or frame.empty:
        raise ValueError(f"{context}: nonempty city column required")
    result = frame.copy()
    if result["city"].isna().any():
        raise ValueError(f"{context}: missing city key")
    result["city"] = result["city"].astype(str).str.strip()
    if result["city"].eq("").any() or result["city"].duplicated().any():
        raise ValueError(f"{context}: empty or duplicate city key")
    return result


def read_targets(path: Path) -> pd.DataFrame:
    table = require_city_keys(pd.read_csv(path), "training targets")
    values = require_numeric(table, TARGET_COLUMNS[1:], "training targets")
    if (values < 0).any().any():
        raise ValueError("Training targets must be nonnegative")
    for field in ("annealed_n_zones", "infeasible_count"):
        if not np.equal(values[field], np.floor(values[field])).all():
            raise ValueError(f"{field} must contain integer counts")
    if (values["annealed_n_zones"] < 1).any() or (values["infeasible_rate"] > 1).any():
        raise ValueError("Invalid zone count or infeasible rate")
    table[values.columns] = values
    return table[TARGET_COLUMNS].sort_values("city").reset_index(drop=True)


def build_features(cities: list[str], path: Path | None) -> pd.DataFrame:
    if path is not None:
        features = require_city_keys(pd.read_csv(path), "features")
    else:
        # A caller-supplied cache is opt-in; default features use runtime logic.
        compass.FEATURE_CACHE_PATH = ROOT / "__uncached_compass_features__"
        if compass.FEATURE_CACHE_PATH.exists():
            raise FileExistsError(compass.FEATURE_CACHE_PATH)
        rows = []
        for index, city in enumerate(cities, 1):
            rows.append(compass.city_feature_row(city))
            print(f"features {index}/{len(cities)}: {city}", flush=True)
        features = require_city_keys(pd.DataFrame(rows), "features")
    missing = set(cities) - set(features["city"])
    if missing:
        raise ValueError(f"Features missing cities: {sorted(missing)}")
    features = features.set_index("city").loc[cities].reset_index()
    prohibited = set(TARGET_COLUMNS[1:] + CATEGORICAL_FIELDS + ["type_id"])
    if prohibited.intersection(features.columns):
        raise ValueError("Feature input must not contain teacher targets or labels")
    columns = [col for col in features if col not in {"city", "state", "city_name"}]
    if any(col.startswith(("true_", "oracle_", "constraint_difficulty")) for col in columns):
        raise ValueError("Oracle outcomes cannot be predictor features")
    if len(columns) != 140:
        raise ValueError(f"Expected 140 numeric city features; got {len(columns)}")
    features[columns] = require_numeric(features, columns, "features")
    if "state" not in features:
        features["state"] = features["city"].str.split("-", n=1).str[0]
    if "city_name" not in features:
        features["city_name"] = features["city"].str.split("-", n=1).str[-1]
    return features


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return require_numeric(frame, [column], "static labels")[column]


def oracle_constraint_difficulty_score(table: pd.DataFrame) -> pd.Series:
    require_numeric(table, TARGET_COLUMNS[1:], "difficulty targets")
    upper = [field for field in DIRECT_TARGET_FIELDS
             if field in compass.UNDER_PREDICTION_DANGEROUS_TARGETS]
    lower = [field for field in DIRECT_TARGET_FIELDS
             if field in compass.OVER_PREDICTION_DANGEROUS_TARGETS]
    score = pd.Series(0.0, index=table.index)
    for col in upper + ["infeasible_count", "infeasible_rate"]:
        score += _zscore_for_difficulty(table[col])
    for col in lower:
        score -= _zscore_for_difficulty(table[col])
    return score.rename("constraint_difficulty_score")


def model_parameters(args: argparse.Namespace) -> dict:
    return dict(n_estimators=args.n_estimators, max_depth=4, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                random_state=20260601, n_jobs=args.n_jobs)


def fit_bundle(features: pd.DataFrame, labels: pd.DataFrame, targets: pd.DataFrame,
               difficulty: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray,
               fold: int | None, thresholds: dict, args: argparse.Namespace) -> dict:
    numeric_cols = [col for col in features if col not in {"city", "state", "city_name"}]
    numeric = features[numeric_cols]
    y = targets[DIRECT_TARGET_FIELDS]
    train_cities = features.iloc[train_idx]["city"].tolist()
    test_cities = features.iloc[test_idx]["city"].tolist()
    if set(train_cities) & set(test_cities):
        raise ValueError("Training and held-out cities overlap")
    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", MultiOutputRegressor(XGBRegressor(
            objective="reg:squarederror", **model_parameters(args)))),
    ])
    baseline.fit(numeric.iloc[train_idx], y.iloc[train_idx])
    baseline_bundle = dict(model=baseline, feature_columns=numeric_cols,
                           target_fields=DIRECT_TARGET_FIELDS, fold=fold,
                           train_cities=train_cities, test_cities=test_cities)
    classifier = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=args.classifier_estimators, min_samples_leaf=5,
            class_weight=CLASS_WEIGHTS, random_state=20260601, n_jobs=args.n_jobs)),
    ])
    classifier.fit(numeric.iloc[train_idx], difficulty.iloc[train_idx])
    train = pd.concat([numeric, labels[DIRECT_LABEL_FIELDS]], axis=1).iloc[train_idx].copy()
    train[DIFFICULTY_FIELD] = difficulty.iloc[train_idx].to_numpy()
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FIELDS),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", compass.FeasibilityAwareXGBMultiOutputRegressor(
            DIRECT_TARGET_FIELDS, XGBRegressor, model_parameters(args))),
    ])
    model.fit(train, y.iloc[train_idx])
    return dict(
        model=model, difficulty_classifier=classifier,
        global_baseline_model_bundle=baseline_bundle,
        feature_columns=numeric_cols + CATEGORICAL_FIELDS,
        numeric_feature_columns=numeric_cols,
        direct_categorical_feature_columns=DIRECT_LABEL_FIELDS,
        categorical_feature_columns=CATEGORICAL_FIELDS,
        constraint_difficulty_field=DIFFICULTY_FIELD,
        target_fields=DIRECT_TARGET_FIELDS, difficulty_thresholds=thresholds,
        difficulty_class_weights=CLASS_WEIGHTS,
        difficulty_parameter_relaxation=compass.DIFFICULTY_RELAXATION_RULES,
        asymmetric_loss_weights={"dangerous": 4.0, "safe": 1.0},
        fold=fold, train_cities=train_cities, test_cities=test_cities,
        training_protocol={
            "difficulty_threshold_fit_scope": "full_teacher_dataset",
            "static_label_threshold_fit_scope": "full_feature_dataset",
            "difficulty_classifier_fit_scope": "training_cities_only",
            "parameter_model_fit_scope": "training_cities_only",
            "baseline_fit_scope": "same_training_cities",
            "feature_builder": "Compass.city_feature_row",
            "xgboost_parameters": model_parameters(args),
            "classifier_estimators": args.classifier_estimators,
            "targets_sha256": hashlib.sha256(args.targets.read_bytes()).hexdigest(),
            "runtime_sha256": hashlib.sha256(Path(compass.__file__).read_bytes()).hexdigest(),
            "versions": {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
        },
    )


def held_out_predictions(bundle: dict, features: pd.DataFrame, labels: pd.DataFrame,
                         targets: pd.DataFrame, difficulty: pd.Series,
                         indices: np.ndarray) -> pd.DataFrame:
    records = []
    for index in indices:
        feature_row = features.iloc[index].to_dict()
        feature_row.update(labels.iloc[index][DIRECT_LABEL_FIELDS].to_dict())
        # Uses exactly the inference path, including the paired fold baseline.
        predicted = compass.model_predict_direct(bundle, feature_row)
        require_numeric(pd.DataFrame([predicted]), DIRECT_TARGET_FIELDS, "predictions")
        record = dict(city=feature_row["city"], fold=bundle["fold"], **predicted)
        record.update({field: feature_row[field] for field in DIRECT_LABEL_FIELDS})
        record["type_id"] = feature_row["type_id"]
        record["constraint_difficulty_oracle_label"] = difficulty.iloc[index]
        record.update({f"true_{col}": targets.iloc[index][col] for col in DIRECT_TARGET_FIELDS})
        records.append(record)
    return pd.DataFrame(records)


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pkl.tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=ROOT / "Training_Targets.csv")
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--feature-output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--city", action="append", help="Optional explicit training subset")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--classifier-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.n_estimators, args.classifier_estimators, args.n_jobs) < 1:
        parser.error("Estimator and thread counts must be positive")
    targets = read_targets(args.targets)
    if args.city:
        missing = set(args.city) - set(targets["city"])
        if missing:
            raise ValueError(f"Unknown target cities: {sorted(missing)}")
        targets = targets[targets["city"].isin(args.city)].reset_index(drop=True)
    if len(targets) < 5:
        raise ValueError("Five-fold training requires at least five cities")
    output = args.output_dir.resolve()
    if not args.prepare_only and (output / "Models").exists():
        raise FileExistsError(f"Choose a fresh --output-dir; refusing to mix runs: {output / 'Models'}")
    features = build_features(targets["city"].tolist(), args.features_csv)
    labels = build_labels(features)
    # Preserve the existing experiment's globally defined Jenks classes.
    # The classifier and regressors themselves are trained inside each fold.
    scores = oracle_constraint_difficulty_score(targets)
    thresholds = constraint_difficulty_thresholds(scores)
    difficulty = apply_constraint_difficulty_thresholds(scores, thresholds)
    output.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output / "Labels.csv", index=False)
    if args.feature_output:
        args.feature_output.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.feature_output, index=False)
    if args.prepare_only:
        return 0
    folds = GroupKFold(n_splits=5).split(features, groups=features["city"])
    parts = []
    for fold, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"training fold {fold}/5: {len(train_idx)} train, {len(test_idx)} test", flush=True)
        bundle = fit_bundle(features, labels, targets, difficulty, train_idx, test_idx,
                            fold, thresholds, args)
        path = output / "Models" / "5Fold_TypologyAware_XGBoost-Type" / f"fold_{fold}.pkl"
        save_bundle(bundle, path)
        parts.append(held_out_predictions(bundle, features, labels, targets, difficulty, test_idx))
    oof = require_city_keys(pd.concat(parts, ignore_index=True), "OOF predictions")
    if set(oof["city"]) != set(targets["city"]):
        raise ValueError("OOF cities do not exactly cover the target dataset")
    predictions = output / "Predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    oof.sort_values(["fold", "city"]).to_csv(predictions / "TypologyAware_oof_Predictions.csv", index=False)
    print(f"training full-data service models: {len(targets)} cities", flush=True)
    final = fit_bundle(features, labels, targets, difficulty, np.arange(len(targets)),
                       np.array([], dtype=int), None, thresholds, args)
    service = output / "Models" / "ServiceDeliveryModel"
    save_bundle(final, service / "TypologyAware_XGBoost-Type.pkl")
    save_bundle(final["global_baseline_model_bundle"], service / "Global_XGBoost.pkl")
    print(f"saved five folds and service models to {output / 'Models'}", flush=True)
    return 0


# Label definitions below are copied from the existing training implementation.
def robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    median = float(values.median()) if values.notna().any() else 0.0
    q25 = float(values.quantile(0.25)) if values.notna().any() else median
    q75 = float(values.quantile(0.75)) if values.notna().any() else median
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(values.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return ((values.fillna(median) - median) / scale).clip(-6.0, 6.0)

def high_label(score: pd.Series, high: str = "High", low: str = "Low") -> tuple[pd.Series, float]:
    threshold = float(pd.to_numeric(score, errors="coerce").quantile(0.75))
    labels = pd.Series(low, index=score.index, dtype=object)
    labels.loc[pd.to_numeric(score, errors="coerce").fillna(-np.inf) >= threshold] = high
    return labels, threshold

def build_labels(features: pd.DataFrame) -> pd.DataFrame:
    out = features[["city", "state", "city_name"]].copy()
    out = out.rename(columns={"city": "dataset_code", "city_name": "city"})

    spatial_score = (
        robust_z(numeric(features, "city_distance_scale_miles"))
        + robust_z(numeric(features, "unit_count"))
        + robust_z(numeric(features, "milp_unit_count"))
        - robust_z(numeric(features, "graph_density"))
        - robust_z(numeric(features, "graph_degree_mean"))
    )
    out["spatial_service_difficulty_score"] = spatial_score
    out["spatial_service_difficulty"], out["spatial_service_difficulty_threshold"] = high_label(spatial_score)

    capacity_score = (
        -robust_z(numeric(features, "capacity_student_ratio", default=1.0))
        + robust_z(numeric(features, "students_max"))
        + robust_z(numeric(features, "students_weighted_mean"))
        + robust_z(numeric(features, "block_ele_cap_max"))
        + robust_z(numeric(features, "block_ele_cap_weighted_mean"))
    )
    out["capacity_tightness_score"] = capacity_score
    out["capacity_tightness"], out["capacity_tightness_threshold"] = high_label(
        capacity_score,
        high="Tight",
        low="Loose",
    )

    school_imbalance_score = (
        robust_z(numeric(features, "students_std"))
        + robust_z(numeric(features, "students_max"))
        + robust_z(numeric(features, "students_weighted_std"))
        + robust_z(numeric(features, "school_count_std"))
        + robust_z(numeric(features, "school_count_max"))
        + robust_z(numeric(features, "school_count_weighted_std"))
    )
    out["school_distribution_imbalance_score"] = school_imbalance_score
    out["school_distribution_imbalance"], out["school_distribution_imbalance_threshold"] = high_label(
        school_imbalance_score
    )

    demographic_score = (
        robust_z(numeric(features, "FRL_pct_std"))
        + robust_z(numeric(features, "FRL_pct_weighted_std"))
        + robust_z(numeric(features, "focal_group_pct_std"))
        + robust_z(numeric(features, "focal_group_pct_weighted_std"))
        + robust_z(numeric(features, "diversity_index_std"))
        + robust_z(numeric(features, "school_quality_index_std"))
    )
    out["demographic_constraint_risk_score"] = demographic_score
    out["demographic_constraint_risk"], out["demographic_constraint_risk_threshold"] = high_label(
        demographic_score
    )

    sparse_graph_score = (
        robust_z(numeric(features, "graph_component_count"))
        + robust_z(numeric(features, "graph_isolate_count"))
        - robust_z(numeric(features, "graph_density"))
        - robust_z(numeric(features, "graph_degree_mean"))
        + robust_z(numeric(features, "graph_degree_std"))
    )
    disconnected = (numeric(features, "graph_component_count") > 1) | (numeric(features, "graph_isolate_count") > 0)
    graph_labels, graph_threshold = high_label(sparse_graph_score)
    graph_labels.loc[disconnected] = "High"
    out["graph_fragmentation_risk_score"] = sparse_graph_score
    out["graph_fragmentation_risk_threshold"] = graph_threshold
    out["graph_fragmentation_risk"] = graph_labels

    out["type_id"] = out[DIRECT_LABEL_FIELDS].agg("__".join, axis=1)
    out["status"] = "ok"
    out["error"] = ""
    return out

def _zscore_for_difficulty(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    median = float(values.median()) if values.notna().any() else 0.0
    q25 = float(values.quantile(0.25)) if values.notna().any() else median
    q75 = float(values.quantile(0.75)) if values.notna().any() else median
    scale = q75 - q25
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = float(values.std(ddof=0))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return ((values.fillna(median) - median) / scale).clip(-6.0, 6.0)

def signed_log_difficulty_score(scores: pd.Series) -> pd.Series:
    values = pd.to_numeric(scores, errors="coerce").astype(float)
    transformed = np.sign(values) * np.log1p(np.abs(values))
    return pd.Series(transformed.to_numpy(dtype=float), index=scores.index, name="signed_log_constraint_difficulty_score")

def jenks_natural_breaks(values: pd.Series, n_classes: int) -> list[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float).sort_values().to_numpy()
    n = len(clean)
    if n == 0:
        return [0.0] * (n_classes + 1)
    unique = np.unique(clean)
    if len(unique) <= n_classes:
        breaks = [float(unique[0])]
        breaks.extend(float(v) for v in unique[1:])
        while len(breaks) < n_classes:
            breaks.append(float(unique[-1]))
        breaks.append(float(unique[-1]))
        return breaks[: n_classes + 1]

    lower = np.zeros((n + 1, n_classes + 1), dtype=np.int32)
    variance = np.full((n + 1, n_classes + 1), np.inf, dtype=float)
    for class_index in range(1, n_classes + 1):
        lower[1, class_index] = 1
        variance[1, class_index] = 0.0

    for end_index in range(2, n + 1):
        sum_values = 0.0
        sum_squares = 0.0
        weight = 0.0
        current_variance = 0.0
        for offset in range(1, end_index + 1):
            start_index = end_index - offset + 1
            value = clean[start_index - 1]
            sum_squares += value * value
            sum_values += value
            weight += 1.0
            current_variance = sum_squares - (sum_values * sum_values) / weight
            previous_index = start_index - 1
            if previous_index == 0:
                continue
            for class_index in range(2, n_classes + 1):
                candidate = current_variance + variance[previous_index, class_index - 1]
                if variance[end_index, class_index] >= candidate:
                    lower[end_index, class_index] = start_index
                    variance[end_index, class_index] = candidate
        lower[end_index, 1] = 1
        variance[end_index, 1] = current_variance

    breaks = [float(clean[0])] * (n_classes + 1)
    breaks[n_classes] = float(clean[-1])
    end_index = n
    class_index = n_classes
    while class_index >= 2:
        split_index = int(lower[end_index, class_index] - 2)
        breaks[class_index - 1] = float(clean[split_index])
        end_index = int(lower[end_index, class_index] - 1)
        class_index -= 1
    return breaks

def _inverse_signed_log(value: float) -> float:
    return float(np.sign(value) * np.expm1(abs(value)))

def constraint_difficulty_thresholds(scores: pd.Series) -> dict[str, float]:
    """Return signed-log Jenks cut points for oracle difficulty labels."""

    clean = pd.to_numeric(scores, errors="coerce").dropna()
    if clean.empty:
        return {
            "policy": CONSTRAINT_DIFFICULTY_THRESHOLD_POLICY,
            "easy_max": 0.0,
            "moderate_max": 0.0,
            "hard_max": 0.0,
            "easy_max_signed_log": 0.0,
            "moderate_max_signed_log": 0.0,
            "hard_max_signed_log": 0.0,
        }
    signed = signed_log_difficulty_score(clean)
    breaks = jenks_natural_breaks(signed, len(CONSTRAINT_DIFFICULTY_CLASSES))
    return {
        "policy": CONSTRAINT_DIFFICULTY_THRESHOLD_POLICY,
        "signed_log_breaks": [float(value) for value in breaks],
        "raw_score_breaks_approx": [_inverse_signed_log(value) for value in breaks],
        "easy_max": _inverse_signed_log(breaks[1]),
        "moderate_max": _inverse_signed_log(breaks[2]),
        "hard_max": _inverse_signed_log(breaks[3]),
        "easy_max_signed_log": float(breaks[1]),
        "moderate_max_signed_log": float(breaks[2]),
        "hard_max_signed_log": float(breaks[3]),
    }

def apply_constraint_difficulty_thresholds(
    scores: pd.Series,
    thresholds: dict[str, float],
) -> pd.Series:
    easy_max = float(thresholds.get("easy_max_signed_log", thresholds.get("easy_max", 0.0)))
    moderate_max = float(thresholds.get("moderate_max_signed_log", thresholds.get("moderate_max", easy_max)))
    hard_max = float(thresholds.get("hard_max_signed_log", thresholds.get("hard_max", moderate_max)))
    values = signed_log_difficulty_score(pd.to_numeric(scores, errors="coerce")).fillna(easy_max)
    labels = pd.Series("Moderate", index=scores.index, dtype=object)
    labels.loc[values <= easy_max] = "Easy"
    labels.loc[(values > easy_max) & (values <= moderate_max)] = "Moderate"
    labels.loc[(values > moderate_max) & (values <= hard_max)] = "Hard"
    labels.loc[values > hard_max] = "Extreme"
    return labels


if __name__ == "__main__":
    raise SystemExit(main())
