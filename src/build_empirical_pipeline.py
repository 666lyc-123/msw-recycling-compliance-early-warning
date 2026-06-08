from __future__ import annotations

import json
import math
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from scipy.stats import norm
import shap


BASE = Path(r"D:\VC code\MDPI2")
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
TABLES = BASE / "outputs" / "tables"
FIGURES = BASE / "outputs" / "figures"
DOCS = BASE / "docs"
for folder in (RAW, PROCESSED, TABLES, FIGURES, DOCS):
    folder.mkdir(parents=True, exist_ok=True)

FETCH_FAILURES: list[dict[str, str]] = []

EU27 = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "EL": "Greece",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LV": "Latvia",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
}

# Country list extracted from the 2023 European Commission country early-warning
# reports for the 2025 municipal waste 55% preparation-for-reuse/recycling target.
# The report package states that 18 Member States are at risk of missing one or
# both 2025 targets; for this paper we code this as an external early-warning
# label rather than a ground-truth failure outcome.
EC_EEA_2025_RISK = {
    "BG",
    "HR",
    "CY",
    "EE",
    "FI",
    "FR",
    "EL",
    "HU",
    "IE",
    "LV",
    "LT",
    "MT",
    "PL",
    "PT",
    "RO",
    "SK",
    "ES",
    "SE",
}

YEARS = list(range(2005, 2024))
SSL_CTX = ssl._create_unverified_context()


@dataclass(frozen=True)
class EurostatSpec:
    variable: str
    dataset: str
    filters: dict[str, str]
    unit: str
    definition: str
    source_code: str
    leakage_role: str


SPECS = [
    EurostatSpec(
        "recycling_rate",
        "cei_wm011",
        {"wst_oper": "RCY", "unit": "PC"},
        "%",
        "Recycling rate of municipal waste.",
        "Eurostat cei_wm011",
        "outcome_only",
    ),
    EurostatSpec(
        "waste_generated_kg_pc",
        "env_wasmun",
        {"wst_oper": "GEN", "unit": "KG_HAB"},
        "kg per capita",
        "Municipal waste generated per inhabitant.",
        "Eurostat env_wasmun",
        "safe_predictor",
    ),
    EurostatSpec(
        "landfill_kg_pc",
        "env_wasmun",
        {"wst_oper": "DSP_L_OTH", "unit": "KG_HAB"},
        "kg per capita",
        "Municipal waste disposed through landfill and other disposal operations per inhabitant.",
        "Eurostat env_wasmun",
        "lagged_predictor",
    ),
    EurostatSpec(
        "incineration_kg_pc",
        "env_wasmun",
        {"wst_oper": "DSP_I_RCV_E", "unit": "KG_HAB"},
        "kg per capita",
        "Municipal waste treated through disposal-incineration and energy recovery per inhabitant.",
        "Eurostat env_wasmun",
        "lagged_predictor",
    ),
    EurostatSpec(
        "material_recycling_kg_pc",
        "env_wasmun",
        {"wst_oper": "RCY_M", "unit": "KG_HAB"},
        "kg per capita",
        "Municipal waste material recycling per inhabitant.",
        "Eurostat env_wasmun",
        "leakage_if_contemporaneous",
    ),
    EurostatSpec(
        "compost_digest_kg_pc",
        "env_wasmun",
        {"wst_oper": "RCY_C_D", "unit": "KG_HAB"},
        "kg per capita",
        "Municipal waste composting and digestion per inhabitant.",
        "Eurostat env_wasmun",
        "leakage_if_contemporaneous",
    ),
    EurostatSpec(
        "circular_material_use_rate",
        "cei_srm030",
        {"unit": "PC"},
        "%",
        "Circular material use rate.",
        "Eurostat cei_srm030",
        "safe_predictor",
    ),
    EurostatSpec(
        "resource_productivity",
        "cei_pc030",
        {"unit": "EUR_KG_CLV15"},
        "EUR per kg, chain linked volumes 2015",
        "Resource productivity.",
        "Eurostat cei_pc030",
        "safe_predictor",
    ),
    EurostatSpec(
        "environmental_tax_gdp",
        "env_ac_tax",
        {"tax": "ENV", "unit": "PC_GDP"},
        "% GDP",
        "Total environmental taxes as percentage of GDP.",
        "Eurostat env_ac_tax",
        "safe_predictor",
    ),
    EurostatSpec(
        "gdp_pc_eur",
        "nama_10_pc",
        {"na_item": "B1GQ", "unit": "CP_EUR_HAB"},
        "euro per capita",
        "GDP per capita at current prices.",
        "Eurostat nama_10_pc",
        "safe_predictor",
    ),
    EurostatSpec(
        "household_consumption_pc_eur",
        "nama_10_pc",
        {"na_item": "P31_S14_S15", "unit": "CP_EUR_HAB"},
        "euro per capita",
        "Household and NPISH final consumption expenditure per capita.",
        "Eurostat nama_10_pc",
        "safe_predictor",
    ),
    EurostatSpec(
        "population_density",
        "demo_r_d3dens",
        {"unit": "PER_KM2"},
        "persons per km2",
        "Population density.",
        "Eurostat demo_r_d3dens",
        "safe_predictor",
    ),
]


def eurostat_json(dataset: str, filters: dict[str, str], cache_name: str) -> dict:
    path = RAW / cache_name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    params = {"format": "JSON", "lang": "en"}
    params.update(filters)
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?"
        + urlencode(params, doseq=True)
    )
    last_error = None
    for attempt in range(12):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Connection": "close"})
            with urlopen(req, timeout=90, context=SSL_CTX) as resp:
                data = json.load(resp)
            path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.55)
            return data
        except Exception as exc:  # Eurostat occasionally drops SSL connections.
            last_error = exc
            time.sleep(2.0 + attempt * 1.5)
    raise RuntimeError(f"Eurostat fetch failed for {dataset}: {last_error}")


def eurostat_json_batch(dataset: str, filters: dict[str, str], cache_name: str) -> dict:
    path = RAW / cache_name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    params = {"format": "JSON", "lang": "en", "geo": list(EU27.keys()), "time": [str(y) for y in YEARS]}
    params.update(filters)
    url = (
        f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?"
        + urlencode(params, doseq=True)
    )
    last_error = None
    for attempt in range(12):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Connection": "close"})
            with urlopen(req, timeout=120, context=SSL_CTX) as resp:
                data = json.load(resp)
            path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.8)
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2.0 + attempt * 1.5)
    raise RuntimeError(f"Eurostat batch fetch failed for {dataset}: {last_error}")


def jsonstat_to_frame(data: dict) -> pd.DataFrame:
    dims = data["id"]
    sizes = data["size"]
    cat_codes = []
    cat_labels = []
    for dim in dims:
        cat = data["dimension"][dim]["category"]
        ordered = sorted(cat["index"].items(), key=lambda kv: kv[1])
        codes = [code for code, _ in ordered]
        labels = cat.get("label", {})
        cat_codes.append(codes)
        cat_labels.append([labels.get(code, code) for code in codes])
    rows = []
    values = data.get("value", {})
    total = int(np.prod(sizes))
    for linear in range(total):
        if str(linear) not in values:
            continue
        remainder = linear
        indices = []
        for size in reversed(sizes):
            indices.append(remainder % size)
            remainder //= size
        indices = list(reversed(indices))
        row = {}
        for dim, idx, codes, labels in zip(dims, indices, cat_codes, cat_labels):
            row[dim] = codes[idx]
            row[f"{dim}_label"] = labels[idx]
        row["value"] = values[str(linear)]
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_variable(spec: EurostatSpec) -> pd.DataFrame:
    try:
        data = eurostat_json_batch(spec.dataset, spec.filters, f"{spec.dataset}_{spec.variable}_EU27_2005_2023.json")
    except Exception as exc:
        FETCH_FAILURES.append(
            {
                "variable": spec.variable,
                "dataset": spec.dataset,
                "geo": "EU27_BATCH",
                "error": repr(exc),
            }
        )
        return pd.DataFrame(columns=["geo", "country", "year", spec.variable])
    frame = jsonstat_to_frame(data)
    if frame.empty:
        return pd.DataFrame(columns=["geo", "country", "year", spec.variable])
    frame["country"] = frame["geo"].map(EU27)
    frame["year"] = frame["time"].astype(int)
    frame = frame[frame["geo"].isin(EU27) & frame["year"].between(min(YEARS), max(YEARS))]
    return frame[["geo", "country", "year", "value"]].rename(columns={"value": spec.variable}).reset_index(drop=True)


def build_panel() -> pd.DataFrame:
    panel = pd.MultiIndex.from_product([EU27.keys(), YEARS], names=["geo", "year"]).to_frame(index=False)
    panel["country"] = panel["geo"].map(EU27)
    for spec in SPECS:
        frame = fetch_variable(spec)
        panel = panel.merge(frame[["geo", "year", spec.variable]], on=["geo", "year"], how="left")
    panel = panel.sort_values(["geo", "year"]).reset_index(drop=True)
    panel["landfill_rate_proxy"] = 100 * panel["landfill_kg_pc"] / panel["waste_generated_kg_pc"]
    panel["incineration_rate_proxy"] = 100 * panel["incineration_kg_pc"] / panel["waste_generated_kg_pc"]
    panel["material_recycling_rate_proxy"] = 100 * panel["material_recycling_kg_pc"] / panel["waste_generated_kg_pc"]
    panel["compost_digest_rate_proxy"] = 100 * panel["compost_digest_kg_pc"] / panel["waste_generated_kg_pc"]
    for col in [
        "recycling_rate",
        "waste_generated_kg_pc",
        "landfill_rate_proxy",
        "incineration_rate_proxy",
        "material_recycling_rate_proxy",
        "compost_digest_rate_proxy",
        "circular_material_use_rate",
        "resource_productivity",
        "environmental_tax_gdp",
        "gdp_pc_eur",
        "household_consumption_pc_eur",
        "population_density",
    ]:
        panel[f"{col}_lag1"] = panel.groupby("geo")[col].shift(1)
        panel[f"{col}_lag2"] = panel.groupby("geo")[col].shift(2)
    panel["recycling_rate_trend_3y"] = panel.groupby("geo")["recycling_rate"].diff(3) / 3
    panel["landfill_rate_trend_3y"] = panel.groupby("geo")["landfill_rate_proxy"].diff(3) / 3
    panel["incineration_rate_trend_3y"] = panel.groupby("geo")["incineration_rate_proxy"].diff(3) / 3
    panel["ec_eea_2025_risk_label"] = panel["geo"].isin(EC_EEA_2025_RISK).astype(int)
    return panel


SAFE_FEATURES = [
    "recycling_rate_lag1",
    "recycling_rate_lag2",
    "waste_generated_kg_pc_lag1",
    "landfill_rate_proxy_lag1",
    "landfill_rate_trend_3y",
    "incineration_rate_proxy_lag1",
    "incineration_rate_trend_3y",
    "circular_material_use_rate_lag1",
    "resource_productivity_lag1",
    "environmental_tax_gdp_lag1",
    "gdp_pc_eur_lag1",
    "household_consumption_pc_eur_lag1",
    "population_density_lag1",
    "recycling_rate_trend_3y",
]

LEAKAGE_STRESS_FEATURES = SAFE_FEATURES + [
    "material_recycling_rate_proxy_lag1",
    "compost_digest_rate_proxy_lag1",
]


def make_supervised(panel: pd.DataFrame, horizon: int, feature_set: str = "safe") -> tuple[pd.DataFrame, list[str]]:
    df = panel.copy()
    df["target_year"] = df["year"] + horizon
    df[f"y_h{horizon}"] = df.groupby("geo")["recycling_rate"].shift(-horizon)
    features = SAFE_FEATURES if feature_set == "safe" else LEAKAGE_STRESS_FEATURES
    keep = ["geo", "country", "year", "target_year", f"y_h{horizon}"] + features
    out = df[keep].copy()
    out = out[out[f"y_h{horizon}"].notna()].reset_index(drop=True)
    return out, features


def model_specs():
    return {
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "ElasticNet": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 2, 20), cv=3, max_iter=20000),
        "RandomForest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=250, learning_rate=0.04, max_depth=2),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=7,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        ),
    }


def build_pipeline(estimator, numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def last_observation_predict(test: pd.DataFrame) -> np.ndarray:
    values = test["recycling_rate_lag1"].to_numpy(dtype=float)
    return values


def trend_predict(train: pd.DataFrame, test: pd.DataFrame, y_col: str) -> np.ndarray:
    preds = []
    global_slope = np.polyfit(train["year"], train[y_col], 1)[0] if train["year"].nunique() > 1 else 0.0
    for _, row in test.iterrows():
        hist = train[train["geo"] == row["geo"]].dropna(subset=[y_col])
        if len(hist) >= 4 and hist["year"].nunique() >= 4:
            slope, intercept = np.polyfit(hist["year"], hist[y_col], 1)
            pred = slope * row["target_year"] + intercept
        else:
            pred = row["recycling_rate_lag1"] + global_slope * (row["target_year"] - row["year"])
        preds.append(pred)
    return np.array(preds)


def fill_prediction_nans(pred: np.ndarray, train: pd.DataFrame, y_col: str) -> np.ndarray:
    out = np.array(pred, dtype=float, copy=True)
    fallback = float(train[y_col].mean())
    out[~np.isfinite(out)] = fallback
    return out


def rolling_evaluation(panel: pd.DataFrame, horizon: int, feature_set: str = "safe") -> tuple[pd.DataFrame, pd.DataFrame]:
    df, features = make_supervised(panel, horizon, feature_set)
    y_col = f"y_h{horizon}"
    prediction_rows = []
    metric_rows = []
    models = model_specs()
    for target_year in range(2016, 2024):
        test = df[df["target_year"] == target_year].copy()
        train = df[df["target_year"] < target_year].copy()
        if test.empty or len(train) < 80:
            continue
        y_test = test[y_col].to_numpy()
        baseline_preds = {
            "LastObservation": fill_prediction_nans(last_observation_predict(test), train, y_col),
            "CountryTrend": fill_prediction_nans(trend_predict(train, test, y_col), train, y_col),
        }
        for model_name, pred in baseline_preds.items():
            for idx, value in zip(test.index, pred):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "train_end_target_year": target_year - 1,
                        "target_year": target_year,
                        "geo": test.loc[idx, "geo"],
                        "country": test.loc[idx, "country"],
                        "actual": test.loc[idx, y_col],
                        "predicted": float(np.clip(value, 0, 100)),
                    }
                )
        X_train, X_test = train[features], test[features]
        for model_name, estimator in models.items():
            pipe = build_pipeline(estimator, features)
            pipe.fit(X_train, train[y_col])
            pred = fill_prediction_nans(pipe.predict(X_test), train, y_col)
            pred = np.clip(pred, 0, 100)
            for idx, value in zip(test.index, pred):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "train_end_target_year": target_year - 1,
                        "target_year": target_year,
                        "geo": test.loc[idx, "geo"],
                        "country": test.loc[idx, "country"],
                        "actual": test.loc[idx, y_col],
                        "predicted": float(value),
                    }
                )
    preds = pd.DataFrame(prediction_rows)
    for (model, h, fs), group in preds.groupby(["model", "horizon", "feature_set"]):
        group = group.dropna(subset=["actual", "predicted"])
        err = group["predicted"] - group["actual"]
        metric_rows.append(
            {
                "model": model,
                "horizon": h,
                "feature_set": fs,
                "n": len(group),
                "rmse": math.sqrt(mean_squared_error(group["actual"], group["predicted"])),
                "mae": mean_absolute_error(group["actual"], group["predicted"]),
                "r2": r2_score(group["actual"], group["predicted"]),
                "rank_corr": group[["actual", "predicted"]].corr(method="spearman").iloc[0, 1],
                "bias": err.mean(),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(["horizon", "feature_set", "mae"])
    return preds, metrics


def train_final_model(panel: pd.DataFrame, horizon: int, feature_set: str = "safe", model_name: str = "XGBoost"):
    df, features = make_supervised(panel, horizon, feature_set)
    y_col = f"y_h{horizon}"
    train = df[df["target_year"] <= 2023].copy()
    pipe = build_pipeline(model_specs()[model_name], features)
    pipe.fit(train[features], train[y_col])
    return pipe, features


def residual_sigma(predictions: pd.DataFrame, model_name: str, horizon: int, feature_set: str = "safe") -> float:
    subset = predictions[
        (predictions["model"] == model_name)
        & (predictions["horizon"] == horizon)
        & (predictions["feature_set"] == feature_set)
    ].dropna(subset=["actual", "predicted"])
    if subset.empty:
        return 5.0
    residual = subset["actual"] - subset["predicted"]
    sigma = float(residual.std(ddof=1))
    return max(sigma, 1.5)


def forecast_target_year(panel: pd.DataFrame, target_year: int, model_name: str, sigma: float) -> pd.DataFrame:
    horizon = target_year - 2023
    if horizon <= 0:
        raise ValueError("Target year must be after 2023 for final forecast.")
    # Use a one-year model recursively for 2025 when needed; longer targets are
    # scenario stress tests in the manuscript rather than hard forecasts.
    use_horizon = min(horizon, 2)
    pipe, features = train_final_model(panel, use_horizon, "safe", model_name)
    base = panel[panel["year"] == 2023].copy()
    base["target_year"] = 2023 + use_horizon
    pred = np.clip(pipe.predict(base[features]), 0, 100)
    out = base[["geo", "country"]].copy()
    out["target_year"] = target_year
    out["predicted_recycling_rate"] = pred
    target = {2025: 55, 2030: 60, 2035: 65}[target_year]
    out["target_rate"] = target
    out["predicted_gap"] = out["predicted_recycling_rate"] - target
    out["prediction_sigma"] = sigma
    out["prediction_interval_lower"] = np.clip(out["predicted_recycling_rate"] - 1.96 * sigma, 0, 100)
    out["prediction_interval_upper"] = np.clip(out["predicted_recycling_rate"] + 1.96 * sigma, 0, 100)
    out["risk_probability"] = norm.cdf((target - out["predicted_recycling_rate"]) / sigma)
    out["model"] = model_name
    out["ec_eea_2025_risk_label"] = out["geo"].isin(EC_EEA_2025_RISK).astype(int)
    return out.sort_values("risk_probability", ascending=False)


def external_validation(forecast_2025: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.30, 0.81, 0.05):
        pred_label = (forecast_2025["risk_probability"] >= threshold).astype(int)
        true = forecast_2025["ec_eea_2025_risk_label"].astype(int)
        tp = int(((pred_label == 1) & (true == 1)).sum())
        fp = int(((pred_label == 1) & (true == 0)).sum())
        tn = int(((pred_label == 0) & (true == 0)).sum())
        fn = int(((pred_label == 0) & (true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
        try:
            auc = roc_auc_score(true, forecast_2025["risk_probability"])
        except ValueError:
            auc = np.nan
        brier = brier_score_loss(true, forecast_2025["risk_probability"])
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "brier_score": brier,
            }
        )
    return pd.DataFrame(rows)


def leave_one_country_out(panel: pd.DataFrame, horizon: int, model_name: str) -> pd.DataFrame:
    df, features = make_supervised(panel, horizon, "safe")
    y_col = f"y_h{horizon}"
    rows = []
    for geo, country in EU27.items():
        train = df[df["geo"] != geo].copy()
        test = df[(df["geo"] == geo) & (df["target_year"] >= 2016)].copy()
        if test.empty or len(train) < 80:
            continue
        pipe = build_pipeline(model_specs()[model_name], features)
        pipe.fit(train[features], train[y_col])
        pred = np.clip(fill_prediction_nans(pipe.predict(test[features]), train, y_col), 0, 100)
        rows.append(
            {
                "geo": geo,
                "country": country,
                "horizon": horizon,
                "model": model_name,
                "n": len(test),
                "rmse": math.sqrt(mean_squared_error(test[y_col], pred)),
                "mae": mean_absolute_error(test[y_col], pred),
                "r2": r2_score(test[y_col], pred) if len(test) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def scenario_sensitivity(panel: pd.DataFrame, model_name: str, sigma: float) -> pd.DataFrame:
    pipe, features = train_final_model(panel, 2, "safe", model_name)
    base = panel[panel["year"] == 2023].copy()
    target = 55
    med_landfill = float(base["landfill_rate_proxy_lag1"].median(skipna=True))
    med_cmu = float(base["circular_material_use_rate_lag1"].median(skipna=True))
    p75_cmu = float(base["circular_material_use_rate_lag1"].quantile(0.75))
    p75_resource = float(base["resource_productivity_lag1"].quantile(0.75))
    p25_incin = float(base["incineration_rate_proxy_lag1"].quantile(0.25))

    def score(frame, scenario):
        pred = np.clip(pipe.predict(frame[features]), 0, 100)
        risk = norm.cdf((target - pred) / sigma)
        return pd.DataFrame({"geo": frame["geo"], "country": frame["country"], "scenario": scenario, "predicted_recycling_rate": pred, "risk_probability": risk})

    frames = [score(base, "Baseline")]
    s1 = base.copy()
    mask = s1["landfill_rate_proxy_lag1"] > med_landfill
    s1.loc[mask, "landfill_rate_proxy_lag1"] = med_landfill
    frames.append(score(s1, "S1 landfill transition"))
    s2 = base.copy()
    s2["circular_material_use_rate_lag1"] = np.maximum(s2["circular_material_use_rate_lag1"], p75_cmu)
    s2["resource_productivity_lag1"] = np.maximum(s2["resource_productivity_lag1"], p75_resource)
    frames.append(score(s2, "S2 circular capacity"))
    s3 = base.copy()
    mask = s3["incineration_rate_proxy_lag1"] > s3["incineration_rate_proxy_lag1"].median(skipna=True)
    s3.loc[mask, "incineration_rate_proxy_lag1"] = p25_incin
    frames.append(score(s3, "S3 incineration lock-in mitigation"))
    s4 = s2.copy()
    mask = s4["landfill_rate_proxy_lag1"] > med_landfill
    s4.loc[mask, "landfill_rate_proxy_lag1"] = med_landfill
    mask = s4["incineration_rate_proxy_lag1"] > s4["incineration_rate_proxy_lag1"].median(skipna=True)
    s4.loc[mask, "incineration_rate_proxy_lag1"] = p25_incin
    frames.append(score(s4, "S4 combined transition package"))
    out = pd.concat(frames, ignore_index=True)
    baseline = out[out["scenario"] == "Baseline"][["geo", "risk_probability"]].rename(columns={"risk_probability": "baseline_risk"})
    out = out.merge(baseline, on="geo", how="left")
    out["risk_change_vs_baseline"] = out["risk_probability"] - out["baseline_risk"]
    return out


def explainability_outputs(panel: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df, features = make_supervised(panel, 2, "safe")
    y_col = "y_h2"
    train = df[df["target_year"] <= 2023].copy()
    pipe = build_pipeline(model_specs()[model_name], features)
    pipe.fit(train[features], train[y_col])
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    x_trans = pre.transform(train[features])
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_trans)
        shap_arr = np.asarray(shap_values)
        importance = pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": np.abs(shap_arr).mean(axis=0),
                "mean_shap": shap_arr.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(TABLES / "shap_global_importance.csv", index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(8, 6))
        top = importance.head(12).sort_values("mean_abs_shap")
        ax.barh(top["feature"], top["mean_abs_shap"], color="#1f4e79")
        ax.set_title("Global SHAP importance for two-year recycling-rate forecasts")
        ax.set_xlabel("Mean absolute SHAP value")
        fig.tight_layout()
        fig.savefig(FIGURES / "figure5_shap_global_importance.png", dpi=220)
        plt.close(fig)
        return importance
    except Exception:
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        else:
            importance = pd.DataFrame({"feature": features, "importance": np.nan})
        importance.to_csv(TABLES / "model_feature_importance.csv", index=False, encoding="utf-8-sig")
        return importance


def descriptive_tables(panel: pd.DataFrame) -> None:
    vars_for_stats = [
        "recycling_rate",
        "waste_generated_kg_pc",
        "landfill_rate_proxy",
        "incineration_rate_proxy",
        "circular_material_use_rate",
        "resource_productivity",
        "environmental_tax_gdp",
        "gdp_pc_eur",
        "household_consumption_pc_eur",
        "population_density",
    ]
    stats = panel[vars_for_stats].agg(["count", "mean", "std", "min", "median", "max"]).T.reset_index()
    stats = stats.rename(columns={"index": "variable"})
    stats.to_csv(TABLES / "table2_descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    miss = panel[["geo", "country", "year"] + vars_for_stats].isna().sum().reset_index()
    miss.columns = ["variable", "missing_count"]
    miss["missing_share"] = miss["missing_count"] / len(panel)
    miss.to_csv(TABLES / "missingness_report.csv", index=False, encoding="utf-8-sig")
    dictionary = pd.DataFrame(
        [
            {
                "variable": s.variable,
                "definition": s.definition,
                "unit": s.unit,
                "source_code": s.source_code,
                "leakage_role": s.leakage_role,
                "downloaded_from": "Eurostat dissemination API",
            }
            for s in SPECS
        ]
        + [
            {
                "variable": "ec_eea_2025_risk_label",
                "definition": "External early-warning label for Member States identified as at risk in the 2023 EC-EEA package.",
                "unit": "0/1",
                "source_code": "European Commission/EEA 2023 early-warning assessment",
                "leakage_role": "external_validation_only",
                "downloaded_from": "EC/EEA country reports and EEA publication",
            }
        ]
    )
    dictionary.to_csv(TABLES / "table1_variable_dictionary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(BASE / "data" / "processed" / "empirical_documentation.xlsx", engine="openpyxl") as writer:
        dictionary.to_excel(writer, sheet_name="variable_dictionary", index=False)
        stats.to_excel(writer, sheet_name="descriptive_statistics", index=False)
        miss.to_excel(writer, sheet_name="missingness", index=False)


def figures(panel: pd.DataFrame, forecast_2025: pd.DataFrame, metrics: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(11, 6))
    for geo, grp in panel.dropna(subset=["recycling_rate"]).groupby("geo"):
        ax.plot(grp["year"], grp["recycling_rate"], color="#7f8c8d", alpha=0.45, linewidth=1)
    eu = panel.groupby("year")["recycling_rate"].mean()
    ax.plot(eu.index, eu.values, color="#0b4f6c", linewidth=3, label="EU-27 mean")
    for target, val in [(2025, 55), (2030, 60), (2035, 65)]:
        ax.axhline(val, linestyle="--", linewidth=1.3, label=f"{target} target ({val}%)")
    ax.set_title("EU-27 municipal waste recycling trajectories and policy targets")
    ax.set_xlabel("Year")
    ax.set_ylabel("Recycling rate (%)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_recycling_trajectories.png", dpi=220)
    plt.close(fig)

    latest = panel[panel["year"] == 2023].sort_values("recycling_rate")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = np.where(latest["recycling_rate"] >= 55, "#2a9d8f", "#d1495b")
    ax.barh(latest["country"], latest["recycling_rate"] - 55, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("2023 distance to the EU 2025 municipal waste recycling target")
    ax.set_xlabel("Distance to 55% target (percentage points)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_2023_distance_to_target.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df = forecast_2025.sort_values("risk_probability", ascending=False)
    colors = np.where(plot_df["ec_eea_2025_risk_label"] == 1, "#d1495b", "#2a9d8f")
    ax.bar(plot_df["geo"], plot_df["risk_probability"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("Model-based 2025 non-compliance risk and EC-EEA external labels")
    ax.set_xlabel("Member State")
    ax.set_ylabel("Risk probability proxy")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_2025_risk_vs_external_label.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    m = metrics[(metrics["feature_set"] == "safe") & (metrics["horizon"] == 1)].sort_values("mae")
    ax.bar(m["model"], m["mae"], color="#1f4e79")
    ax.set_title("Rolling-origin one-year forecast MAE by model")
    ax.set_ylabel("MAE (percentage points)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_model_mae.png", dpi=220)
    plt.close(fig)


def summary_markdown(panel: pd.DataFrame, metrics: pd.DataFrame, forecast_2025: pd.DataFrame, validation: pd.DataFrame) -> None:
    best = metrics[(metrics["feature_set"] == "safe") & (metrics["horizon"] == 1)].sort_values("mae").iloc[0]
    trend = metrics[(metrics["feature_set"] == "safe") & (metrics["horizon"] == 1) & (metrics["model"] == "CountryTrend")].iloc[0]
    best_validation = validation.sort_values("f1", ascending=False).iloc[0]
    n_obs = int(panel["recycling_rate"].notna().sum())
    high_risk = forecast_2025[forecast_2025["risk_probability"] >= best_validation["threshold"]]
    text = f"""# Empirical Upgrade Snapshot

Generated by `src/build_empirical_pipeline.py`.

## Data

- Panel scope: EU-27, 2005-2023.
- Non-missing municipal recycling-rate observations: {n_obs}.
- Data sources: Eurostat dissemination API plus EC-EEA 2025 early-warning external labels.
- Leakage control: contemporaneous material-recycling and composting variables are excluded from the main model; only lagged stress-test versions are documented separately.

## Preliminary Rolling-Origin Result

- Best one-year safe-feature model by MAE: **{best['model']}**.
- MAE: **{best['mae']:.2f}** percentage points.
- Country-trend baseline MAE: **{trend['mae']:.2f}** percentage points.
- Relative MAE change vs. country trend: **{(trend['mae'] - best['mae']) / trend['mae'] * 100:.1f}%**.

## External Validation Against EC-EEA 2025 Early Warning

- Best F1 threshold on model risk probability: **{best_validation['threshold']:.2f}**.
- Precision: **{best_validation['precision']:.2f}**.
- Recall: **{best_validation['recall']:.2f}**.
- F1: **{best_validation['f1']:.2f}**.
- AUC: **{best_validation['auc']:.2f}**.
- Brier score: **{best_validation['brier_score']:.3f}**.
- Countries flagged by the model at that threshold: {', '.join(high_risk['geo'])}.

## Caution

These are automatically generated preliminary results. They are good enough to replace the current concept-only Results section with real tables and figures, but the manuscript should still frame 2030/2035 as scenario projections and avoid causal claims from SHAP or scenario changes.
"""
    (DOCS / "empirical_upgrade_snapshot.md").write_text(text, encoding="utf-8")


def main() -> None:
    panel = build_panel()
    if FETCH_FAILURES:
        pd.DataFrame(FETCH_FAILURES).to_csv(TABLES / "data_fetch_failures.csv", index=False, encoding="utf-8-sig")
    else:
        stale_failures = TABLES / "data_fetch_failures.csv"
        if stale_failures.exists():
            stale_failures.unlink()
    panel.to_csv(PROCESSED / "processed_panel.csv", index=False, encoding="utf-8-sig")
    descriptive_tables(panel)

    all_preds = []
    all_metrics = []
    for feature_set in ("safe", "leakage_stress"):
        for horizon in (1, 2):
            preds, metrics = rolling_evaluation(panel, horizon, feature_set)
            all_preds.append(preds)
            all_metrics.append(metrics)
    predictions = pd.concat(all_preds, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True).sort_values(["horizon", "feature_set", "mae"])
    predictions.to_csv(TABLES / "rolling_origin_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLES / "table3_rolling_origin_model_performance.csv", index=False, encoding="utf-8-sig")

    best_model = metrics[(metrics["feature_set"] == "safe") & (metrics["horizon"] == 2)].sort_values("mae").iloc[0]["model"]
    sigma_2025 = residual_sigma(predictions, str(best_model), 2, "safe")
    forecast_2025 = forecast_target_year(panel, 2025, str(best_model), sigma_2025)
    forecast_2025.to_csv(TABLES / "table4_2025_target_gaps_and_risk_probabilities.csv", index=False, encoding="utf-8-sig")
    validation = external_validation(forecast_2025)
    validation.to_csv(TABLES / "table5_external_validation_ec_eea.csv", index=False, encoding="utf-8-sig")
    loco = leave_one_country_out(panel, 1, str(best_model))
    loco.to_csv(TABLES / "leave_one_country_out_validation.csv", index=False, encoding="utf-8-sig")
    scenarios = scenario_sensitivity(panel, str(best_model), sigma_2025)
    scenarios.to_csv(TABLES / "table6_policy_sensitivity_scenarios.csv", index=False, encoding="utf-8-sig")
    explainability_outputs(panel, str(best_model))

    figures(panel, forecast_2025, metrics)
    summary_markdown(panel, metrics, forecast_2025, validation)

    print(f"processed_panel={PROCESSED / 'processed_panel.csv'}")
    print(f"metrics={TABLES / 'table3_rolling_origin_model_performance.csv'}")
    print(f"forecast_2025={TABLES / 'table4_2025_target_gaps_and_risk_probabilities.csv'}")
    print(f"summary={DOCS / 'empirical_upgrade_snapshot.md'}")


if __name__ == "__main__":
    main()
