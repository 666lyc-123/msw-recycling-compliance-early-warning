from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

BASE = Path(r"D:\VC code\MDPI2")
sys.path.insert(0, str(BASE / "src"))
import build_empirical_pipeline as bep  # noqa: E402
import run_decisive_revision_experiments as dre  # noqa: E402

TABLES = BASE / "outputs" / "tables"
DOCS = BASE / "docs"
TABLES.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)
TARGET = 55.0
FIXED_THRESHOLD = 0.50
BOOTSTRAP_B = 1000
RNG = np.random.default_rng(20260526)


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(BASE / "data" / "processed" / "processed_panel.csv")
    return dre.recompute_lags(panel)


def scenario_summary_at_threshold(threshold: float = FIXED_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = pd.read_csv(TABLES / "table6_policy_sensitivity_scenarios.csv")
    baseline = scenarios[scenarios["scenario"] == "Baseline"][["geo", "risk_probability"]].rename(columns={"risk_probability": "baseline_risk"})
    scenarios = scenarios.drop(columns=["baseline_risk", "risk_change_vs_baseline"], errors="ignore").merge(baseline, on="geo", how="left")
    scenarios["risk_change_vs_baseline"] = scenarios["risk_probability"] - scenarios["baseline_risk"]
    scenarios["main_threshold"] = threshold
    scenarios["high_risk_flag_main_threshold"] = (scenarios["risk_probability"] >= threshold).astype(int)
    summary = scenarios.groupby("scenario", sort=False).agg(
        mean_predicted_recycling_rate=("predicted_recycling_rate", "mean"),
        mean_risk_probability=("risk_probability", "mean"),
        high_risk_countries=("high_risk_flag_main_threshold", "sum"),
        mean_risk_change_vs_baseline=("risk_change_vs_baseline", "mean"),
    ).reset_index()
    summary["classification_threshold"] = threshold
    return scenarios, summary


def long_horizon_rolling_validation(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = [
        ("LatestObservedRate", ""),
        ("CountryTrend", ""),
        ("Ridge", "full_safe"),
        ("LightGBM", "full_safe"),
    ]
    for horizon in range(1, 8):
        for method, feature_set in methods:
            actuals, preds = [], []
            years = range(2012 + horizon, 2024)
            for target_year in years:
                base_year = target_year - horizon
                if base_year < panel["year"].min():
                    continue
                actual = panel[panel["year"] == target_year][["geo", "recycling_rate"]].dropna()
                if actual.empty:
                    continue
                if method == "LatestObservedRate":
                    pred = panel[panel["year"] == base_year][["geo", "recycling_rate"]].rename(columns={"recycling_rate": "predicted_recycling_rate"})
                elif method == "CountryTrend":
                    pred = dre.country_trend_forecast(panel, base_year, target_year)[["geo", "predicted_recycling_rate"]]
                else:
                    features = dre.FEATURE_SETS[feature_set]
                    df = dre.make_supervised_custom(panel, horizon, features)
                    y_col = f"y_h{horizon}"
                    train = df[df["target_year"] < target_year].copy()
                    test = df[df["target_year"] == target_year].copy()
                    if test.empty or len(train) < 80:
                        continue
                    pipe = bep.build_pipeline(bep.model_specs()[method], features)
                    pipe.fit(train[features], train[y_col])
                    pred_values = np.clip(pipe.predict(test[features]), 0, 100)
                    pred = pd.DataFrame({"geo": test["geo"], "predicted_recycling_rate": pred_values})
                joined = actual.merge(pred, on="geo", how="inner").dropna()
                if joined.empty:
                    continue
                actuals.extend(joined["recycling_rate"].tolist())
                preds.extend(joined["predicted_recycling_rate"].tolist())
            if actuals:
                y = np.asarray(actuals, dtype=float)
                p = np.asarray(preds, dtype=float)
                residual = y - p
                pseudo_target = (y < TARGET).astype(int)
                score = norm.cdf((TARGET - p) / max(float(np.std(residual, ddof=1)), 1.5))
                pred_label = (score >= FIXED_THRESHOLD).astype(int)
                auc = roc_auc_score(pseudo_target, score) if len(np.unique(pseudo_target)) == 2 else np.nan
                brier = brier_score_loss(pseudo_target, score) if len(np.unique(pseudo_target)) == 2 else np.nan
                rows.append({
                    "horizon": horizon,
                    "model": method,
                    "feature_set": feature_set,
                    "n": len(y),
                    "rmse": float(np.sqrt(mean_squared_error(y, p))),
                    "mae": float(mean_absolute_error(y, p)),
                    "r2": float(r2_score(y, p)),
                    "residual_sd": float(np.std(residual, ddof=1)),
                    "pseudo_target_auc": float(auc) if np.isfinite(auc) else np.nan,
                    "pseudo_target_brier": float(brier) if np.isfinite(brier) else np.nan,
                    "pseudo_target_f1_at_050": float(f1_score(pseudo_target, pred_label, zero_division=0)) if len(np.unique(pseudo_target)) == 2 else np.nan,
                })
    return pd.DataFrame(rows).sort_values(["horizon", "mae"])


def paired_delta_ci(panel: pd.DataFrame) -> pd.DataFrame:
    forecasts = pd.read_csv(TABLES / "table9_information_cutoff_country_forecasts.csv")
    rows = []
    for cutoff in sorted(forecasts["cutoff_year"].unique()):
        full = forecasts[(forecasts["cutoff_year"] == cutoff) & (forecasts["method"] == "LightGBM") & (forecasts["feature_set"] == "full_safe")][["geo", "ec_eea_2025_risk_label", "risk_probability"]].rename(columns={"risk_probability": "risk_full"})
        base = forecasts[(forecasts["cutoff_year"] == cutoff) & (forecasts["method"] == "LatestObservedRate") & (forecasts["feature_set"].fillna("") == "")][["geo", "risk_probability"]].rename(columns={"risk_probability": "risk_base"})
        merged = full.merge(base, on="geo", how="inner")
        y = merged["ec_eea_2025_risk_label"].astype(int).to_numpy()
        sf = merged["risk_full"].to_numpy(float)
        sb = merged["risk_base"].to_numpy(float)
        n = len(y)

        def metrics(score):
            pred = (score >= FIXED_THRESHOLD).astype(int)
            return {
                "auc": roc_auc_score(y, score),
                "f1": f1_score(y, pred, zero_division=0),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "brier": brier_score_loss(y, score),
            }

        mf = metrics(sf)
        mb = metrics(sb)
        deltas = {m: mf[m] - mb[m] for m in mf}
        boot = {m: [] for m in mf}
        for _ in range(BOOTSTRAP_B):
            idx = RNG.integers(0, n, n)
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            sff = sf[idx]
            sbb = sb[idx]
            pf = (sff >= FIXED_THRESHOLD).astype(int)
            pb = (sbb >= FIXED_THRESHOLD).astype(int)
            boot["auc"].append(roc_auc_score(yy, sff) - roc_auc_score(yy, sbb))
            boot["f1"].append(f1_score(yy, pf, zero_division=0) - f1_score(yy, pb, zero_division=0))
            boot["precision"].append(precision_score(yy, pf, zero_division=0) - precision_score(yy, pb, zero_division=0))
            boot["recall"].append(recall_score(yy, pf, zero_division=0) - recall_score(yy, pb, zero_division=0))
            boot["brier"].append(brier_score_loss(yy, sff) - brier_score_loss(yy, sbb))
        row = {"cutoff_year": int(cutoff), "n": n}
        for metric, delta in deltas.items():
            arr = np.asarray(boot[metric], dtype=float)
            row[f"delta_{metric}"] = float(delta)
            row[f"delta_{metric}_ci_low"] = float(np.quantile(arr, 0.025)) if arr.size else np.nan
            row[f"delta_{metric}_ci_high"] = float(np.quantile(arr, 0.975)) if arr.size else np.nan
            if metric in {"auc", "f1", "precision", "recall"}:
                row[f"p_delta_{metric}_gt0"] = float((arr > 0).mean()) if arr.size else np.nan
            else:
                row[f"p_delta_{metric}_lt0"] = float((arr < 0).mean()) if arr.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def model_config_table() -> pd.DataFrame:
    import lightgbm, xgboost, sklearn, scipy, matplotlib
    return pd.DataFrame([
        {"component": "Imputation", "specification": "SimpleImputer(strategy='median'), fitted inside each training fold only"},
        {"component": "Scaling", "specification": "StandardScaler fitted inside each training fold through a sklearn Pipeline"},
        {"component": "Ridge model", "specification": "RidgeCV; alpha grid from 1e-3 to 1e3 with 25 logarithmic values"},
        {"component": "ElasticNet model", "specification": "ElasticNetCV; l1_ratio={0.1, 0.5, 0.9}; alpha grid from 1e-3 to 1e2; 3-fold internal CV; max_iter=20000"},
        {"component": "RandomForest model", "specification": "400 trees; min_samples_leaf=3; random_state=42"},
        {"component": "GradientBoosting model", "specification": "250 trees; learning_rate=0.04; max_depth=2; random_state=42"},
        {"component": "XGBoost model", "specification": "250 trees; max_depth=2; learning_rate=0.04; subsample=0.9; colsample_bytree=0.9; squared-error objective; random_state=42; n_jobs=1"},
        {"component": "LightGBM model", "specification": "250 trees; learning_rate=0.04; num_leaves=7; min_child_samples=8; subsample=0.9; colsample_bytree=0.9; random_state=42"},
        {"component": "External-label threshold", "specification": "Main classification threshold fixed ex ante at 0.50; threshold sweep reported only as sensitivity"},
        {"component": "Bootstrap", "specification": "Country-level paired bootstrap with 1000 resamples for incremental value intervals"},
        {"component": "Random seed", "specification": "42 for sklearn/XGBoost/LightGBM model specifications; 20260526 for revision bootstrap"},
        {"component": "Python", "specification": platform.python_version()},
        {"component": "pandas", "specification": pd.__version__},
        {"component": "numpy", "specification": np.__version__},
        {"component": "scikit-learn", "specification": sklearn.__version__},
        {"component": "LightGBM package", "specification": lightgbm.__version__},
        {"component": "XGBoost package", "specification": xgboost.__version__},
        {"component": "scipy", "specification": scipy.__version__},
        {"component": "matplotlib", "specification": matplotlib.__version__},
        {"component": "Repository", "specification": "Reproducibility package prepared locally; permanent DOI to be inserted before final journal submission"},
    ])


def write_report(scenario_summary, horizon, deltas):
    lines = ["# Revision 2 Experiments", ""]
    lines.append("## Scenario summary at fixed 0.50 threshold")
    for r in scenario_summary.itertuples():
        lines.append(f"- {r.scenario}: mean risk={r.mean_risk_probability:.3f}, high-risk countries={int(r.high_risk_countries)}, mean risk change={r.mean_risk_change_vs_baseline:.3f}.")
    lines.append("")
    lines.append("## Long-horizon rolling validation")
    best = horizon[horizon["model"].isin(["LightGBM", "Ridge", "LatestObservedRate", "CountryTrend"])].sort_values(["horizon", "mae"]).groupby("horizon").head(2)
    for r in best.itertuples():
        lines.append(f"- h={int(r.horizon)}, {r.model}: MAE={r.mae:.3f}, RMSE={r.rmse:.3f}, residual SD={r.residual_sd:.3f}, pseudo-target AUC={r.pseudo_target_auc:.3f}, Brier={r.pseudo_target_brier:.3f}.")
    lines.append("")
    lines.append("## Incremental value paired bootstrap")
    for r in deltas.itertuples():
        lines.append(f"- cutoff {int(r.cutoff_year)}: delta Brier={r.delta_brier:.3f} [{r.delta_brier_ci_low:.3f}, {r.delta_brier_ci_high:.3f}], P(delta Brier<0)={r.p_delta_brier_lt0:.3f}; delta F1={r.delta_f1:.3f} [{r.delta_f1_ci_low:.3f}, {r.delta_f1_ci_high:.3f}].")
    (DOCS / "revision2_major_response_experiments.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    panel = load_panel()
    scenario_detail, scenario_summary = scenario_summary_at_threshold(FIXED_THRESHOLD)
    scenario_detail.to_csv(TABLES / "table13_policy_sensitivity_scenarios_threshold050.csv", index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(TABLES / "table13a_policy_sensitivity_summary_threshold050.csv", index=False, encoding="utf-8-sig")

    horizon = long_horizon_rolling_validation(panel)
    horizon.to_csv(TABLES / "table14_long_horizon_rolling_validation.csv", index=False, encoding="utf-8-sig")

    deltas = paired_delta_ci(panel)
    deltas.to_csv(TABLES / "table15_incremental_value_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    config = model_config_table()
    config.to_csv(TABLES / "appendix_table_a3_model_configuration_reproducibility.csv", index=False, encoding="utf-8-sig")

    write_report(scenario_summary, horizon, deltas)
    print(scenario_summary.to_string(index=False))
    print(horizon[horizon["model"].isin(["LightGBM", "Ridge", "LatestObservedRate", "CountryTrend"])].sort_values(["horizon", "mae"]).groupby("horizon").head(2).to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
