from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


TABLES = BASE / "outputs" / "tables"
FIGURES = BASE / "outputs" / "figures"
DOCS = BASE / "docs"
for folder in (TABLES, FIGURES, DOCS):
    folder.mkdir(parents=True, exist_ok=True)

TARGET_YEAR = 2025
TARGET_RATE = 55.0
FIXED_THRESHOLD = 0.50
BOOTSTRAP_B = 300
RNG = np.random.default_rng(20260525)
CUTOFF_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

FEATURE_SETS = {
    "recycling_lags_only": [
        "recycling_rate_lag1",
        "recycling_rate_lag2",
        "recycling_rate_trend_3y",
    ],
    "recycling_lags_plus_landfill": [
        "recycling_rate_lag1",
        "recycling_rate_lag2",
        "recycling_rate_trend_3y",
        "landfill_rate_proxy_lag1",
        "landfill_rate_trend_3y",
    ],
    "treatment_path": [
        "recycling_rate_lag1",
        "recycling_rate_lag2",
        "recycling_rate_trend_3y",
        "landfill_rate_proxy_lag1",
        "landfill_rate_trend_3y",
        "incineration_rate_proxy_lag1",
        "incineration_rate_trend_3y",
        "waste_generated_kg_pc_lag1",
    ],
    "full_safe": bep.SAFE_FEATURES,
}


def load_panel() -> pd.DataFrame:
    path = BASE / "data" / "processed" / "processed_panel.csv"
    if path.exists():
        return pd.read_csv(path)
    return bep.build_panel()


def recompute_lags(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["geo", "year"]).reset_index(drop=True).copy()
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
        if col in panel:
            panel[f"{col}_lag1"] = panel.groupby("geo")[col].shift(1)
            panel[f"{col}_lag2"] = panel.groupby("geo")[col].shift(2)
    panel["recycling_rate_trend_3y"] = panel.groupby("geo")["recycling_rate"].diff(3) / 3
    panel["landfill_rate_trend_3y"] = panel.groupby("geo")["landfill_rate_proxy"].diff(3) / 3
    panel["incineration_rate_trend_3y"] = panel.groupby("geo")["incineration_rate_proxy"].diff(3) / 3
    panel["ec_eea_2025_risk_label"] = panel["geo"].isin(bep.EC_EEA_2025_RISK).astype(int)
    return panel


def make_supervised_custom(panel: pd.DataFrame, horizon: int, features: list[str]) -> pd.DataFrame:
    df = panel.copy()
    df["target_year"] = df["year"] + horizon
    df[f"y_h{horizon}"] = df.groupby("geo")["recycling_rate"].shift(-horizon)
    keep = ["geo", "country", "year", "target_year", f"y_h{horizon}"] + features
    out = df[keep].copy()
    return out[out[f"y_h{horizon}"].notna()].reset_index(drop=True)


def country_trend_forecast(panel: pd.DataFrame, cutoff_year: int, target_year: int) -> pd.DataFrame:
    rows = []
    for geo, grp in panel[panel["year"] <= cutoff_year].dropna(subset=["recycling_rate"]).groupby("geo"):
        grp = grp.sort_values("year")
        if len(grp) >= 4 and grp["year"].nunique() >= 4:
            slope, intercept = np.polyfit(grp["year"], grp["recycling_rate"], 1)
            pred = slope * target_year + intercept
        elif len(grp):
            pred = grp["recycling_rate"].iloc[-1]
        else:
            pred = np.nan
        rows.append({"geo": geo, "country": bep.EU27[geo], "predicted_recycling_rate": float(np.clip(pred, 0, 100))})
    return pd.DataFrame(rows)


def baseline_forecast(panel: pd.DataFrame, cutoff_year: int, method: str) -> pd.DataFrame:
    if method == "LatestObservedRate":
        base = panel[panel["year"] == cutoff_year][["geo", "country", "recycling_rate"]].copy()
        return base.rename(columns={"recycling_rate": "predicted_recycling_rate"})
    if method == "DistanceToTargetRule":
        base = panel[panel["year"] == cutoff_year][["geo", "country", "recycling_rate"]].copy()
        return base.rename(columns={"recycling_rate": "predicted_recycling_rate"})
    if method == "CountryTrend":
        return country_trend_forecast(panel, cutoff_year, TARGET_YEAR)
    raise ValueError(method)


def model_forecast(panel: pd.DataFrame, cutoff_year: int, model_name: str, feature_set_name: str) -> pd.DataFrame:
    horizon = TARGET_YEAR - cutoff_year
    features = FEATURE_SETS[feature_set_name]
    y_col = f"y_h{horizon}"
    df = make_supervised_custom(panel, horizon, features)
    train = df[df["target_year"] <= cutoff_year].copy()
    base = panel[panel["year"] == cutoff_year][["geo", "country"] + features].copy()
    if len(train) < 80:
        raise ValueError(f"Not enough training rows for cutoff={cutoff_year}, horizon={horizon}: {len(train)}")
    pipe = bep.build_pipeline(bep.model_specs()[model_name], features)
    pipe.fit(train[features], train[y_col])
    pred = np.clip(pipe.predict(base[features]), 0, 100)
    return pd.DataFrame({"geo": base["geo"], "country": base["country"], "predicted_recycling_rate": pred})


def rolling_residuals(panel: pd.DataFrame, horizon: int, method: str, feature_set_name: str | None = None, cutoff_year: int = 2023) -> np.ndarray:
    residuals: list[float] = []
    if method in {"LatestObservedRate", "DistanceToTargetRule", "CountryTrend"}:
        for ty in range(2014, cutoff_year + 1):
            base_year = ty - horizon
            if base_year < panel["year"].min():
                continue
            actual = panel[panel["year"] == ty][["geo", "recycling_rate"]].dropna()
            if actual.empty:
                continue
            if method in {"LatestObservedRate", "DistanceToTargetRule"}:
                pred = panel[panel["year"] == base_year][["geo", "recycling_rate"]].rename(columns={"recycling_rate": "predicted_recycling_rate"})
            else:
                pred = country_trend_forecast(panel, base_year, ty)[["geo", "predicted_recycling_rate"]]
            joined = actual.merge(pred, on="geo", how="inner").dropna()
            residuals.extend((joined["recycling_rate"] - joined["predicted_recycling_rate"]).tolist())
    else:
        features = FEATURE_SETS[feature_set_name or "full_safe"]
        df = make_supervised_custom(panel, horizon, features)
        y_col = f"y_h{horizon}"
        for ty in range(2014, cutoff_year + 1):
            test = df[df["target_year"] == ty].copy()
            train = df[df["target_year"] < ty].copy()
            if test.empty or len(train) < 80:
                continue
            pipe = bep.build_pipeline(bep.model_specs()[method], features)
            pipe.fit(train[features], train[y_col])
            pred = np.clip(pipe.predict(test[features]), 0, 100)
            residuals.extend((test[y_col].to_numpy() - pred).tolist())
    arr = np.asarray(residuals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def sigma_from_residuals(residuals: np.ndarray) -> float:
    if residuals.size < 10:
        return 5.0
    return max(float(np.std(residuals, ddof=1)), 1.5)


def enrich_forecast(frame: pd.DataFrame, cutoff_year: int, method: str, sigma: float, residual_n: int, feature_set: str = "") -> pd.DataFrame:
    out = frame.copy()
    out["cutoff_year"] = cutoff_year
    out["horizon"] = TARGET_YEAR - cutoff_year
    out["target_year"] = TARGET_YEAR
    out["target_rate"] = TARGET_RATE
    out["method"] = method
    out["feature_set"] = feature_set
    out["predicted_gap"] = out["predicted_recycling_rate"] - TARGET_RATE
    out["sigma"] = sigma
    out["residual_n"] = residual_n
    out["risk_probability"] = norm.cdf((TARGET_RATE - out["predicted_recycling_rate"]) / sigma)
    out["ec_eea_2025_risk_label"] = out["geo"].isin(bep.EC_EEA_2025_RISK).astype(int)
    return out.sort_values("risk_probability", ascending=False)


def calculate_metrics(forecast: pd.DataFrame, threshold: float = FIXED_THRESHOLD) -> dict[str, float]:
    y = forecast["ec_eea_2025_risk_label"].astype(int).to_numpy()
    score = forecast["risk_probability"].astype(float).to_numpy()
    pred_label = (score >= threshold).astype(int)
    out: dict[str, float] = {
        "threshold": threshold,
        "tp": int(((pred_label == 1) & (y == 1)).sum()),
        "fp": int(((pred_label == 1) & (y == 0)).sum()),
        "tn": int(((pred_label == 0) & (y == 0)).sum()),
        "fn": int(((pred_label == 0) & (y == 1)).sum()),
        "precision": precision_score(y, pred_label, zero_division=0),
        "recall": recall_score(y, pred_label, zero_division=0),
        "f1": f1_score(y, pred_label, zero_division=0),
        "brier_score": brier_score_loss(y, score),
    }
    try:
        out["auc"] = roc_auc_score(y, score)
    except ValueError:
        out["auc"] = np.nan
    return out


def bootstrap_ci(forecast: pd.DataFrame, threshold: float = FIXED_THRESHOLD) -> dict[str, float]:
    values: dict[str, list[float]] = {m: [] for m in ["auc", "f1", "precision", "recall", "brier_score"]}
    y = forecast["ec_eea_2025_risk_label"].astype(int).to_numpy()
    score = forecast["risk_probability"].astype(float).to_numpy()
    n = len(forecast)
    for _ in range(BOOTSTRAP_B):
        idx = RNG.integers(0, n, n)
        yy = y[idx]
        ss = score[idx]
        pp = (ss >= threshold).astype(int)
        if len(np.unique(yy)) == 2:
            values["auc"].append(roc_auc_score(yy, ss))
        values["f1"].append(f1_score(yy, pp, zero_division=0))
        values["precision"].append(precision_score(yy, pp, zero_division=0))
        values["recall"].append(recall_score(yy, pp, zero_division=0))
        values["brier_score"].append(brier_score_loss(yy, ss))
    out = {}
    for metric, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[f"{metric}_ci_low"] = float(np.quantile(arr, 0.025))
            out[f"{metric}_ci_high"] = float(np.quantile(arr, 0.975))
        else:
            out[f"{metric}_ci_low"] = np.nan
            out[f"{metric}_ci_high"] = np.nan
    return out


def run_cutoff_experiments(panel: pd.DataFrame, include_ci: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_rows = []
    metric_rows = []
    threshold_rows = []
    methods = [
        ("LatestObservedRate", None),
        ("DistanceToTargetRule", None),
        ("CountryTrend", None),
        ("Ridge", "full_safe"),
        ("LightGBM", "recycling_lags_only"),
        ("LightGBM", "recycling_lags_plus_landfill"),
        ("LightGBM", "full_safe"),
    ]
    for cutoff in CUTOFF_YEARS:
        horizon = TARGET_YEAR - cutoff
        for method, feature_set in methods:
            if method in {"LatestObservedRate", "DistanceToTargetRule", "CountryTrend"}:
                fc = baseline_forecast(panel, cutoff, method)
                residuals = rolling_residuals(panel, horizon, method, cutoff_year=cutoff)
                fs = ""
            else:
                fc = model_forecast(panel, cutoff, method, feature_set or "full_safe")
                residuals = rolling_residuals(panel, horizon, method, feature_set, cutoff)
                fs = feature_set or "full_safe"
            sigma = sigma_from_residuals(residuals)
            enriched = enrich_forecast(fc, cutoff, method, sigma, len(residuals), fs)
            forecast_rows.append(enriched)
            metrics = calculate_metrics(enriched, FIXED_THRESHOLD)
            metrics.update(
                {
                    "cutoff_year": cutoff,
                    "horizon": horizon,
                    "method": method,
                    "feature_set": fs,
                    "sigma": sigma,
                    "residual_n": len(residuals),
                }
            )
            if include_ci:
                metrics.update(bootstrap_ci(enriched, FIXED_THRESHOLD))
            metric_rows.append(metrics)
            if cutoff in {2021, 2023} and method == "LightGBM" and fs == "full_safe":
                for threshold in np.arange(0.30, 0.81, 0.05):
                    row = calculate_metrics(enriched, float(threshold))
                    row.update({"cutoff_year": cutoff, "horizon": horizon, "method": method, "feature_set": fs})
                    threshold_rows.append(row)
    forecasts = pd.concat(forecast_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    return forecasts, metrics, thresholds


def run_lightgbm_fullsafe_cutoff_metrics(panel: pd.DataFrame, include_ci: bool = False) -> pd.DataFrame:
    """Lightweight robustness rerun used for the capped-landfill check."""
    rows = []
    for cutoff in CUTOFF_YEARS:
        horizon = TARGET_YEAR - cutoff
        fc = model_forecast(panel, cutoff, "LightGBM", "full_safe")
        residuals = rolling_residuals(panel, horizon, "LightGBM", "full_safe", cutoff)
        sigma = sigma_from_residuals(residuals)
        enriched = enrich_forecast(fc, cutoff, "LightGBM", sigma, len(residuals), "full_safe")
        metrics = calculate_metrics(enriched, FIXED_THRESHOLD)
        metrics.update(
            {
                "cutoff_year": cutoff,
                "horizon": horizon,
                "method": "LightGBM",
                "feature_set": "full_safe",
                "sigma": sigma,
                "residual_n": len(residuals),
            }
        )
        if include_ci:
            metrics.update(bootstrap_ci(enriched, FIXED_THRESHOLD))
        rows.append(metrics)
    return pd.DataFrame(rows)


def rolling_ablation(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in [1, 2]:
        for feature_set, features in FEATURE_SETS.items():
            for model_name in ["Ridge", "LightGBM"]:
                df = make_supervised_custom(panel, horizon, features)
                y_col = f"y_h{horizon}"
                actuals, preds = [], []
                for ty in range(2016, 2024):
                    test = df[df["target_year"] == ty].copy()
                    train = df[df["target_year"] < ty].copy()
                    if test.empty or len(train) < 80:
                        continue
                    pipe = bep.build_pipeline(bep.model_specs()[model_name], features)
                    pipe.fit(train[features], train[y_col])
                    pred = np.clip(pipe.predict(test[features]), 0, 100)
                    actuals.extend(test[y_col].tolist())
                    preds.extend(pred.tolist())
                if actuals:
                    y = np.asarray(actuals)
                    p = np.asarray(preds)
                    rows.append(
                        {
                            "horizon": horizon,
                            "model": model_name,
                            "feature_set": feature_set,
                            "n": len(y),
                            "rmse": math.sqrt(mean_squared_error(y, p)),
                            "mae": mean_absolute_error(y, p),
                            "r2": r2_score(y, p),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["horizon", "model", "mae"])


def landfill_outlier_analysis(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outliers = panel[panel["landfill_rate_proxy"] > 100][
        ["geo", "country", "year", "waste_generated_kg_pc", "landfill_kg_pc", "landfill_rate_proxy"]
    ].copy()
    capped = panel.copy()
    capped["landfill_rate_proxy"] = capped["landfill_rate_proxy"].clip(upper=100)
    capped = recompute_lags(capped)
    subset = run_lightgbm_fullsafe_cutoff_metrics(capped, include_ci=False)
    subset["robustness_variant"] = "landfill_proxy_capped_at_100"
    return outliers, subset


def scenario_definition_table(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel[panel["year"] == 2023].copy()
    return pd.DataFrame(
        [
            {
                "scenario": "S1 landfill transition",
                "changed_variables": "landfill_rate_proxy_lag1",
                "adjustment_rule": f"Countries above the 2023 EU median are set to the EU median ({base['landfill_rate_proxy_lag1'].median(skipna=True):.2f}).",
                "empirical_basis": "Within-sample 2023 cross-country median; no extrapolation above observed support.",
                "interpretation_limit": "Sensitivity only; landfill reduction is not balanced against destination treatment flows.",
            },
            {
                "scenario": "S2 circular capacity",
                "changed_variables": "circular_material_use_rate_lag1; resource_productivity_lag1",
                "adjustment_rule": f"Values below the 2023 75th percentile are raised to P75: CMU={base['circular_material_use_rate_lag1'].quantile(0.75):.2f}, resource productivity={base['resource_productivity_lag1'].quantile(0.75):.2f}.",
                "empirical_basis": "Observed EU-27 upper-quartile benchmark.",
                "interpretation_limit": "Sensitivity only; not a causal investment effect.",
            },
            {
                "scenario": "S3 incineration lock-in mitigation",
                "changed_variables": "incineration_rate_proxy_lag1",
                "adjustment_rule": f"Countries above the 2023 EU median are set to the 2023 EU 25th percentile ({base['incineration_rate_proxy_lag1'].quantile(0.25):.2f}).",
                "empirical_basis": "Observed EU-27 lower-quartile benchmark.",
                "interpretation_limit": "Sensitivity only; does not impose material-flow accounting constraints.",
            },
            {
                "scenario": "S4 combined transition package",
                "changed_variables": "S1 + S2 + S3 variables",
                "adjustment_rule": "Applies S1, S2, and S3 simultaneously within observed cross-country support.",
                "empirical_basis": "Combination of observed EU-27 median and quartile benchmarks.",
                "interpretation_limit": "Stress test of structural inertia, not a feasible policy bundle estimate.",
            },
        ]
    )


def plot_cutoff_auc(metrics: pd.DataFrame) -> None:
    plot = metrics[metrics["method"].isin(["LatestObservedRate", "CountryTrend", "Ridge", "LightGBM"])].copy()
    plot["label"] = plot.apply(lambda r: r["method"] if r["feature_set"] in {"", "full_safe"} else f"{r['method']} ({r['feature_set']})", axis=1)
    keep_labels = ["LatestObservedRate", "CountryTrend", "Ridge", "LightGBM", "LightGBM (recycling_lags_only)", "LightGBM (recycling_lags_plus_landfill)"]
    plot = plot[plot["label"].isin(keep_labels)]
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=220)
    for label, grp in plot.groupby("label"):
        grp = grp.sort_values("cutoff_year")
        ax.plot(grp["cutoff_year"], grp["auc"], marker="o", linewidth=1.8, label=label)
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(CUTOFF_YEARS)
    ax.set_xlabel("Information cutoff year")
    ax.set_ylabel("AUC against EC-EEA label")
    ax.set_title("Strict Information-Cutoff External Validation")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure8_information_cutoff_external_validation.png", bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, ablation: pd.DataFrame, landfill_outliers: pd.DataFrame, capped_metrics: pd.DataFrame) -> None:
    main = metrics[(metrics["method"] == "LightGBM") & (metrics["feature_set"] == "full_safe")].sort_values("cutoff_year")
    baseline = metrics[(metrics["method"] == "LatestObservedRate") & (metrics["feature_set"] == "")].sort_values("cutoff_year")
    comparison = main.merge(
        baseline[["cutoff_year", "auc", "f1", "brier_score"]],
        on="cutoff_year",
        suffixes=("_lightgbm", "_latest"),
    )
    lines = ["# Decisive Revision Experiments", ""]
    lines.append("## P0 Result: Strict Information-Cutoff Validation")
    for _, r in main.iterrows():
        lines.append(
            f"- Cutoff {int(r.cutoff_year)} (h={int(r.horizon)}): AUC={r.auc:.3f}, F1@0.50={r.f1:.3f}, "
            f"Precision={r.precision:.3f}, Recall={r.recall:.3f}, Brier={r.brier_score:.3f}."
        )
    lines.append("")
    lines.append("## Incremental Value over Latest-Rate Baseline")
    for _, r in comparison.iterrows():
        lines.append(
            f"- Cutoff {int(r.cutoff_year)}: ΔAUC={r.auc_lightgbm-r.auc_latest:+.3f}, "
            f"ΔF1={r.f1_lightgbm-r.f1_latest:+.3f}, ΔBrier={r.brier_score_lightgbm-r.brier_score_latest:+.3f}."
        )
    lines.append("")
    lines.append("## Simple Baseline Check")
    for _, r in baseline.iterrows():
        lines.append(
            f"- Latest observed rate baseline, cutoff {int(r.cutoff_year)}: AUC={r.auc:.3f}, F1@0.50={r.f1:.3f}, Brier={r.brier_score:.3f}."
        )
    lines.append("")
    lines.append("## Feature Ablation")
    best_ab = ablation.sort_values(["horizon", "mae"]).groupby("horizon").head(4)
    for _, r in best_ab.iterrows():
        lines.append(f"- Horizon {int(r.horizon)}, {r.model}, {r.feature_set}: MAE={r.mae:.3f}, RMSE={r.rmse:.3f}, R2={r.r2:.3f}.")
    lines.append("")
    lines.append("## Landfill Proxy >100")
    if landfill_outliers.empty:
        lines.append("- No landfill proxy value exceeds 100%.")
    else:
        for _, r in landfill_outliers.iterrows():
            lines.append(f"- {r.country} {int(r.year)}: landfill proxy={r.landfill_rate_proxy:.2f}% ({r.landfill_kg_pc:.1f}/{r.waste_generated_kg_pc:.1f} kg per capita).")
    lines.append("")
    lines.append("## Capped-Landfill Robustness")
    for _, r in capped_metrics.sort_values("cutoff_year").iterrows():
        lines.append(f"- Cutoff {int(r.cutoff_year)}, LightGBM full_safe after capping: AUC={r.auc:.3f}, F1={r.f1:.3f}, Brier={r.brier_score:.3f}.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "The decisive question is whether strict pre-2023 information cutoffs still identify EC-EEA risk labels and whether the full framework improves over a simple latest-rate target-gap rule. "
        "If the full model does not improve materially, the manuscript should be reframed as a transparent target-gap early-warning and validation tool rather than as a strong ML-driven discovery paper."
    )
    report_text = "\n".join(lines).replace("螖", "Delta ")
    (DOCS / "decisive_revision_experiments.md").write_text(report_text, encoding="utf-8")


def main() -> None:
    panel = load_panel()
    panel = recompute_lags(panel)
    forecasts, metrics, thresholds = run_cutoff_experiments(panel)
    forecasts.to_csv(TABLES / "table9_information_cutoff_country_forecasts.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLES / "table9_information_cutoff_external_validation.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(TABLES / "table9b_threshold_sensitivity_fixed_vs_tuned.csv", index=False, encoding="utf-8-sig")
    lightgbm = metrics[(metrics["method"] == "LightGBM") & (metrics["feature_set"] == "full_safe")].copy()
    latest = metrics[(metrics["method"] == "LatestObservedRate") & (metrics["feature_set"] == "")][
        ["cutoff_year", "auc", "f1", "precision", "recall", "brier_score"]
    ].copy()
    incremental = lightgbm.merge(latest, on="cutoff_year", suffixes=("_lightgbm", "_latest_rate"))
    for metric in ["auc", "f1", "precision", "recall", "brier_score"]:
        incremental[f"delta_{metric}"] = incremental[f"{metric}_lightgbm"] - incremental[f"{metric}_latest_rate"]
    incremental.to_csv(TABLES / "table9c_incremental_value_vs_latest_rate.csv", index=False, encoding="utf-8-sig")

    ablation = rolling_ablation(panel)
    ablation.to_csv(TABLES / "table10_feature_ablation_rolling_performance.csv", index=False, encoding="utf-8-sig")

    landfill_outliers, capped_metrics = landfill_outlier_analysis(panel)
    landfill_outliers.to_csv(TABLES / "table11_landfill_proxy_outliers.csv", index=False, encoding="utf-8-sig")
    capped_metrics.to_csv(TABLES / "table11b_landfill_capped_external_validation.csv", index=False, encoding="utf-8-sig")

    scenario_defs = scenario_definition_table(panel)
    scenario_defs.to_csv(TABLES / "table12_policy_scenario_definitions.csv", index=False, encoding="utf-8-sig")

    plot_cutoff_auc(metrics)
    write_report(metrics, ablation, landfill_outliers, capped_metrics)

    print(metrics.sort_values(["cutoff_year", "method", "feature_set"]).to_string(index=False))
    print(f"report={DOCS / 'decisive_revision_experiments.md'}")


if __name__ == "__main__":
    main()
