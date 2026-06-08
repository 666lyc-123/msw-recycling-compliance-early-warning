# Externally Validated Early Warning and Treatment-Pathway Diagnosis of EU Municipal Solid Waste Recycling Compliance Risk

This repository contains the processed data, analysis code, tables, and manuscript figures supporting the article:

**Externally Validated Early Warning and Treatment-Pathway Diagnosis of EU Municipal Solid Waste Recycling Compliance Risk**

The repository is intended for editorial and reviewer reproducibility checks. It excludes submission documents, cover letters, manuscript Word files, author-private notes, and intermediate drafting files.

## Repository structure

- `data/processed/`: processed EU-27 country-year panel and accompanying documentation.
- `figures/`: the four manuscript figures retained in the submitted article.
- `tables/core_manuscript_tables/`: core tables appearing in the manuscript.
- `tables/supporting_reproducibility_tables/`: robustness, validation, missingness, feature, prediction, and reference-verification tables.
- `src/`: scripts used to construct the empirical pipeline, external validation, risk diagnosis, and reference verification.

## Data sources

The processed panel and derived outputs are based on public European waste and circular-economy data products, including Eurostat municipal-waste and circular-economy indicators, EC-EEA early-warning evidence, and EEA country/profile materials used for external validation and policy-report comparison.

Because the original source datasets are public third-party data products, this repository provides the processed analytical panel and all derived tables/figures used for the manuscript. Original source-data rights and terms remain with the original providers.

## Main reproducibility files

- `data/processed/processed_panel.csv`: processed country-year panel used for modeling and diagnosis.
- `tables/core_manuscript_tables/table4_2025_target_gaps_and_risk_probabilities.csv`: 2025 target gaps and risk probabilities.
- `tables/core_manuscript_tables/table5_external_validation_ec_eea.csv`: EC-EEA external validation metrics.
- `tables/core_manuscript_tables/table9_information_cutoff_external_validation.csv`: information-cutoff validation results.
- `tables/core_manuscript_tables/table19_country_risk_diagnosis_profiles.csv`: country-level risk diagnosis profiles.
- `tables/core_manuscript_tables/table20_external_policy_report_validation.csv`: external policy-report validation.
- `figures/figure2_recycling_trajectories.png`: EU recycling trajectories.
- `figures/figure4_2025_risk_vs_external_label.png`: 2025 risk versus EC-EEA external labels.
- `figures/figure8_information_cutoff_external_validation.png`: information-cutoff validation.
- `figures/figure9_country_bottleneck_diagnosis.png`: treatment-pathway bottleneck diagnosis.

## Software

The analysis was run with Python. The main packages are listed in `requirements.txt`. Some outputs are supplied directly as CSV/PNG files so reviewers can inspect the results without rerunning every script.

## Suggested citation

If this repository is cited, use:

Data and code for: Externally Validated Early Warning and Treatment-Pathway Diagnosis of EU Municipal Solid Waste Recycling Compliance Risk.

## Data availability statement

The processed data, analysis code, validation outputs, manuscript tables, and figures supporting this study are available in this repository. Original source data are public Eurostat, EC-EEA, and EEA data products.
