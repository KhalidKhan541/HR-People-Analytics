# HR People Analytics & Attrition Prediction

A production-grade Python toolkit for generating, analysing, and modelling employee attrition data. The pipeline produces synthetic HR datasets, engineers risk-aware features, runs descriptive and equity analyses, and trains a logistic regression classifier — all configurable through a single YAML file.

---

## Architecture

```
hr_analytics/
├── __init__.py            # Package metadata & version
├── run.py                 # CLI entry-point (argparse)
├── configs/
│   └── default.yaml       # Pipeline configuration
└── src/
    ├── __init__.py        # Public API re-exports
    ├── generate.py        # Synthetic data generation
    ├── preprocess.py      # Validation, missing values, encoding
    ├── features.py        # Feature engineering & risk scoring
    ├── analytics.py       # Descriptive analytics & equity tests
    ├── model.py           # Logistic regression training
    └── pipeline.py        # Pipeline orchestration & output
```

### Data Flow

```
default.yaml ──► generate ──► preprocess ──► features ──► analytics ──► model ──► outputs/
                                                                                   ├── hr_analytics_enriched.csv
                                                                                   ├── analytics_summary.json
│                                                                                   ├── equity_analysis.json
                                                                                   └── model_metrics.json
```

---

## Features

| Module | Capabilities |
|--------|-------------|
| **generate** | Configurable synthetic employee data; realistic attrition simulation based on satisfaction, performance, tenure, promotions |
| **preprocess** | Schema validation, median/mode imputation, IQR outlier detection, one-hot encoding |
| **features** | Rating trend & volatility, promotion gap flags, manager-change flags, salary-vs-department ratio, engagement score, composite risk score |
| **analytics** | Attrition breakdowns by department / level / gender / tenure, salary statistics, performance distributions |
| **equity** | Chi-square independence tests, four-fifths rule bias ratios |
| **model** | Logistic regression with class balancing, stratified train/test split, cross-validation, ROC-AUC, feature importances |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Default (10 000 employees, seed 42)
python -m hr_analytics.run full

# Custom employee count and seed
python -m hr_analytics.run full -n 5000 -s 123

# Analyse an existing CSV
python -m hr_analytics.run full -d data/hr_data.csv
```

### 3. Individual sub-commands

```bash
# Generate synthetic data only
python -m hr_analytics.run generate -n 8000 -o data

# Run analytics on existing data
python -m hr_analytics.run analyze data/hr_data.csv -o outputs

# Train the model only
python -m hr_analytics.run train data/hr_data.csv -o outputs
```

### 4. Verbose / debug logging

```bash
python -m hr_analytics.run full -v
```

---

## Configuration

All settings live in `hr_analytics/configs/default.yaml`.

```yaml
data:
  n_employees: 10000
  seed: 42
  output_path: data/hr_data.csv

departments: [Engineering, Sales, Marketing, HR, Finance, Operations, Product, Legal]
levels: [Junior, Mid, Senior, Lead, Manager, Director, VP]
genders: [Male, Female, Non-binary]
tenure_bands: ["0-1", "1-3", "3-5", "5-10", "10+"]

feature_engineering:
  recent_rating: true
  rating_trend: true
  avg_rating: true
  promotion_gap_threshold: 24    # months
  manager_change_threshold: 2
  risk_score_weights:
    promotion_gap: 0.25
    manager_change: 0.15
    salary_below_avg: 0.15
    satisfaction_below_avg: 0.15
    low_rating: 0.15
    declining_trend: 0.10
    rating_volatility: 0.05

model:
  test_size: 0.2
  cv_folds: 5
  random_state: 42

output:
  output_dir: outputs
  save_csv: true
```

Override individual settings via CLI flags or by providing a custom YAML.

---

## Output Files

| File | Description |
|------|-------------|
| `hr_analytics_enriched.csv` | Full dataset with all engineered features |
| `analytics_summary.json` | Descriptive statistics: attrition rates, salary stats, performance distributions |
| `equity_analysis.json` | Chi-square tests and bias ratios for demographic fairness checks |
| `model_metrics.json` | Logistic regression metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, feature importances, cross-validation scores |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24.0 | Numerical computation |
| pandas | >= 2.0.0 | Data manipulation |
| scikit-learn | >= 1.3.0 | Model training & evaluation |
| scipy | >= 1.10.0 | Statistical tests (chi-square) |
| pyyaml | >= 6.0 | YAML configuration parsing |

---

## Development

```bash
# Install in editable mode
pip install -e .

# Run with debug logging
python -m hr_analytics.run full -v

# Run tests (if pytest is available)
pytest tests/ -v
```

---

## License

MIT
