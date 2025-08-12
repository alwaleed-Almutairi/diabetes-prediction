# Diabetes Risk Modeling (Quick Start)

Predict diabetes risk from routine labs and OGTT readings using clean, reproducible notebooks.

## What’s inside
- `notebooks/diabetes.ipynb` — main workflow (EDA → FE → train/val/test → plots → save model → input form).
- `data/diabetes_sample_5000.csv` — sample dataset (~5k rows).
- `models/model.joblib` — saved pipeline (created after training).
- `requirements.txt` — minimal dependencies.

## Setup
```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
python -m venv .venv          # (macOS/Linux: python3 -m venv .venv)
# Windows:
.\.venv\Scriptsctivate
# macOS/Linux:
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
jupyter lab  # or: jupyter notebook
```

## Run
1. Open `notebooks/diabetes.ipynb`.
2. Run cells top-to-bottom:
   - Load `data/diabetes_sample_5000.csv`.
   - Apply **feature engineering** (`add_clinical_features`).
   - Train models (LogReg baseline; RF/XGBoost optional).
   - Validate (threshold via PR-curve), view metrics/plots.
   - **Save** model → `models/model.joblib`.

## Try your own readings
- At the end of the notebook, use **“USER INPUT FORM”** (ipywidgets).
- Enter values → click **Predict** → see **AT RISK / SAFE** + probability.
- If you see a warning about `add_clinical_features`, run that FE cell first.

## Optional
- **GPU (XGBoost):** set `tree_method="gpu_hist"` in the XGBoost cell.
- **Data Warehouse:** load with `pandas.read_sql(...)` via SQLAlchemy, then follow the same pipeline.

## Troubleshooting (very short)
- Widgets not showing → `pip install ipywidgets` and restart kernel.
- Different columns → update `feature_cols` and keep FE consistent at train & inference.
- Keep the **test set untouched** until final evaluation; prefer K-Fold CV on the train split.

> Educational use only — not a medical device or diagnosis.
