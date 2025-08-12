
# Diabetes Risk Modeling — Ultra-Detailed Guide (Step-by-Step, Line-by-Line)

**Audience:** Anyone with basic Python and little/no ML background.  
**Goal:** Explain every step, why we do it, and how the algorithms work — so you can defend design choices and answer questions.

---

## 0) What this project does (plain English)

We build a **binary classifier** that predicts whether a person is likely to have diabetes (`diabetes_label ∈ {0,1}`) using common lab/clinical features (e.g., age, gender, BMI, fasting glucose, OGTT values, insulin, c‑peptide).  
We split the data into **Train / Validation / Test**, engineer extra clinically meaningful features, train multiple models, **tune a threshold** on the validation set, and finally **report metrics** on the test set.

You can use it with:  
- A **CSV** (e.g., `diabetes_sample_5000.csv`)  
- A **Data Warehouse** (BigQuery/Snowflake/SQL Server/etc.; sample connectors included below)  
- **Local CPU** or **External GPU** (XGBoost/LightGBM GPU tips included below)

---

## 1) Data: what the columns mean

- `age` — Age in years (float or int)  
- `gender` — `"Male"` or `"Female"` (string/categorical)  
- `bmi` — Body Mass Index (kg/m²), float  
- `glucose_fasting` — Fasting Plasma Glucose (mg/dL), float  
- `insulin_fasting` — Fasting insulin (µU/mL), float  
- `c_peptide_fasting` — Fasting C‑peptide (ng/mL), float  
- `ogtt_1h_glucose` — 1‑hour OGTT glucose (mg/dL), float  
- `ogtt_2h_glucose` — 2‑hour OGTT glucose (mg/dL), float  
- `diabetes_label` — Target (1 = diabetes, 0 = no diabetes)

> **Note**: The model assumes glucose units are **mg/dL** (standard US units). If your source is mmol/L, you’ll need to convert or adapt formulas (e.g., HOMA-IR).

---

## 2) Loading the CSV and quick data quality checks

**Why:** Before modeling, we confirm the file loads, no missing critical columns, and we summarize missing values/duplicates/outliers.

```python
import pandas as pd

csv_path = "diabetes_sample_5000.csv"  # <- your CSV file
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("Head:\n", df.head(3))

# Basic quality
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
```

**What happens:**  
- `pd.read_csv` reads file into `df`.  
- `.shape` shows rows/columns.  
- `.isna().sum()` counts missing values.  
- `.duplicated().sum()` finds exact duplicated rows (good to know!).

---

## 3) Selecting the target and features

**Why:** We need to tell the model which column to predict (target) and which columns are inputs (features).

```python
TARGET_COL = "diabetes_label"
feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int).copy()
```

- `TARGET_COL` is the column we want to predict.  
- `feature_cols` is **all other columns** (we can prune later).  
- `X` holds inputs; `y` holds labels.

---

## 4) Train / Validation / Test split (stratified)

**Why:** We want to evaluate generalization on **unseen** data (test), and tune decisions (like **classification threshold**) on **validation** without touching the test set.

```python
from sklearn.model_selection import train_test_split

# 80% Train+Val / 20% Test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 80% Train / 20% Val (of the 80% -> final 64/16/20 split)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, stratify=y_train_val, random_state=42
)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Prevalence — Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f} | Test: {y_test.mean():.3f}")
```

- `stratify=y` preserves the positive rate in each split.  
- `random_state=42` ensures reproducibility.  
- We **don’t** look at the test set until the very end.

---

## 5) Feature Engineering (row-wise; no leakage)

**Why:** Add medically meaningful predictors beyond raw values — improves model accuracy and interpretability.

Below is the **exact function** used in your notebook — with **line-by-line explanation**:

```python
import numpy as np
import pandas as pd

ENGINEERED_COLS = [
    "ogtt_delta_1h","ogtt_slope_0_1h","ogtt_delta_2h","ogtt_slope_0_2h","ogtt_slope_1_2h",
    "ogtt_auc_trap","homa_ir","fpg_prediabetes","fpg_diabetes","ogtt2h_prediabetes",
    "ogtt2h_diabetes","bmi_overweight","bmi_obese","insulin_cpep_ratio",
    "bmi_x_fpg","bmi_x_age","age_sq","bmi_sq","glucose_fasting_sq",
    "log1p_insulin_fasting","log1p_c_peptide_fasting","log1p_glucose_fasting",
]

def _to_float(s):
    # Convert a Series to float, turning bad strings into NaN safely
    return pd.to_numeric(s, errors="coerce").astype(float)

def add_clinical_features(X: pd.DataFrame, drop_existing: bool = True) -> pd.DataFrame:
    X = X.copy()  # don't modify caller's data in-place
    
    # If we already added features (you re-ran the cell), remove them to avoid duplicates
    if drop_existing:
        X.drop(columns=[c for c in ENGINEERED_COLS if c in X.columns], errors="ignore", inplace=True)

    # Short names for important columns (may be missing in some datasets)
    g0  = X.get("glucose_fasting")
    g1  = X.get("ogtt_1h_glucose")
    g2  = X.get("ogtt_2h_glucose")
    bmi = X.get("bmi")
    ins = X.get("insulin_fasting")
    cpe = X.get("c_peptide_fasting")
    age = X.get("age")

    eps = 1e-6  # small constant to avoid division by zero

    # --- OGTT dynamics (how glucose changes over the 2-hour test) ---
    if g0 is not None and g1 is not None:
        X["ogtt_delta_1h"]   = (_to_float(g1) - _to_float(g0))       # rise from fasting to 1h
        X["ogtt_slope_0_1h"] = (_to_float(g1) - _to_float(g0))       # same as delta (1-hour interval)
    if g0 is not None and g2 is not None:
        X["ogtt_delta_2h"]   = (_to_float(g2) - _to_float(g0))       # rise from fasting to 2h
        X["ogtt_slope_0_2h"] = (_to_float(g2) - _to_float(g0))       # same, 2-hour interval
    if g1 is not None and g2 is not None:
        X["ogtt_slope_1_2h"] = (_to_float(g2) - _to_float(g1))       # change between 1h and 2h
    
    # Trapezoid AUC (area under glucose curve from 0h->1h->2h): a simple aggregate of OGTT
    if (g0 is not None) and (g1 is not None) and (g2 is not None):
        g0f, g1f, g2f = _to_float(g0), _to_float(g1), _to_float(g2)
        X["ogtt_auc_trap"] = 0.5*(g0f + g1f) + 0.5*(g1f + g2f)

    # --- HOMA-IR: insulin resistance proxy (mg/dL version) ---
    if (ins is not None) and (g0 is not None):
        X["homa_ir"] = _to_float(ins) * _to_float(g0) / 405.0

    # --- Clinical flags at common guideline cut-offs ---
    if g0 is not None:
        g0f = _to_float(g0)
        X["fpg_prediabetes"] = ((g0f >= 100) & (g0f < 126)).astype(int)  # impaired fasting glucose
        X["fpg_diabetes"]    = (g0f >= 126).astype(int)                   # fasting diabetes threshold
    if g2 is not None:
        g2f = _to_float(g2)
        X["ogtt2h_prediabetes"] = ((g2f >= 140) & (g2f < 200)).astype(int) # IGT
        X["ogtt2h_diabetes"]    = (g2f >= 200).astype(int)                 # diabetes by 2h OGTT

    if bmi is not None:
        bmif = _to_float(bmi)
        X["bmi_overweight"] = ((bmif >= 25) & (bmif < 30)).astype(int)
        X["bmi_obese"]      = (bmif >= 30).astype(int)

    # --- Ratios and interactions (capture non-linear relations simply) ---
    if (ins is not None) and (cpe is not None):
        X["insulin_cpep_ratio"] = _to_float(ins) / (_to_float(cpe) + eps)  # secretion vs insulin proxy
    if (bmi is not None) and (g0 is not None):
        X["bmi_x_fpg"] = _to_float(bmi) * _to_float(g0)
    if (bmi is not None) and (age is not None):
        X["bmi_x_age"] = _to_float(bmi) * _to_float(age)

    # --- Mild nonlinearity (squared terms) ---
    if "age" in X.columns:
        X["age_sq"] = _to_float(X["age"]) ** 2
    if "bmi" in X.columns:
        X["bmi_sq"] = _to_float(X["bmi"]) ** 2
    if "glucose_fasting" in X.columns:
        X["glucose_fasting_sq"] = _to_float(X["glucose_fasting"]) ** 2

    # --- Log transforms for skewed variables (robust with clip to >=0) ---
    if "insulin_fasting" in X.columns:
        X["log1p_insulin_fasting"] = np.log1p(_to_float(X["insulin_fasting"]).clip(lower=0))
    if "c_peptide_fasting" in X.columns:
        X["log1p_c_peptide_fasting"] = np.log1p(_to_float(X["c_peptide_fasting"]).clip(lower=0))
    if "glucose_fasting" in X.columns:
        X["log1p_glucose_fasting"] = np.log1p(_to_float(X["glucose_fasting"]).clip(lower=0))

    return X
```

**How to use it:** place after the split, then run:

```python
X_train = add_clinical_features(X_train)
X_val   = add_clinical_features(X_val)
X_test  = add_clinical_features(X_test)
```

This is **row-wise** (uses only the same person’s row), so it **cannot leak** information from validation/test into train.

---

## 6) Preprocessing (turn raw columns into model-ready numbers)

**Why:** Models need clean numeric arrays.  
We handle: missing values, scaling numeric columns, and encoding categorical columns.

### 6.1 Column selection
```python
from sklearn.compose import make_column_selector

num_selector = make_column_selector(pattern=None, dtype_include=["number"])
cat_selector = make_column_selector(pattern=None, dtype_include=["object", "category"])
```

- **Numeric columns**: get standardized (mean=0, std=1) for linear models.  
- **Categorical columns**: get one‑hot encoded into 0/1 vectors.

### 6.2 Build preprocessors

- **Linear models** (Logistic Regression / Neural nets) benefit from scaled inputs.
- **Tree models** (RF/XGBoost) do **not** require scaling; they just need missing values handled and categories encoded.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

pre_linear = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler())
    ]), num_selector),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse=False))
    ]), cat_selector),
])

pre_tree = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_selector),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore"))
    ]), cat_selector),
])
```

- `SimpleImputer` handles missing values.  
- `StandardScaler` scales numeric features for linear models.  
- `OneHotEncoder` turns categories into binary columns.

---

## 7) Models and how they work (with code)

### 7.1 Logistic Regression (linear model)

**Intuition:** Finds a **straight line (or hyperplane)** in feature space that separates class 0 vs 1.  
**Pros:** fast, interpretable. **Cons:** struggles with complex nonlinear boundaries.

```python
from sklearn.linear_model import LogisticRegression

lr_pipeline = Pipeline([
    ("pre", pre_linear),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"))
])

lr_pipeline.fit(X_train, y_train)    # learn weights
proba_val = lr_pipeline.predict_proba(X_val)[:,1]  # predict probabilities
```

### 7.2 Random Forest (ensemble of decision trees)

**Intuition:** Grows many decision trees on random subsets of rows/features, and averages their votes.  
**Pros:** handles nonlinearity and interactions automatically. **Cons:** larger models, less interpretable than linear.

```python
from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline([
    ("pre", pre_tree),
    ("clf", RandomForestClassifier(
        n_estimators=800, max_depth=10, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)
proba_val = rf_pipeline.predict_proba(X_val)[:,1]
```

### 7.3 XGBoost (gradient-boosted trees)

**Intuition:** Builds trees **sequentially**, each new tree corrects errors of the previous ones. Often top accuracy.  
**Pros:** high accuracy, handles nonlinearity. **Cons:** many hyperparameters; can overfit if not tuned.

```python
# pip install xgboost --quiet   # if needed

import xgboost as xgb

xgb_pipeline = Pipeline([
    ("pre", pre_tree),
    ("clf", xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, eval_metric="auc",
        tree_method="hist"  # set to "gpu_hist" if using GPU
    ))
])

xgb_pipeline.fit(X_train, y_train)
proba_val = xgb_pipeline.predict_proba(X_val)[:,1]
```

> **GPU note:** On a CUDA GPU box, switch `tree_method="gpu_hist"` (and optionally `predictor="gpu_predictor"`).

---

## 8) Threshold tuning on the Validation set (why and how)

**Why:** Models output probabilities. You must pick a cutoff **threshold** to convert to 0/1.  
Use validation to choose a threshold that balances **Precision** and **Recall** for your use case.

```python
from sklearn.metrics import precision_recall_curve

def pr_opt_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    if len(t) <= 1:
        return 0.5  # fallback
    J = p + r - 1           # Youden-like index on PR
    return float(t[np.argmax(J[:-1])])
```

- **Precision**: of predicted positives, how many are truly positive?  
- **Recall**: of actual positives, how many did we catch?  
- The **best threshold** is not always 0.5 — we pick it by validation!

Apply it:
```python
thr = pr_opt_threshold(y_val, proba_val)
pred_test = (xgb_pipeline.predict_proba(X_test)[:,1] >= thr).astype(int)
```

---

## 9) Metrics and what they mean

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score
)

proba_test = xgb_pipeline.predict_proba(X_test)[:,1]
pred_test  = (proba_test >= thr).astype(int)

auroc = roc_auc_score(y_test, proba_test)         # ranking quality across thresholds
ap    = average_precision_score(y_test, proba_test) # average precision (area under PR)
brier = brier_score_loss(y_test, proba_test)      # calibration (lower is better)
prec  = precision_score(y_test, pred_test)
rec   = recall_score(y_test, pred_test)
prev  = float(y_test.mean())
lift  = (prec / prev) if prev > 0 else float("nan")

print(f"AUC={auroc:.4f}  AP={ap:.4f}  Prec={prec:.3f}  Rec={rec:.3f}  Brier={brier:.4f}  Prev={prev:.3f}  Thr={thr:.3f}  Lift={lift:.2f}x")
```

- **AUC (ROC)**: probability a random positive scores higher than a random negative.  
- **AP (AUPRC)**: area under Precision-Recall; useful for imbalanced data.  
- **Brier**: measures how close probabilities are to true outcomes (calibration).  
- **Lift**: how much better your precision is than the base rate.

---

## 10) Plots (what they show)

- **Target distribution**: check class balance (imbalance affects metrics/threshold).  
- **Correlation heatmap (numeric)**: linear relationships between features/target.  
- **Age/BMI hist by label**: visualize separation.  
- **PR and ROC curves**: show performance across thresholds.  
- **Feature importance**: which features the model uses most.  
- **Learning curve**: do we benefit from more data?  
- **Validation curve**: how a hyperparameter affects CV score.

> These plots help you explain *why* the model works and whether it’s overfitting.

---

## 11) K-fold Cross-Validation (stronger validation)

**Why:** Instead of one split, rotate multiple folds to estimate generalization more robustly.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print("5-fold CV ROC AUC (mean±std):", scores.mean(), "±", scores.std())
```

> Use CV on the training set to compare models. Keep the **test set** unseen until final reporting.

---

## 12) Scaling to large data (e.g., 2 million rows)

- Prefer **tree models** (RF/XGB) on CPUs with **n_jobs=-1** and moderate depth.  
- For XGBoost, consider **GPU** (`tree_method="gpu_hist"`) for big data if you have an NVIDIA GPU with CUDA.  
- Use chunked reading (`pd.read_csv(..., chunksize=...)`) or consider **Dask/Polars** for bigger-than-memory.  
- Avoid Python loops — keep operations vectorized in pandas/NumPy.

---

## 13) Connecting to a Data Warehouse (examples)

> Keep credentials secure (env vars / secrets manager). The code below is commented so it won’t run accidentally.

**BigQuery (pandas-gbq):**
```python
# pip install pandas-gbq google-cloud-bigquery --quiet
# from google.oauth2 import service_account
# import pandas as pd
#
# credentials = service_account.Credentials.from_service_account_file("gcp-key.json")
# sql = "SELECT age, gender, bmi, glucose_fasting, insulin_fasting, c_peptide_fasting, ogtt_1h_glucose, ogtt_2h_glucose, diabetes_label FROM project.dataset.table"
# df = pd.read_gbq(sql, project_id="your-project", credentials=credentials)
```

**Snowflake (SQLAlchemy):**
```python
# pip install snowflake-sqlalchemy --quiet
# from sqlalchemy import create_engine
# import pandas as pd
#
# engine = create_engine("snowflake://<user>:<password>@<account>/<db>/<schema>?warehouse=<wh>&role=<role>")
# df = pd.read_sql("SELECT ... FROM ...", engine)
```

**SQL Server (ODBC / pymssql):**
```python
# pip install sqlalchemy pymssql --quiet
# from sqlalchemy import create_engine
# import pandas as pd
#
# engine = create_engine("mssql+pymssql://user:pwd@host:1433/DBNAME")
# df = pd.read_sql("SELECT ... FROM dbo.your_table", engine)
```

**S3 Parquet (pyarrow):**
```python
# pip install s3fs pyarrow --quiet
# import pandas as pd
#
# df = pd.read_parquet("s3://bucket/path/data.parquet")
```

Once `df` is loaded from your warehouse, the rest of the pipeline is identical.

---

## 14) Using external GPUs

- **XGBoost:** set `tree_method="gpu_hist"` (fastest) and optionally `predictor="gpu_predictor"`.  
- **LightGBM:** use `device="gpu"` (requires GPU-enabled build).  
- **Neural nets (Keras/PyTorch):** will auto-detect GPU if installed with CUDA.

**XGBoost GPU example:**
```python
xgb.XGBClassifier(
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    n_estimators=800, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="auc",
    random_state=42
)
```

---

## 15) Common pitfalls & fixes

- **Pandas `.clip(min=0)` error** → Use `.clip(lower=0)` (pandas uses lower/upper).  
- **OneHotEncoder + pandas output warning** → Set `sparse=False` for DataFrame output, or rely on default NumPy array output.  
- **BrokenProcessPool in Joblib** → Usually environment conflict; set `n_jobs=1` or restart kernel/venv.  
- **Perfect validation AUC = 1.0** → Check for **leakage** or **duplicates across splits**.  
- **Mismatched units (mmol/L vs mg/dL)** → Convert units or adapt formulas (e.g., HOMA-IR).

---

## 16) How to present results (talk track)

1) **Problem & Data** — what’s being predicted, what features, how much data.  
2) **Quality checks** — missing values, duplicates, outliers.  
3) **Split strategy** — Train/Val/Test, stratified.  
4) **Feature engineering** — medical rationale (HOMA-IR, OGTT deltas).  
5) **Models compared** — LR (interpretable), RF (nonlinear), XGB (state-of-the-art).  
6) **Threshold selection** — tuned on validation to balance precision/recall.  
7) **Final metrics on Test** — AUC/AP/Precision/Recall/Brier/Lift.  
8) **Plots** — PR/ROC, feature importances, learning/validation curves.  
9) **Scalability & Ops** — GPU option, warehouse integration, versioning, reproducibility.

---

## 17) Repro: quick run order

1) Load CSV → build `df`  
2) Choose `TARGET_COL` / `feature_cols`  
3) Train/Val/Test split (stratified)  
4) **Feature Engineering** (this guide’s function)  
5) Define preprocessors (`pre_linear`, `pre_tree`)  
6) Train models (LR, RF, XGB)  
7) Tune threshold on **Validation**  
8) Evaluate on **Test**  
9) Generate plots  
10) (Optional) K-fold CV, calibration, GPU variants, warehouse connectors

---

*You now have both the “how” and the “why” for each step. If you want, I can also generate a slide deck version (PowerPoint) with the same content and charts.*
