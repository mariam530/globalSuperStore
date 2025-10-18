# app.py
# ============================================================
# Streamlit: Full Analysis + Profit Modeling (Classification & Regression)
# + Manual Prediction (single row, no file upload)
# ============================================================
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)

# ---------------------- App Config ----------------------
st.set_page_config(page_title="Profit Analysis & Modeling", layout="wide")
st.title("📊 Profit Analysis & Modeling")

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", min_value=0, value=42, step=1)
    downsample = st.checkbox("Speed mode: downsample to 20k rows if larger", value=True)
    st.markdown("---")
    st.caption("Using local CSV: Global_Superstore2.csv (latin1). Dates will be auto-parsed when possible.")

# ---------------------- Helpers ----------------------
def _ohe_sparse():
    """OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
          .str.strip()
          .str.replace(r"[()\[\]\s]+", "_", regex=True)
          .str.replace(r"__+", "_", regex=True)
          .str.strip("_")
          .str.lower()
    )
    return df

def _auto_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_like_cols = [c for c in df.columns if any(k in c for k in ["date", "order", "ship", "invoice", "time"])]
    for c in date_like_cols:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        except Exception:
            pass
    return df

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Domain features used across the app."""
    X = X.copy()
    # Unit price & gross margin
    if {"sales", "quantity"}.issubset(X.columns):
        qty = pd.to_numeric(X["quantity"], errors="coerce")
        sales = pd.to_numeric(X["sales"], errors="coerce")
        X["unit_price"] = np.where(qty > 0, sales / qty, np.nan)
    if {"profit", "sales"}.issubset(X.columns):
        sales = pd.to_numeric(X["sales"], errors="coerce")
        profit = pd.to_numeric(X["profit"], errors="coerce")
        X["gross_margin_pct"] = np.where(sales != 0, profit / sales, np.nan)
    # Discount cleanup
    if "discount" in X.columns and "discount_pct" not in X.columns:
        X["discount_pct"] = pd.to_numeric(X["discount"], errors="coerce").clip(lower=0, upper=1)
    if "discount_pct" in X.columns:
        X["discount_level"] = pd.cut(
            pd.to_numeric(X["discount_pct"], errors="coerce"),
            bins=[-0.001, 0.0, 0.10, 0.20, 0.30, 1.0],
            labels=["0%", "0-10%", "10-20%", "20-30%", "30%+"]
        )
    # Profitable flag
    if "profit" in X.columns:
        X["is_profitable"] = (pd.to_numeric(X["profit"], errors="coerce") > 0).astype(int)
    # Ship delay
    if "order_date" in X.columns and "ship_date" in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X["order_date"]) and pd.api.types.is_datetime64_any_dtype(X["ship_date"]):
            X["ship_delay_days"] = (X["ship_date"] - X["order_date"]).dt.days
    # Date parts
    if "order_date" in X.columns and pd.api.types.is_datetime64_any_dtype(X["order_date"]):
        od = X["order_date"]
        X["order_year"] = od.dt.year
        X["order_month"] = od.dt.month
        X["order_quarter"] = od.dt.quarter
        X["order_yearmonth"] = od.dt.to_period("M").astype(str)
        X["order_weekday"] = od.dt.day_name()
        X["order_dow"] = od.dt.dayofweek
        X["order_weekend"] = X["order_dow"].isin([5, 6]).astype(int)
    # Unit price bands
    if "unit_price" in X.columns:
        q = X["unit_price"].quantile([0.25, 0.5, 0.75])
        try:
            X["unitprice_band"] = pd.cut(
                X["unit_price"],
                bins=[-np.inf, q.iloc[0], q.iloc[1], q.iloc[2], np.inf],
                labels=["Low", "Mid", "High", "Premium"]
            )
        except Exception:
            pass
    return X

def _build_quick_clf_pipeline(X_df, y, random_state=42):
    """Quick fallback pipeline if classification tab wasn't visited yet."""
    X_local = X_df.copy()
    for c in X_local.columns:
        if pd.api.types.is_datetime64_any_dtype(X_local[c]):
            X_local[c] = pd.to_datetime(X_local[c], errors="coerce").dt.to_period("M").astype(str)
    num_cols = X_local.select_dtypes(include=[np.number, "bool", "boolean"]).columns.tolist()
    cat_cols = X_local.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", MaxAbsScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _ohe_sparse())]), cat_cols),
        ],
        remainder="drop", sparse_threshold=1.0
    )
    clf = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))])
    clf.fit(X_local, y)
    return clf, num_cols, cat_cols

def _build_quick_reg_pipeline(X_df, y, random_state=42):
    """Quick fallback pipeline if regression tab wasn't visited yet."""
    X_local = X_df.copy()
    for c in X_local.columns:
        if pd.api.types.is_datetime64_any_dtype(X_local[c]):
            X_local[c] = pd.to_datetime(X_local[c], errors="coerce").dt.to_period("M").astype(str)
    num_cols = X_local.select_dtypes(include=[np.number, "bool", "boolean"]).columns.tolist()
    cat_cols = X_local.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", MaxAbsScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _ohe_sparse())]), cat_cols),
        ],
        remainder="drop", sparse_threshold=1.0
    )
    reg = Pipeline([("prep", preprocess), ("reg", RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1))])
    reg.fit(X_local, y)
    return reg, num_cols, cat_cols

# ---------------------- Load & Clean (direct file) ----------------------
df = pd.read_csv("Global_Superstore2.csv", encoding="latin1")
df = _normalize_columns(df)
df = _auto_parse_dates(df)

st.subheader("📄 Data Preview")
st.write(df.head())
st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

# Ensure profit column
if "profit" not in df.columns:
    cand = [c for c in df.columns if c.lower() == "profit"]
    if cand:
        df = df.rename(columns={cand[0]: "profit"})
    else:
        st.error("Column 'profit' not found. Please ensure your dataset has a 'profit' column.")
        st.stop()

# Optional downsample
if downsample and len(df) > 20_000:
    df = df.sample(20_000, random_state=random_state).reset_index(drop=True)
    st.warning("Dataset downsampled to 20,000 rows for speed.")

# Feature engineering
df = add_features(df)

# ============================================================
#                         TABS
# ============================================================
tabs = st.tabs([
    "Overview", "Analysis", "Visualizations",
    "Model: Classification", "Model: Regression",
    "Manual Prediction",  # NEW
    "Export"
])

# ---------------------- Overview Tab ----------------------
with tabs[0]:
    st.subheader("Dataset Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Numeric Summary**")
        num_desc = df.select_dtypes(include=[np.number, "boolean", "bool"]).describe().T
        st.write(num_desc if not num_desc.empty else "No numeric columns.")
    with c2:
        st.write("**Missing Values**")
        miss = df.isna().sum().sort_values(ascending=False)
        st.write(miss[miss > 0].to_frame("missing_count") if (miss > 0).any() else "No missing values detected.")

# ---------------------- Analysis Tab ----------------------
with tabs[1]:
    st.subheader("Common Analysis Questions")

    st.markdown("**1) Top/Bottom by profit**")
    top_n = st.slider("N (top/bottom)", 3, 50, 10, 1, key="n_top")
    groupable_cols = [c for c in df.columns if df[c].dtype == object or df[c].dtype.name == "category"]
    by_col = st.selectbox("Group by", options=groupable_cols + ["(no group)"])
    if by_col != "(no group)":
        g = df.groupby(by_col, dropna=False)["profit"].sum().sort_values(ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Top**")
            st.write(g.head(top_n).to_frame("sum_profit"))
        with c2:
            st.write("**Bottom**")
            st.write(g.tail(top_n).to_frame("sum_profit"))
    else:
        st.info("Choose a categorical column to aggregate by.")

    st.markdown("---")
    st.markdown("**2) Average metrics by segment/category**")
    group_cols = st.multiselect("Select grouping columns", options=df.columns.tolist(),
                                default=[c for c in ["segment", "category"] if c in df.columns])
    metrics = st.multiselect(
        "Select metrics",
        options=[c for c in ["profit", "sales", "quantity", "shipping_cost", "discount_pct", "unit_price"] if c in df.columns],
        default=[c for c in ["profit", "sales"] if c in df.columns]
    )
    if group_cols and metrics:
        agg_df = df.groupby(group_cols, dropna=False)[metrics].mean().reset_index()
        st.write(agg_df)

    st.markdown("---")
    st.markdown("**3) Correlation (numeric)**")
    numdf = df.select_dtypes(include=[np.number, "boolean", "bool"])
    if not numdf.empty:
        corr = numdf.corr(numeric_only=True)
        fig = px.imshow(corr, title="Correlation Heatmap", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns to show correlations.")

    st.markdown("---")
    st.markdown("**4) Time series (if order_date available)**")
    if "order_yearmonth" in df.columns:
        ts_metric = st.selectbox("Metric", options=[c for c in ["profit", "sales", "quantity"] if c in df.columns], index=0)
        ts = df.groupby("order_yearmonth")[ts_metric].sum().reset_index()
        fig = px.line(ts, x="order_yearmonth", y=ts_metric, title=f"{ts_metric} over time (monthly)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("order_date/order_yearmonth not available.")

    st.markdown("---")
    st.markdown("**5) Shipping delay vs profit (if available)**")
    if {"ship_delay_days", "profit"}.issubset(df.columns):
        color_col = "segment" if "segment" in df.columns else None
        fig = px.scatter(df, x="ship_delay_days", y="profit", color=color_col, title="Shipping delay vs Profit")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ship_delay_days not available to plot.")

# ---------------------- Classification Tab ----------------------
with tabs[3]:
    st.header("🔎 Classification: profit > 0")
    # Target
    y = (pd.to_numeric(df["profit"], errors="coerce") > 0).astype(int)
    X = df.copy()

    # Leakage guard
    LEAK_LIKE = {"profit", "is_profitable", "profit_bucket", "gross_margin_pct", "profitability_category"}
    leak_cols = [c for c in X.columns if (c.lower() in LEAK_LIKE) or ("profit" in c.lower())]
    X = X.drop(columns=leak_cols, errors="ignore")

    # Datetime → monthly bins (strings)
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = pd.to_datetime(X[c], errors="coerce").dt.to_period("M").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    num_cols = X_train.select_dtypes(include=[np.number, "bool", "boolean"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()

    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", MaxAbsScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _ohe_sparse())]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )

    model_name = st.selectbox("Model", ["LogisticRegression", "RandomForest", "GradientBoosting"])
    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    else:
        model = GradientBoostingClassifier(random_state=random_state)

    clf_pipe = Pipeline([("prep", preprocess), ("clf", model)])
    with st.spinner("Training classifier..."):
        clf_pipe.fit(X_train, y_train)

    # Default threshold=0.5
    y_pred = clf_pipe.predict(X_test)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{prec:.3f}")
    c2.metric("Recall", f"{rec:.3f}")
    c3.metric("F1", f"{f1:.3f}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(cm,
                          index=pd.Index(["True 0","True 1"], name="Actual"),
                          columns=pd.Index(["Pred 0","Pred 1"], name="Predicted")))

    # Threshold tuning (if available)
    thr = 0.5
    if hasattr(clf_pipe, "predict_proba"):
        st.subheader("🎚️ Threshold Tuning (maximize F1)")
        thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
        proba = clf_pipe.predict_proba(X_test)[:, 1]
        y_thr = (proba >= thr).astype(int)
        d1, d2, d3 = st.columns(3)
        d1.metric("Precision (thr)", f"{precision_score(y_test, y_thr, zero_division=0):.3f}")
        d2.metric("Recall (thr)", f"{recall_score(y_test, y_thr, zero_division=0):.3f}")
        d3.metric("F1 (thr)", f"{f1_score(y_test, y_thr, zero_division=0):.3f}")

        out_df = pd.DataFrame({"y_true": y_test, "proba": proba, "y_pred_thr": y_thr})
        st.download_button("⬇️ Download classification predictions (CSV)",
                           out_df.to_csv(index=False).encode(),
                           file_name="profit_classification_preds.csv")

# ---------------------- Regression Tab ----------------------
with tabs[4]:
    st.header("📈 Regression: predict profit (continuous)")
    y = pd.to_numeric(df["profit"], errors="coerce")
    X = df.drop(columns=["profit"], errors="ignore")

    # Datetime → monthly bins
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = pd.to_datetime(X[c], errors="coerce").dt.to_period("M").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    num_cols_r = X_train.select_dtypes(include=[np.number, "bool", "boolean"]).columns.tolist()
    cat_cols_r = X_train.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()

    preprocess_r = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", MaxAbsScaler())]), num_cols_r),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _ohe_sparse())]), cat_cols_r),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )

    model_name = st.selectbox("Model", ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"])
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(random_state=random_state)

    reg_pipe = Pipeline([("prep", preprocess_r), ("reg", model)])
    with st.spinner("Training regressor..."):
        reg_pipe.fit(X_train, y_train)

    y_pred = reg_pipe.predict(X_test)
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r1, r2, r3 = st.columns(3)
    r1.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
    r2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
    r3.metric("RMSE", f"{rmse:.3f}")

    out_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "residual": y_test - y_pred})
    st.download_button("⬇️ Download regression predictions (CSV)",
                       out_df.to_csv(index=False).encode(),
                       file_name="profit_regression_preds.csv")

# ---------------------- Manual Prediction Tab (NEW) ----------------------
with tabs[5]:
    st.header("📝 Manual Prediction (single row)")

    df_ref = df.copy()

    # Requested columns (normalized)
    requested_cols = [
        "segment", "city", "state", "country", "postal_code", "market", "region",
        "product_id", "category", "sub_category", "product_name",
        "sales", "quantity", "discount", "profit", "shipping_cost", "order_priority"
    ]
    available = [c for c in requested_cols if c in df_ref.columns]

    # Build suggestions for categoricals (top-50)
    cat_suggestions = {}
    for c in df_ref.select_dtypes(include=["object", "category"]).columns:
        cat_suggestions[c] = list(df_ref[c].dropna().astype(str).value_counts().head(50).index)

    # Numeric defaults (median)
    num_defaults = {}
    for c in df_ref.select_dtypes(include=[np.number, "bool", "boolean"]).columns:
        try:
            num_defaults[c] = float(pd.to_numeric(df_ref[c], errors="coerce").median())
        except Exception:
            num_defaults[c] = 0.0

    numeric_fields = [c for c in ["sales","quantity","discount","profit","shipping_cost"] if c in available]
    categorical_fields = [c for c in available if c not in numeric_fields]

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🔢 Numeric")
        num_inputs = {}
        for col in numeric_fields:
            default = num_defaults.get(col, 0.0)
            step = 0.01 if col == "discount" else 1.0
            fmt = "%.6f" if step == 0.01 else "%.4f"
            num_inputs[col] = st.number_input(col, value=float(default), step=step, format=fmt)

    with c2:
        st.subheader("🏷️ Categorical")
        cat_inputs = {}
        for col in categorical_fields:
            opts = cat_suggestions.get(col, [])
            if opts:
                choice = st.selectbox(col, options=["(enter manually)"] + opts)
                if choice == "(enter manually)":
                    cat_inputs[col] = st.text_input(f"{col} (manual)", value="")
                else:
                    cat_inputs[col] = choice
            else:
                cat_inputs[col] = st.text_input(col, value="")

    # Construct a single row
    row = {}
    for k, v in num_inputs.items(): row[k] = v
    for k, v in cat_inputs.items(): row[k] = v if (v is not None and v != "") else None
    one = pd.DataFrame([row])

    # Make sure all dataset columns exist for feature computation
    for m in set(df_ref.columns) - set(one.columns):
        one[m] = np.nan

    # Normalize + parse dates + features
    one = _normalize_columns(one)
    one = _auto_parse_dates(one)
    one = add_features(one)

    st.markdown("---")
    st.subheader("Choose task(s) and predict")

    do_class = st.checkbox("🔎 Classification (profit > 0)", value=True)
    do_reg   = st.checkbox("📈 Regression (predict profit value)", value=True)

    # ===== Classification =====
    if do_class:
        st.markdown("### 🔎 Classification Result")
        X_clf = one.copy()
        LEAK_LIKE = {"profit", "is_profitable", "profit_bucket", "gross_margin_pct", "profitability_category"}
        leak_cols = [c for c in X_clf.columns if (c.lower() in LEAK_LIKE) or ("profit" in c.lower())]
        X_clf = X_clf.drop(columns=leak_cols, errors="ignore")
        for c in X_clf.columns:
            if pd.api.types.is_datetime64_any_dtype(X_clf[c]):
                X_clf[c] = pd.to_datetime(X_clf[c], errors="coerce").dt.to_period("M").astype(str)

        if "clf_pipe" in locals() and "num_cols" in locals() and "cat_cols" in locals():
            pipe = clf_pipe; num_cols_ref, cat_cols_ref = num_cols, cat_cols
        else:
            y_tmp = (pd.to_numeric(df_ref["profit"], errors="coerce") > 0).astype(int)
            X_tmp = df_ref.copy()
            X_tmp = X_tmp.drop(columns=leak_cols, errors="ignore")
            pipe, num_cols_ref, cat_cols_ref = _build_quick_clf_pipeline(X_tmp, y_tmp, random_state=random_state)

        needed = list(set(num_cols_ref) | set(cat_cols_ref))
        for col in set(needed) - set(X_clf.columns): X_clf[col] = np.nan
        X_align = X_clf[needed]

        if hasattr(pipe, "predict_proba"):
            proba = float(pipe.predict_proba(X_align)[:, 1][0])
            pred  = int(proba >= 0.5)
            st.write({"pred_class": pred, "proba": proba})
        else:
            pred = int(pipe.predict(X_align)[0])
            st.write({"pred_class": pred})

    # ===== Regression =====
    if do_reg:
        st.markdown("### 📈 Regression Result")
        X_reg = one.drop(columns=["profit"], errors="ignore").copy()
        for c in X_reg.columns:
            if pd.api.types.is_datetime64_any_dtype(X_reg[c]):
                X_reg[c] = pd.to_datetime(X_reg[c], errors="coerce").dt.to_period("M").astype(str)

        if "reg_pipe" in locals() and "num_cols_r" in locals() and "cat_cols_r" in locals():
            rpipe = reg_pipe; num_cols_ref_r, cat_cols_ref_r = num_cols_r, cat_cols_r
        else:
            y_tmp = pd.to_numeric(df_ref["profit"], errors="coerce")
            X_tmp = df_ref.drop(columns=["profit"], errors="ignore").copy()
            rpipe, num_cols_ref_r, cat_cols_ref_r = _build_quick_reg_pipeline(X_tmp, y_tmp, random_state=random_state)

        needed_r = list(set(num_cols_ref_r) | set(cat_cols_ref_r))
        for col in set(needed_r) - set(X_reg.columns): X_reg[col] = np.nan
        X_align_r = X_reg[needed_r]

        y_pred_one = float(rpipe.predict(X_align_r)[0])
        st.write({"y_pred_profit": y_pred_one})

    st.info("✳️ Manual tab uses the same preprocessing and features. If you open the model tabs first, it will reuse those trained pipelines.")

# ---------------------- Export Tab ----------------------
with tabs[6]:
    st.header("📤 Export & Save")
    st.write("Download the enriched dataset (with engineered features):")
    st.download_button(
        "⬇️ Download enriched CSV",
        df.to_csv(index=False).encode(),
        file_name="enriched_dataset.csv"
    )

    st.markdown("**Quick Chart — Unit Price Band (if available)**")
    if "unitprice_band" in df.columns:
        vc = (df["unitprice_band"].value_counts(sort=False, dropna=False)
              .rename_axis("unitprice_band").reset_index(name="count"))
        vc = vc[vc["unitprice_band"].notna()]
        fig = px.bar(vc, x="unitprice_band", y="count", title="Counts by Unit Price Band")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("unitprice_band not available — it appears when `sales` and `quantity` exist.")




