# app.py
# ============================================================
# Streamlit: Full Analysis + Profit Modeling (Classification & Regression)
# + Inference on new CSV/XLSX files
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

def _load_new_table(file) -> pd.DataFrame:
    """Load CSV/Excel for inference (uploader)."""
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df_new = pd.read_excel(file, engine=None)
        return _auto_parse_dates(_normalize_columns(df_new))
    raw = file.read()
    for enc in ["utf-8","utf-8-sig","latin1","iso-8859-1","windows-1256"]:
        for sep in [None, ",",";","\t","|"]:
            try:
                bio = io.BytesIO(raw)
                df_new = pd.read_csv(bio, encoding=enc, sep=sep, engine="python")
                return _auto_parse_dates(_normalize_columns(df_new))
            except Exception:
                continue
    bio = io.BytesIO(raw)
    df_new = pd.read_csv(bio, engine="python")
    return _auto_parse_dates(_normalize_columns(df_new))

def _align_for_inference(df_new: pd.DataFrame, num_cols_ref, cat_cols_ref) -> pd.DataFrame:
    """Match new-data columns to training schema; convert datetimes to 'M' strings."""
    df_new = df_new.copy()
    for c in df_new.columns:
        if pd.api.types.is_datetime64_any_dtype(df_new[c]):
            df_new[c] = pd.to_datetime(df_new[c], errors="coerce").dt.to_period("M").astype(str)
    needed = set(num_cols_ref) | set(cat_cols_ref)
    for c in needed - set(df_new.columns):
        df_new[c] = np.nan
    return df_new

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

# ---------------------- Feature Engineering ----------------------
def add_features(X: pd.DataFrame) -> pd.DataFrame:
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
    if "discount" in X.columns:
        X["discount_pct"] = pd.to_numeric(X["discount"], errors="coerce").clip(lower=0, upper=1)
        X["discount_level"] = pd.cut(
            X["discount_pct"],
            bins=[-0.001, 0.0, 0.10, 0.20, 0.30, 1.0],
            labels=["0%", "0-10%", "10-20%", "20-30%", "30%+"]
        )
    # Profitable flag
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

df = add_features(df)

# ============================================================
#                         TABS
# ============================================================
tabs = st.tabs([
    "Overview", "Analysis", "Visualizations", "Model: Classification", "Model: Regression", "Export"
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

# ---------------------- Visualizations Tab ----------------------
with tabs[2]:
    st.subheader("Quick Visualizations")
    col_x = st.selectbox("X", options=df.columns, index=list(df.columns).index("unitprice_band") if "unitprice_band" in df.columns else 0)
    kind = st.selectbox("Chart type", ["Bar (counts)", "Histogram", "Box", "Scatter"])
    if kind == "Bar (counts)":
        vc = (df[col_x].value_counts(dropna=False)
              .rename_axis(col_x).reset_index(name="count"))
        vc = vc[vc[col_x].notna()]
        fig = px.bar(vc, x=col_x, y="count", title=f"Counts by {col_x}")
        st.plotly_chart(fig, use_container_width=True)
    elif kind == "Histogram":
        fig = px.histogram(df, x=col_x, title=f"Histogram: {col_x}")
        st.plotly_chart(fig, use_container_width=True)
    elif kind == "Box":
        fig = px.box(df, y=col_x, title=f"Box: {col_x}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        col_y = st.selectbox("Y", options=df.columns, index=list(df.columns).index("profit") if "profit" in df.columns else 1)
        color = st.selectbox("Color (optional)", options=[None] + df.columns.tolist())
        fig = px.scatter(df, x=col_x, y=col_y, color=color, title=f"Scatter: {col_x} vs {col_y}")
        st.plotly_chart(fig, use_container_width=True)

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

    # Threshold tuning
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

    # ===================== PREDICTION ON NEW DATA (Classification) =====================
    st.markdown("---")
    st.subheader("🧪 Predict on new data (Classification)")
    upl_c = st.file_uploader("Upload CSV/XLSX for classification prediction", type=["csv","xlsx","xls"], key="upl_clf")
    if upl_c is not None:
        new_df = _load_new_table(upl_c)
        new_df = _align_for_inference(new_df, num_cols, cat_cols)
        # Drop leakage columns if present
        new_df = new_df.drop(columns=[c for c in new_df.columns if (c.lower() in LEAK_LIKE) or ("profit" in c.lower())], errors="ignore")

        if hasattr(clf_pipe, "predict_proba"):
            thr_used = thr  # use slider value
            p = clf_pipe.predict_proba(new_df)[:, 1]
            pred = (p >= thr_used).astype(int)
            out = new_df.copy()
            out["proba"] = p
            out["pred_class"] = pred
        else:
            pred = clf_pipe.predict(new_df)
            out = new_df.copy()
            out["pred_class"] = pred

        st.write(out.head())
        st.download_button("⬇️ Download classification inference (CSV)",
                           out.to_csv(index=False).encode(),
                           file_name="classification_inference.csv")

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

    # ===================== PREDICTION ON NEW DATA (Regression) =====================
    st.markdown("---")
    st.subheader("🧪 Predict on new data (Regression)")
    upl_r = st.file_uploader("Upload CSV/XLSX for regression prediction", type=["csv","xlsx","xls"], key="upl_reg")
    if upl_r is not None:
        new_df_r = _load_new_table(upl_r)
        new_df_r = _align_for_inference(new_df_r, num_cols_r, cat_cols_r)

        preds = reg_pipe.predict(new_df_r)
        out_r = new_df_r.copy()
        out_r["y_pred_profit"] = preds

        if "profit" in out_r.columns:
            y_true_r = pd.to_numeric(out_r["profit"], errors="coerce")
            out_r["residual"] = y_true_r - preds

        st.write(out_r.head())
        st.download_button("⬇️ Download regression inference (CSV)",
                           out_r.to_csv(index=False).encode(),
                           file_name="regression_inference.csv")

# ---------------------- Export Tab ----------------------
with tabs[5]:
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



