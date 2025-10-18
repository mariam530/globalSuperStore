# final.py
# ============================================================
# Streamlit: Profit Analysis & Modeling + Manual Prediction
# ============================================================
import io
import os
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

# ---------- Optional: enable Plotly trendline if statsmodels is installed ----------
try:
    import statsmodels.api as sm  # noqa: F401
    HAS_SM = True
except Exception:
    HAS_SM = False

# ---------------------- App Config ----------------------
st.set_page_config(page_title="Profit Analysis & Modeling", layout="wide")  # MUST be first Streamlit call
st.set_option("client.showErrorDetails", True)  # now safe after set_page_config
st.title("📊 Profit Analysis & Modeling")

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", min_value=0, value=42, step=1)
    downsample = st.checkbox("Speed mode: downsample to 20k rows if larger", value=True)
    st.markdown("---")
    st.caption("Using CSV: Global_Superstore2.csv (tries utf-8 → latin1 → cp1252). Dates auto-parsed when possible.")

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
    clf = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1))])
    clf.fit(X_local, y)
    return clf, num_cols, cat_cols

def _build_quick_reg_pipeline(X_df, y, random_state=42):
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

# ---------------------- Load & Clean (robust) ----------------------
def safe_read_csv(default_path: str) -> pd.DataFrame:
    # If file is missing, let the user upload one instead of crashing
    if not os.path.exists(default_path):
        st.warning(f"CSV not found at: `{default_path}`. Upload a CSV below or place the file next to final.py.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.stop()
        # Try multiple encodings for uploaded file
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                return pd.read_csv(uploaded, encoding=enc)
            except Exception:
                continue
        st.error("Could not read the uploaded CSV with utf-8 / latin1 / cp1252 encodings.")
        st.stop()

    # File exists locally — try multiple encodings
    last_err = None
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            return pd.read_csv(default_path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    st.error(f"Failed to read `{default_path}` with utf-8/latin1/cp1252. Last error: {last_err}")
    st.stop()

# Use the safer loader
df = safe_read_csv("Global_Superstore2.csv")
df = _normalize_columns(df)
df = _auto_parse_dates(df)

st.subheader("📄 Data Preview")
st.write(df.head())
st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

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

df = add_features(df)

# ============================================================
#                         TABS
# ============================================================
tabs = st.tabs([
    "Overview", "Analysis", "Visualizations",
    "Model: Classification", "Model: Regression",
    "Manual Prediction",
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

# ---------------------- Analysis Tab (Uni / Bi / Multi) ----------------------
with tabs[1]:
    st.subheader("Analysis")

    sub_tabs = st.tabs(["Univariate", "Bivariate", "Multivariate"])

    # =============== Univariate ===============
    with sub_tabs[0]:
        st.markdown("Explore single-variable distributions and summaries.")
        options_uni = df.columns.tolist()
        default_uni_idx = options_uni.index("profit") if "profit" in options_uni else 0
        col_uni = st.selectbox("Select a column", options=options_uni, index=default_uni_idx, key="uni_col")

        miss_cnt = int(df[col_uni].isna().sum())
        st.caption(f"Missing values in **{col_uni}**: {miss_cnt:,}")

        if pd.api.types.is_numeric_dtype(df[col_uni]):
            st.write("**Summary (numeric)**")
            st.write(df[col_uni].describe().to_frame().T)

            c1, c2 = st.columns(2)
            with c1:
                st.write("**Histogram**")
                fig = px.histogram(df, x=col_uni, nbins=50, title=f"Histogram: {col_uni}")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.write("**Box Plot**")
                fig = px.box(df, y=col_uni, points="suspectedoutliers", title=f"Box: {col_uni}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("**Category counts**")
            vc = (df[col_uni].value_counts(dropna=False)
                  .rename_axis(col_uni).reset_index(name="count"))
            vc = vc[vc[col_uni].notna()]
            fig = px.bar(vc.head(50), x=col_uni, y="count", title=f"Top categories: {col_uni}")
            st.plotly_chart(fig, use_container_width=True)

    # =============== Bivariate ===============
    with sub_tabs[1]:
        st.markdown("Explore relationships between two variables.")
        cols_all = df.columns.tolist()

        if "sales" in cols_all:
            x_default = "sales"
        elif "quantity" in cols_all:
            x_default = "quantity"
        else:
            x_default = next((c for c in cols_all if c != "profit"), cols_all[0])
        x_index = cols_all.index(x_default)
        col_x = st.selectbox("X", options=cols_all, index=x_index, key="bi_x")

        y_options = [c for c in cols_all if c != col_x]
        y_index = y_options.index("profit") if "profit" in y_options else 0
        col_y = st.selectbox("Y (target/metric)", options=y_options, index=y_index, key="bi_y")

        color_opt = st.selectbox("Color (optional)", options=[None] + cols_all, key="bi_color")

        if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y]):
            fig = px.scatter(
                df, x=col_x, y=col_y, color=color_opt,
                trendline=("ols" if (color_opt is None and HAS_SM) else None),
                title=f"Scatter: {col_x} vs {col_y}"
            )
            st.plotly_chart(fig, use_container_width=True)
            if color_opt is None and not HAS_SM:
                st.caption("ℹ️ Trendline disabled: install `statsmodels` to enable OLS line.")
        elif (not pd.api.types.is_numeric_dtype(df[col_x])) and pd.api.types.is_numeric_dtype(df[col_y]):
            agg_func = st.selectbox("Aggregation", ["sum", "mean", "median", "count"], index=1, key="bi_agg")
            g = getattr(df.groupby(col_x, dropna=False)[col_y], agg_func)().reset_index().sort_values(col_y, ascending=False)
            st.write(g.head(50))
            fig = px.bar(g.head(50), x=col_x, y=col_y,
                         color=color_opt if color_opt in g.columns else None,
                         title=f"{agg_func.upper()}({col_y}) by {col_x}")
            st.plotly_chart(fig, use_container_width=True)
        elif pd.api.types.is_numeric_dtype(df[col_x]) and (not pd.api.types.is_numeric_dtype(df[col_y])):
            agg_func = st.selectbox("Aggregation", ["sum", "mean", "median", "count"], index=1, key="bi_agg_swap")
            g = getattr(df.groupby(col_y, dropna=False)[col_x], agg_func)().reset_index().sort_values(col_x, ascending=False)
            st.write(g.head(50))
            fig = px.bar(g.head(50), x=col_y, y=col_x,
                         color=color_opt if color_opt in g.columns else None,
                         title=f"{agg_func.upper()}({col_x}) by {col_y}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Both selected columns are categorical — showing count table.")
            ct = df.groupby([col_x, col_y], dropna=False).size().reset_index(name="count")
            st.write(ct.head(100))
            fig = px.density_heatmap(ct, x=col_x, y=col_y, z="count", title=f"Counts: {col_x} × {col_y}")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Time series (if order_yearmonth available)**")
        if "order_yearmonth" in df.columns:
            ts_opts = [c for c in ["profit", "sales", "quantity"] if c in df.columns]
            ts_index = ts_opts.index("profit") if "profit" in ts_opts else 0
            ts_metric = st.selectbox("Metric", options=ts_opts, index=ts_index, key="bi_ts_metric")
            ts = df.groupby("order_yearmonth")[ts_metric].sum().reset_index()
            fig = px.line(ts, x="order_yearmonth", y=ts_metric, title=f"{ts_metric} over time (monthly)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("order_date/order_yearmonth not available.")

    # =============== Multivariate ===============
    with sub_tabs[2]:
        st.markdown("Explore multi-variable patterns.")

        st.markdown("**Correlation heatmap (numeric)**")
        numdf = df.select_dtypes(include=[np.number, "boolean", "bool"])
        if not numdf.empty:
            corr = numdf.corr(numeric_only=True)
            fig = px.imshow(corr, title="Correlation Heatmap", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns to show correlations.")

        st.markdown("---")

        if not numdf.empty:
            default_sm = [c for c in ["profit", "sales", "quantity", "unit_price"] if c in numdf.columns][:4]
            choose_cols = st.multiselect("Scatter Matrix: choose up to 6 numeric columns",
                                         options=numdf.columns.tolist(),
                                         default=default_sm,
                                         key="multi_scatter_cols")
            if choose_cols:
                fig = px.scatter_matrix(df, dimensions=choose_cols, title="Scatter Matrix")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.markdown("**Pivot heatmap (category × category → metric)**")
        cat_cols = df.select_dtypes(exclude=[np.number, "boolean", "bool"]).columns.tolist()
        if len(cat_cols) >= 2:
            r_cat = st.selectbox("Rows (categorical)", options=cat_cols, key="multi_r")
            c_cat = st.selectbox("Columns (categorical)", options=[c for c in cat_cols if c != r_cat], key="multi_c")
            metric_opts = [c for c in ["profit","sales","quantity","unit_price","shipping_cost","discount_pct"] if c in df.columns]
            metric_index = metric_opts.index("profit") if "profit" in metric_opts else 0
            metric = st.selectbox("Metric (numeric)", options=metric_opts, index=metric_index, key="multi_metric")
            agg_func = st.selectbox("Aggregation", ["sum", "mean", "median", "count"], index=0, key="multi_agg")

            tmp = df[[r_cat, c_cat, metric]].copy()
            if agg_func == "count":
                pvt = tmp.pivot_table(index=r_cat, columns=c_cat, values=metric, aggfunc="count", fill_value=0)
            else:
                pvt = tmp.pivot_table(index=r_cat, columns=c_cat, values=metric, aggfunc=agg_func, fill_value=0)
            fig = px.imshow(pvt, aspect="auto", title=f"{agg_func.upper()}({metric}) by {r_cat} × {c_cat}")
            st.plotly_chart(fig, use_container_width=True)
            st.write(pvt)
        else:
            st.info("Need at least two categorical columns for a pivot heatmap.")

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
        col_y = st.selectbox("Y", options=df.columns, index=(list(df.columns).index("profit") if "profit" in df.columns else (1 if len(df.columns)>1 else 0)))
        color = st.selectbox("Color (optional)", options=[None] + df.columns.tolist())
        fig = px.scatter(df, x=col_x, y=col_y, color=color, title=f"Scatter: {col_x} vs {col_y}")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- Classification Tab ----------------------
with tabs[3]:
    st.header("🔎 Classification: profit > 0")
    y = (pd.to_numeric(df["profit"], errors="coerce") > 0).astype(int)
    X = df.copy()

    LEAK_LIKE = {"profit", "is_profitable", "profit_bucket", "gross_margin_pct", "profitability_category"}
    leak_cols = [c for c in X.columns if (c.lower() in LEAK_LIKE) or ("profit" in c.lower())]
    X = X.drop(columns=leak_cols, errors="ignore")

    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = pd.to_datetime(X[c], errors="coerce").dt.to_period("M").astype(str)

    # ✅ FIX: Use stratify only when there are >= 2 classes
    stratify_y = y if pd.Series(y).nunique() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
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

    # RandomForest first (default)
    model_name = st.selectbox(
        "Model",
        ["RandomForest", "LogisticRegression", "GradientBoosting"]
    )
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = GradientBoostingClassifier(random_state=random_state)

    clf_pipe = Pipeline([("prep", preprocess), ("clf", model)])
    with st.spinner("Training classifier..."):
        clf_pipe.fit(X_train, y_train)

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

    # RandomForestRegressor first (default)
    model_name = st.selectbox(
        "Model",
        ["RandomForestRegressor", "LinearRegression", "GradientBoostingRegressor"]
    )
    if model_name == "RandomForestRegressor":
        model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    elif model_name == "LinearRegression":
        model = LinearRegression()
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

# ---------------------- Manual Prediction Tab (Predict profit on input) ----------------------
with tabs[5]:
    st.header("📝 Manual Prediction — Predict profit from a single row")

    df_ref = df.copy()

    # We EXCLUDE 'profit' from manual inputs (we're predicting it)
    requested_cols = [
        "segment", "city", "state", "country", "postal_code", "market", "region",
        "product_id", "category", "sub_category", "product_name",
        "sales", "quantity", "discount", "shipping_cost", "order_priority"
    ]
    available = [c for c in requested_cols if c in df_ref.columns]

    # Suggestions for categoricals (top-50)
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

    # Split manual inputs
    numeric_fields = [c for c in ["sales","quantity","discount","shipping_cost"] if c in available]
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

    # Build single-row dataframe
    row = {}
    for k, v in num_inputs.items(): row[k] = v
    for k, v in cat_inputs.items(): row[k] = v if (v is not None and v != "") else None
    one = pd.DataFrame([row])

    # Ensure all dataset columns exist for feature computation
    for m in set(df_ref.columns) - set(one.columns):
        one[m] = np.nan

    # Normalize + parse dates + engineered features
    one = _normalize_columns(one)
    one = _auto_parse_dates(one)
    one = add_features(one)

    st.markdown("---")
    st.subheader("📈 Predicted profit")

    # Prepare X for regression: drop 'profit' (target) if exists
    X_reg = one.drop(columns=["profit"], errors="ignore").copy()
    # Convert datetimes to monthly strings for the pipeline
    for c in X_reg.columns:
        if pd.api.types.is_datetime64_any_dtype(X_reg[c]):
            X_reg[c] = pd.to_datetime(X_reg[c], errors="coerce").dt.to_period("M").astype(str)

    # Use trained reg_pipe if available; otherwise build a quick one
    try:
        if "reg_pipe" in locals() and "num_cols_r" in locals() and "cat_cols_r" in locals():
            rpipe = reg_pipe
            needed_r = list(set(num_cols_r) | set(cat_cols_r))
        else:
            # Build a quick pipeline on the full dataset as a fallback
            y_tmp = pd.to_numeric(df_ref["profit"], errors="coerce")
            X_tmp = df_ref.drop(columns=["profit"], errors="ignore").copy()
            rpipe, num_cols_r_tmp, cat_cols_r_tmp = _build_quick_reg_pipeline(X_tmp, y_tmp, random_state=random_state)
            needed_r = list(set(num_cols_r_tmp) | set(cat_cols_r_tmp))

        # Align manual row to training columns
        for col in set(needed_r) - set(X_reg.columns):
            X_reg[col] = np.nan
        X_align_r = X_reg[needed_r]

        # Predict profit immediately
        y_pred_one = float(rpipe.predict(X_align_r)[0])
        st.success(f"✅ Predicted profit: **{y_pred_one:,.2f}**")

        # Optional: show the engineered row preview
        with st.expander("Show engineered row"):
            st.write(one)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption("Tip: Open the 'Model: Regression' tab once to train a pipeline, or keep this auto mode (it builds a quick pipeline as needed).")

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










