

# app.py
# ============================================================
# Streamlit: Full Analysis + Profit Modeling (Classification & Regression)
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
    st.caption("CSV with a `profit` column is required. Dates will be auto-parsed when possible.")
    st.markdown("---")
    show_all = st.checkbox("🧾 Show everything on one page", value=False)
    show_diag = st.checkbox("🔧 Show diagnostics (columns & dtypes)", value=True)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# ---------------------- Load & Clean ----------------------
@st.cache_data(show_spinner=False)
def load_df_v2(file) -> pd.DataFrame:
    """Robust CSV loader for Streamlit UploadedFile:
    - Reads raw bytes once, retries with multiple encodings.
    - Tries delimiter inference first (sep=None, engine='python'), then common seps.
    - Normalizes column names & parses date-like columns.
    """
    # 1) read raw bytes once (avoid pointer issues with UploadedFile)
    raw = file.read() if hasattr(file, "read") else file
    if hasattr(file, "seek"):
        file.seek(0)

    encodings = ["Windows-1252", "utf-8", "latin1", "ISO-8859-1", "cp1256"]
    seps = [None, ",", ";", "\t", "|"]  # None → infer with engine='python'
    last_err = None
    df = None
    used_enc, used_sep = None, None

    # (A) delimiter inference
    for enc in encodings:
        try:
            buf = io.BytesIO(raw)
            df_try = pd.read_csv(buf, encoding=enc, sep=None, engine="python")
            if df_try.shape[1] > 1:
                df = df_try; used_enc, used_sep = enc, "inferred"
                break
        except Exception as e:
            last_err = e
            df = None

    # (B) explicit separators
    if df is None:
        for enc in encodings:
            for sep in [",", ";", "\t", "|"]:
                try:
                    buf = io.BytesIO(raw)
                    df_try = pd.read_csv(buf, encoding=enc, sep=sep)
                    if df_try.shape[1] > 1:
                        df = df_try; used_enc, used_sep = enc, sep
                        break
                except Exception as e:
                    last_err = e
                    df = None
            if df is not None:
                break

    if df is None:
        st.error("❌ Failed to read CSV properly. Try saving as UTF-8 (comma-separated).")
        if last_err:
            st.caption(f"Last error: {type(last_err).__name__}: {last_err}")
        st.stop()

    # normalize column names
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"[()\s]+", "_", regex=True)
          .str.lower()
    )

    # parse common date columns if present
    for c in df.columns:
        if any(k in c for k in ["date", "order", "ship"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="ignore")
            except Exception:
                pass

    st.caption(f"Decoded with: {used_enc} | Separator: {used_sep} | Shape: {df.shape[0]:,} x {df.shape[1]}")
    return df


df = load_df_v2(uploaded)

st.subheader("📄 Data Preview")
st.write(df.head())
st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

# Diagnostics panel
if show_diag:
    with st.expander("🔧 Diagnostics: columns & dtypes", expanded=True):
        st.write("**Columns:**", list(df.columns))
        dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.write(dtypes_df)
        st.dataframe(df.head(20), use_container_width=True)

# Profit column (assumed present as 'Profit' in raw → becomes 'profit' after normalization)
# No blocking checks; proceed directly.

# Optional downsample
if downsample and len(df) > 20_000:
    df = df.sample(20_000, random_state=random_state).reset_index(drop=True)
    st.warning("Dataset downsampled to 20,000 rows for speed.")

# ---------------------- Feature Engineering ----------------------
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Unit price & gross margin
    if {"sales", "quantity"}.issubset(X.columns):
        X["unit_price"] = np.where(pd.to_numeric(X["quantity"], errors="coerce") > 0,
                                   pd.to_numeric(X.get("sales"), errors="coerce") /
                                   pd.to_numeric(X.get("quantity"), errors="coerce"),
                                   np.nan)
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
    if "profit" in X.columns:
        X["is_profitable"] = (pd.to_numeric(X["profit"], errors="coerce") > 0).astype(int)

    # Ship delay
    if "order_date" in X.columns and "ship_date" in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X["order_date"]) and pd.api.types.is_datetime64_any_dtype(X["ship_date"]):
            X["ship_delay_days"] = (X["ship_date"] - X["order_date"]).dt.days

    # Date parts from order_date
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
        X["unitprice_band"] = pd.cut(
            X["unit_price"],
            bins=[-np.inf, q.iloc[0], q.iloc[1], q.iloc[2], np.inf],
            labels=["Low", "Mid", "High", "Premium"]
        )
    return X


df = add_features(df)

# ============================================================
#                         TABS
# ============================================================


# ---------------------- Feature Engineering ----------------------
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Unit price & gross margin
    if {"sales", "quantity"}.issubset(X.columns):
        X["unit_price"] = np.where(pd.to_numeric(X["quantity"], errors="coerce") > 0,
                                   pd.to_numeric(X.get("sales"), errors="coerce") /
                                   pd.to_numeric(X.get("quantity"), errors="coerce"),
                                   np.nan)
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

    # Date parts from order_date
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
        X["unitprice_band"] = pd.cut(
            X["unit_price"],
            bins=[-np.inf, q.iloc[0], q.iloc[1], q.iloc[2], np.inf],
            labels=["Low", "Mid", "High", "Premium"]
        )
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
        st.write(df.select_dtypes(include=np.number).describe().T)
    with c2:
        st.write("**Missing Values**")
        miss = df.isna().sum().sort_values(ascending=False)
        st.write(miss[miss > 0].to_frame("missing_count"))

# ---------------------- Analysis Tab ----------------------
with tabs[1]:
    st.subheader("Analysis Questions")
    u_tab, b_tab, m_tab = st.tabs(["Univariate", "Bivariate", "Multivariate"])

    # ---------- Helpers ----------
    num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_all = df.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()

    # ---------- Univariate ----------
    with u_tab:
        st.markdown("### 📌 Univariate")
        c1, c2 = st.columns(2)
        with c1:
            if num_cols_all:
                ux = st.selectbox("Numeric column (histogram)", options=num_cols_all, index=0, key="uni_num")
                st.plotly_chart(px.histogram(df, x=ux, nbins=50, title=f"Distribution of {ux}"), use_container_width=True)
            else:
                st.info("No numeric columns detected.")
        with c2:
            if cat_cols_all:
                uc = st.selectbox("Categorical column (counts)", options=cat_cols_all, index=0, key="uni_cat")
                vc = (df[uc].value_counts(dropna=False).rename_axis(uc).reset_index(name="count"))
                vc = vc[vc[uc].notna()]
                st.plotly_chart(px.bar(vc, x=uc, y="count", title=f"Counts by {uc}"), use_container_width=True)
            else:
                st.info("No categorical columns detected.")

        st.markdown("---")
        if "order_yearmonth" in df.columns:
            metric = st.selectbox("Time-series metric", options=[c for c in ["profit", "sales", "quantity"] if c in df.columns], index=0, key="uni_ts")
            ts = df.groupby("order_yearmonth")[metric].sum().reset_index()
            st.plotly_chart(px.line(ts, x="order_yearmonth", y=metric, title=f"{metric} over time (monthly)"), use_container_width=True)

    # ---------- Bivariate ----------
    with b_tab:
        st.markdown("### 🔗 Bivariate")
        # Profit by category
        if "profit" in df.columns and cat_cols_all:
            cat_sel = st.selectbox("Group by (bar of sum profit)", options=cat_cols_all, key="bi_cat")
            g = df.groupby(cat_sel, dropna=False)["profit"].sum().sort_values(ascending=False).reset_index()
            g = g[g[cat_sel].notna()]
            st.plotly_chart(px.bar(g, x=cat_sel, y="profit", title=f"Total Profit by {cat_sel}"), use_container_width=True)
        # Box plot: profit by category
        if "profit" in df.columns and cat_cols_all:
            cat_box = st.selectbox("Profit distribution by (box)", options=cat_cols_all, key="bi_box")
            st.plotly_chart(px.box(df, x=cat_box, y="profit", points=False, title=f"Profit distribution by {cat_box}"), use_container_width=True)
        # Scatter: unit_price vs profit (or sales)
        xnum = None
        for candidate in ["unit_price", "sales", "quantity"]:
            if candidate in df.columns:
                xnum = candidate; break
        if xnum and "profit" in df.columns:
            color_opt = st.selectbox("Color by (optional)", options=[None] + cat_cols_all, key="bi_color")
            st.plotly_chart(px.scatter(df, x=xnum, y="profit", color=color_opt, title=f"{xnum} vs Profit"), use_container_width=True)
        # Correlation heatmap
        if len(num_cols_all) >= 2:
            corr = df[num_cols_all].corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, title="Correlation (numeric)", aspect="auto"), use_container_width=True)

    # ---------- Multivariate ----------
    with m_tab:
        st.markdown("### 🧩 Multivariate")
        # Scatter with color & size
        nx = None
        for c in ["sales", "unit_price", "quantity"]:
            if c in df.columns:
                nx = c; break
        ny = "profit" if "profit" in df.columns else (num_cols_all[0] if num_cols_all else None)
        if nx and ny:
            size_col = "quantity" if ("quantity" in df.columns and ny != "quantity") else None
            color_by = st.selectbox("Color by", options=[None] + cat_cols_all, key="mv_color")
            st.plotly_chart(px.scatter(df, x=nx, y=ny, color=color_by, size=size_col, title=f"{nx} vs {ny} (colored)"), use_container_width=True)
        # Facet grid (category vs time)
        if "order_yearmonth" in df.columns and "profit" in df.columns and cat_cols_all:
            cat_f = st.selectbox("Facet by", options=cat_cols_all, key="mv_facet")
            ts2 = df.groupby(["order_yearmonth", cat_f])["profit"].sum().reset_index()
            st.plotly_chart(px.line(ts2, x="order_yearmonth", y="profit", facet_col=cat_f, facet_col_wrap=3,
                                    title=f"Monthly Profit by {cat_f}"), use_container_width=True)
        # Scatter matrix for top numeric subset
        if len(num_cols_all) >= 3:
            few = num_cols_all[:5]
            st.plotly_chart(px.scatter_matrix(df, dimensions=few, title="Scatter Matrix (top numeric)"), use_container_width=True)

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

    # Datetime → monthly bins
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = pd.to_datetime(X[c], errors="coerce")
            X[c] = X[c].dt.to_period("M").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocess
    def OHE():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

    num_cols = X_train.select_dtypes(include=[np.number, "bool", "boolean"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()

    preprocess = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", MaxAbsScaler())]), num_cols),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OHE())]), cat_cols)],
        remainder="drop", sparse_threshold=1.0
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

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(cm,
                          index=pd.Index(["True 0","True 1"], name="Actual"),
                          columns=pd.Index(["Pred 0","Pred 1"], name="Predicted")))

    # Threshold tuning
    if hasattr(clf_pipe, "predict_proba"):
        st.subheader("🎚️ Threshold Tuning (maximize F1)")
        thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
        proba = clf_pipe.predict_proba(X_test)[:, 1]
        y_thr = (proba >= thr).astype(int)
        prec_b = precision_score(y_test, y_thr, zero_division=0)
        rec_b  = recall_score(y_test, y_thr, zero_division=0)
        f1_b   = f1_score(y_test, y_thr, zero_division=0)
        d1, d2, d3 = st.columns(3)
        d1.metric("Precision (thr)", f"{prec_b:.3f}")
        d2.metric("Recall (thr)", f"{rec_b:.3f}")
        d3.metric("F1 (thr)", f"{f1_b:.3f}")

        out_df = pd.DataFrame({"y_true": y_test, "proba": proba, "y_pred_thr": y_thr})
        st.download_button("⬇️ Download classification predictions (CSV)", out_df.to_csv(index=False).encode(),
                           file_name="profit_classification_preds.csv")

# ---------------------- Regression Tab ----------------------
with tabs[4]:
    st.header("📈 Regression: predict profit (continuous)")
    y = pd.to_numeric(df["profit"], errors="coerce")
    X = df.drop(columns=["profit"], errors="ignore")

    # Datetime → monthly bins
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = pd.to_datetime(X[c], errors="coerce")
            X[c] = X[c].dt.to_period("M").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Preprocess
    def OHE2():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

    num_cols = X_train.select_dtypes(include=[np.number, "bool", "boolean"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()

    preprocess = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", MaxAbsScaler())]), num_cols),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OHE2())]), cat_cols)],
        remainder="drop", sparse_threshold=1.0
    )

    model_name = st.selectbox("Model", ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"])
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(random_state=random_state)

    reg_pipe = Pipeline([("prep", preprocess), ("reg", model)])
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
    st.download_button("⬇️ Download regression predictions (CSV)", out_df.to_csv(index=False).encode(),
                       file_name="profit_regression_preds.csv")

# ---------------------- Export Tab ----------------------
with tabs[5]:
    st.header("📤 Export & Save")
    st.write("Download the enriched dataset (with engineered features):")
    st.download_button("⬇️ Download enriched CSV", df.to_csv(index=False).encode(), file_name="enriched_dataset.csv")

    st.markdown("**Quick Chart — Unit Price Band (if available)**")
    if "unitprice_band" in df.columns:
        vc = (df["unitprice_band"].value_counts(sort=False, dropna=False)
              .rename_axis("unitprice_band").reset_index(name="count"))
        vc = vc[vc["unitprice_band"].notna()]
        fig = px.bar(vc, x="unitprice_band", y="count", title="Counts by Unit Price Band")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("unitprice_band not available — it appears when `sales` and `quantity` exist.")

# ---------------------- One‑Page (optional) ----------------------
if show_all:
    st.markdown("---")
    st.header("🧾 Full Page View")

    # 1) Correlation (if numeric columns exist)
    _num = df.select_dtypes(include=np.number)
    if not _num.empty:
        try:
            _corr = _num.corr(numeric_only=True)
            _fig = px.imshow(_corr, title="Correlation Heatmap", aspect="auto")
            st.plotly_chart(_fig, use_container_width=True)
        except Exception:
            pass

    # 2) Time series (if available)
    if "order_yearmonth" in df.columns:
        _metric = "profit" if "profit" in df.columns else (_num.columns[0] if not _num.empty else None)
        if _metric:
            _ts = df.groupby("order_yearmonth")[_metric].sum().reset_index()
            st.plotly_chart(px.line(_ts, x="order_yearmonth", y=_metric, title=f"{_metric} over time"), use_container_width=True)

    # 3) Profit histogram
    if "profit" in df.columns:
        st.plotly_chart(px.histogram(df, x="profit", title="Histogram: profit"), use_container_width=True)

    # 4) Counts for first categorical column (if exists)
    _cats = df.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()
    if _cats:
        _cx = _cats[0]
        _vc = (df[_cx].value_counts(dropna=False).rename_axis(_cx).reset_index(name="count"))
        _vc = _vc[_vc[_cx].notna()]
        st.plotly_chart(px.bar(_vc, x=_cx, y="count", title=f"Counts by {_cx}"), use_container_width=True)



