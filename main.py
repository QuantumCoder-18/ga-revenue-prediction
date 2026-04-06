import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ── Must be first Streamlit command ───────────────────────────────────────────
st.set_page_config(
    page_title="Revenue Analytics Dashboard",
    page_icon="💰",
    layout="wide"
)

# ── Force correct working directory ──────────────────────────────────────────
os.chdir(r"D:\Projects\ga_revenue_project")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_data.csv", low_memory=False)
    df["totals.transactionRevenue"] = pd.to_numeric(
        df["totals.transactionRevenue"], errors="coerce"
    ).fillna(0)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_resource
def load_model():
    if not os.path.exists("models/lgbm_model.pkl"):
        return None, None
    return joblib.load("models/lgbm_model.pkl")

df        = load_data()
model, feature_cols = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("💰 Revenue Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Go to page", [
    "Overview",
    "Traffic Sources",
    "Device Analytics",
    "Revenue Prediction",
    "Customer Segments"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Rows loaded:** {len(df):,}")
st.sidebar.markdown(f"**Columns:** {df.shape[1]}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_currency(v):
    # Indian number system: Crores and Lakhs
    if v >= 10_000_000: return f"₹{v/10_000_000:.2f} Cr"   # 1 Crore = 10 million
    if v >= 100_000:    return f"₹{v/100_000:.2f} L"        # 1 Lakh = 100 thousand
    if v >= 1_000:      return f"₹{v/1_000:.1f}K"
    return f"₹{v:,.2f}"

def fmt_number(v):
    if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
    if v >= 1_000:     return f"{v/1_000:.1f}K"
    return f"{int(v):,}"

# ── Detect correct column names from what actually exists ─────────────────────
# Your clean_data.csv may use different column name formats
# This finds the right one automatically

def find_col(df, *candidates):
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

revenue_col = find_col(df, "totals.transactionRevenue", "transactionRevenue", "revenue")
country_col = find_col(df, "geoNetwork.country", "country", "geoNetwork_country")
device_col  = find_col(df, "device.deviceCategory", "deviceCategory", "device_deviceCategory")
browser_col = find_col(df, "device.browser", "browser", "device_browser")
os_col      = find_col(df, "device.operatingSystem", "operatingSystem")
channel_col = find_col(df, "trafficSource.channelGrouping", "channelGrouping")
source_col  = find_col(df, "trafficSource.source", "source")
date_col    = find_col(df, "date")

# ── PAGE: Overview ────────────────────────────────────────────────────────────
def show_overview():
    st.title("📊 Overview")

    # KPI row
    total_rev   = df[revenue_col].sum()    if revenue_col else 0
    total_vis   = len(df)
    buyers      = (df[revenue_col] > 0).sum() if revenue_col else 0
    conv        = buyers / total_vis * 100

    bounce_col  = find_col(df, "totals.bounces", "bounces")
    bounce_rate = (df[bounce_col].sum() / total_vis * 100) if bounce_col else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue",   fmt_currency(total_rev))
    c2.metric("Total Visits",    fmt_number(total_vis))
    c3.metric("Conversion Rate", f"{conv:.2f}%")
    c4.metric("Bounce Rate",     f"{bounce_rate:.1f}%")

    st.markdown("---")

    # Revenue trend
    if date_col and revenue_col:
        st.subheader("Revenue trend over time")
        daily = (
            df.groupby(date_col)[revenue_col]
            .sum().reset_index()
            .rename(columns={revenue_col: "Revenue", date_col: "Date"})
        )
        fig = px.line(daily, x="Date", y="Revenue",
                      template="plotly_dark",
                      color_discrete_sequence=["#4f8bf9"])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Visits by device")
        if device_col:
            dev = df[device_col].value_counts().reset_index()
            dev.columns = ["Device", "Visits"]
            fig2 = px.pie(dev, names="Device", values="Visits",
                          template="plotly_dark",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(height=320)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Device column not found in dataset.")

    with right:
        st.subheader("Top 5 countries")
        if country_col:
            top5 = df[country_col].value_counts().head(5).reset_index()
            top5.columns = ["Country", "Visits"]
            fig3 = px.bar(top5, x="Visits", y="Country", orientation="h",
                          template="plotly_dark", color="Visits",
                          color_continuous_scale="Blues")
            fig3.update_layout(height=320)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Country column not found in dataset.")

    with st.expander("View raw data sample"):
        st.dataframe(df.head(50), use_container_width=True)

# ── PAGE: Traffic Sources ─────────────────────────────────────────────────────
def show_traffic():
    st.title("🔗 Traffic Sources")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visits by channel")
        if channel_col:
            ch = df[channel_col].value_counts().reset_index()
            ch.columns = ["Channel", "Visits"]
            fig = px.pie(ch, names="Channel", values="Visits",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Channel column not found.")

    with col2:
        st.subheader("Revenue by channel")
        if channel_col and revenue_col:
            rev_ch = (
                df.groupby(channel_col)[revenue_col]
                .sum().reset_index()
                .rename(columns={channel_col: "Channel", revenue_col: "Revenue"})
                .sort_values("Revenue", ascending=False)
            )
            fig2 = px.bar(rev_ch, x="Channel", y="Revenue",
                          template="plotly_dark", color="Revenue",
                          color_continuous_scale="Teal")
            st.plotly_chart(fig2, use_container_width=True)

    if source_col:
        st.subheader("Top 10 traffic sources")
        top_src = (
            df.groupby(source_col)
            .agg(Visits=(source_col, "count"),
                 Revenue=(revenue_col, "sum"))
            .reset_index()
            .rename(columns={source_col: "Source"})
            .sort_values("Visits", ascending=False).head(10)
        )
        fig3 = px.bar(top_src, x="Visits", y="Source", orientation="h",
                      template="plotly_dark", color="Revenue",
                      color_continuous_scale="Viridis")
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

# ── PAGE: Device Analytics ────────────────────────────────────────────────────
def show_devices():
    st.title("📱 Device Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Device category")
        if device_col:
            dev = df[device_col].value_counts().reset_index()
            dev.columns = ["Device", "Visits"]
            fig = px.bar(dev, x="Device", y="Visits",
                         template="plotly_dark", color="Visits",
                         color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top browsers")
        if browser_col:
            br = df[browser_col].value_counts().head(6).reset_index()
            br.columns = ["Browser", "Visits"]
            fig2 = px.pie(br, names="Browser", values="Visits",
                          template="plotly_dark",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig2, use_container_width=True)

    if os_col and revenue_col:
        st.subheader("Revenue by operating system")
        os_rev = (
            df.groupby(os_col)
            .agg(Visits=(os_col, "count"),
                 Revenue=(revenue_col, "sum"))
            .reset_index()
            .rename(columns={os_col: "OS"})
            .sort_values("Revenue", ascending=False).head(8)
        )
        fig3 = px.bar(os_rev, x="OS", y="Revenue",
                      template="plotly_dark", color="Revenue",
                      color_continuous_scale="Reds")
        st.plotly_chart(fig3, use_container_width=True)

# ── PAGE: Revenue Prediction ──────────────────────────────────────────────────
def show_prediction():
    st.title("🔮 Revenue Prediction")

    if model is None:
        st.error("Model not found. Run train_model.py first.")
        return

    with st.form("pred_form"):
        st.subheader("Enter visitor details")
        c1, c2, c3 = st.columns(3)
        pageviews = c1.slider("Page views",  1, 50,  5)
        hits      = c2.slider("Hits",        1, 80,  8)
        new_visit = c3.selectbox("New visitor?", [1, 0],
                                  format_func=lambda x: "Yes" if x==1 else "No")
        submitted = st.form_submit_button("Predict Revenue", type="primary")

    if submitted:
        sample = {f: float(df[f].median()) if f in df.columns else 0.0
                  for f in feature_cols}
        for k, v in [("totals.pageviews", pageviews),
                     ("totals.hits", hits),
                     ("totals.newVisits", new_visit)]:
            if k in sample:
                sample[k] = v

        X = pd.DataFrame([sample])[feature_cols].fillna(0)
        pred = float(np.expm1(model.predict(X)[0]))
        pred = max(0, pred)

        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.metric("Predicted Revenue", fmt_currency(pred))
        r2.metric("Page Views", pageviews)

        if pred > 200:
            st.success("High-value visitor — strong purchase intent!")
        elif pred > 20:
            st.info("Moderate revenue potential.")
        else:
            st.warning("Low revenue predicted — likely browsing only.")

    st.markdown("---")
    st.subheader("What the model looks at most")
    imp = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True).tail(10)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 template="plotly_dark", color="Importance",
                 color_continuous_scale="Blues")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE: Customer Segments ───────────────────────────────────────────────────
def show_segments():
    st.title("👥 Customer Segments")

    if not revenue_col:
        st.error("Revenue column not found.")
        return

    def tier(r):
        if r > 500: return "High value"
        if r > 50:  return "Medium value"
        if r > 0:   return "Low value"
        return "No purchase"

    df["segment"] = df[revenue_col].apply(tier)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Visitors by segment")
        sc = df["segment"].value_counts().reset_index()
        sc.columns = ["Segment", "Visitors"]
        fig = px.pie(sc, names="Segment", values="Visitors",
                     template="plotly_dark",
                     color_discrete_sequence=["#2ecc71","#3498db",
                                               "#f39c12","#95a5a6"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by segment")
        sr = (df.groupby("segment")[revenue_col]
              .sum().reset_index()
              .rename(columns={revenue_col: "Revenue"}))
        fig2 = px.bar(sr, x="segment", y="Revenue",
                      template="plotly_dark", color="Revenue",
                      color_continuous_scale="Greens")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Segment summary table")
    summary = (
        df.groupby("segment")
        .agg(Visitors=("segment","count"),
             Total_Revenue=(revenue_col,"sum"),
             Avg_Revenue=(revenue_col,"mean"))
        .reset_index()
        .sort_values("Total_Revenue", ascending=False)
    )
    summary["Total_Revenue"] = summary["Total_Revenue"].map("₹{:,.2f}".format)
    summary["Avg_Revenue"]   = summary["Avg_Revenue"].map("₹{:,.4f}".format)
    st.dataframe(summary, use_container_width=True)

# ── Router ────────────────────────────────────────────────────────────────────
if page == "Overview":          show_overview()
elif page == "Traffic Sources": show_traffic()
elif page == "Device Analytics":show_devices()
elif page == "Revenue Prediction": show_prediction()
elif page == "Customer Segments":  show_segments()