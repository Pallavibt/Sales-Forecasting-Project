import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting & Demand Intelligence System")

# --------------------------
# Load Dataset
# --------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    return df

df = load_data()

# --------------------------
# Sidebar
# --------------------------

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# ==========================================================
# PAGE 1
# ==========================================================

if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

    total_sales = df["Sales"].sum()

    total_orders = df["Order ID"].nunique()

    total_products = df["Product ID"].nunique()

    avg_sales = df["Sales"].mean()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Total Sales",f"${total_sales:,.2f}")
    c2.metric("Orders",total_orders)
    c3.metric("Products",total_products)
    c4.metric("Average Sale",f"${avg_sales:.2f}")

    st.divider()

    # -----------------------------
    # Total Sales by Year
    # -----------------------------

    yearly = (
        df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        yearly,
        x="Year",
        y="Sales",
        title="Total Sales by Year"
    )

    st.plotly_chart(fig,use_container_width=True)

    # -----------------------------
    # Monthly Trend
    # -----------------------------

    monthly = (
        df.groupby(pd.Grouper(key="Order Date",freq="M"))["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # -----------------------------
    # Interactive Filters
    # -----------------------------

    col1,col2 = st.columns(2)

    region = col1.selectbox(
        "Region",
        ["All"]+sorted(df["Region"].unique().tolist())
    )

    category = col2.selectbox(
        "Category",
        ["All"]+sorted(df["Category"].unique().tolist())
    )

    filtered = df.copy()

    if region!="All":
        filtered = filtered[
            filtered["Region"]==region
        ]

    if category!="All":
        filtered = filtered[
            filtered["Category"]==category
        ]

    sales = (
        filtered
        .groupby(["Region","Category"])["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        sales,
        x="Region",
        y="Sales",
        color="Category",
        barmode="group",
        title="Sales by Region & Category"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Filtered Dataset")

    st.dataframe(filtered.head(20),use_container_width=True)

# ==========================================================
# PAGE 2
# ==========================================================

elif page == "Forecast Explorer":

    st.header("Forecast Explorer")

    option = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    if option == "Category":

        selected = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )

        data = df[df["Category"] == selected]

    else:

        selected = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )

        data = df[df["Region"] == selected]

    # -------------------------
    # Monthly Sales
    # -------------------------

    monthly = (
        data
        .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
        .sum()
        .reset_index()
    )

    monthly.columns = ["ds", "y"]

    train = monthly.iloc[:-3]
    test = monthly.iloc[-3:]

    # -------------------------
    # Prophet Model
    # -------------------------

    model = Prophet()

    model.fit(train)

    future = model.make_future_dataframe(
        periods=horizon,
        freq="M"
    )

    forecast = model.predict(future)

    # -------------------------
    # Test Prediction
    # -------------------------

    test_forecast = model.predict(test[["ds"]])

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    mae = mean_absolute_error(
        test["y"],
        test_forecast["yhat"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["y"],
            test_forecast["yhat"]
        )
    )

    # -------------------------
    # Forecast Chart
    # -------------------------

    fig = px.line(
        forecast,
        x="ds",
        y="yhat",
        title=f"{selected} Sales Forecast"
    )

    fig.add_scatter(
        x=monthly["ds"],
        y=monthly["y"],
        mode="lines+markers",
        name="Actual Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Future Forecast")

    st.dataframe(
        forecast[["ds", "yhat"]].tail(horizon),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        f"{mae:.2f}"
    )

    col2.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    st.subheader("Forecast Components")

    prophet_fig = model.plot_components(forecast)

    st.pyplot(prophet_fig)

    st.success(
        "Forecast generated successfully using Prophet."
    )

# ==========================================================
# PAGE 3
# ==========================================================

elif page == "Anomaly Report":

    st.header("📌 Sales Anomaly Detection")

    weekly = (
        df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
        .sum()
        .reset_index()
    )

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly["Anomaly"] = iso.fit_predict(
        weekly[["Sales"]]
    )

    anomaly = weekly[
        weekly["Anomaly"] == -1
    ]

    fig = px.line(
        weekly,
        x="Order Date",
        y="Sales",
        title="Weekly Sales with Detected Anomalies"
    )

    fig.add_scatter(
        x=anomaly["Order Date"],
        y=anomaly["Sales"],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="Anomaly"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly,
        use_container_width=True
    )

    st.subheader("Business Interpretation")

    st.markdown("""
- High spikes during **November** may indicate Black Friday or festive sales.
- High sales in **December** usually represent holiday shopping.
- Sharp drops may indicate stock shortages, logistics issues, or unexpected disruptions.
""")
#PAGE 4 – Product Demand Segments
# ==========================================================
# PAGE 4
# ==========================================================

elif page == "Demand Segments":

    st.header("📦 Product Demand Segmentation")

    product = df.groupby("Sub-Category").agg(
        TotalSales=("Sales","sum"),
        AvgOrder=("Sales","mean"),
        Volatility=("Sales","std")
    ).reset_index()

    yearly = df.groupby(
        ["Sub-Category","Year"]
    )["Sales"].sum().reset_index()

    growth = []

    for sub in yearly["Sub-Category"].unique():

        temp = yearly[
            yearly["Sub-Category"] == sub
        ]

        growth.append(
            temp["Sales"].pct_change().mean()
        )

    product["GrowthRate"] = growth

    product.fillna(0, inplace=True)

    scaler = StandardScaler()

    X = scaler.fit_transform(
        product.drop("Sub-Category", axis=1)
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42
    )

    product["Cluster"] = kmeans.fit_predict(X)

    labels = {
        0:"High Volume Stable",
        1:"Growing Demand",
        2:"High Volatility",
        3:"Declining Demand"
    }

    product["Demand Segment"] = product["Cluster"].map(labels)

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(X)

    product["PCA1"] = pca_result[:,0]
    product["PCA2"] = pca_result[:,1]

    fig = px.scatter(
        product,
        x="PCA1",
        y="PCA2",
        color="Demand Segment",
        text="Sub-Category",
        title="Demand Segments"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Cluster Table")

    st.dataframe(
        product[
            [
                "Sub-Category",
                "TotalSales",
                "GrowthRate",
                "Demand Segment"
            ]
        ],
        use_container_width=True
    )

    st.subheader("Recommended Inventory Strategy")

    st.markdown("""

### High Volume Stable
Maintain high inventory with automated replenishment.

### Growing Demand
Increase stock gradually and monitor trends.

### High Volatility
Maintain safety stock and review inventory frequently.

### Declining Demand
Reduce procurement and avoid excess inventory.

""")
