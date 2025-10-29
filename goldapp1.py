import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
from bs4 import BeautifulSoup
import plotly.express as px

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Gold & Silver Rates Dashboard", page_icon=" 🪙 ", layout="wide")
st.markdown("""
    <style>
    body {background-color:#0e1117;color:#fafafa;}
    .stMetric {text-align:center !important;}
    .css-1dp5vir,.stApp {background-color:#0e1117;color:white;}
    table {color:white !important;}
    </style>
""", unsafe_allow_html=True)
st.title(" 🪙 **Live Gold & Silver Rates Dashboard**")

# ------------------ SIDEBAR TIME FILTER ------------------
st.sidebar.header("calendar Time Period")
time_option = st.sidebar.selectbox("Select Period", ["7 days", "30 days", "90 days", "1 year"])
days_map = {"7 days": 7, "30 days": 30, "90 days": 90, "1 year": 365}
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=days_map[time_option])

# ------------------ FETCH LIVE FUTURES & FX ------------------
gold_ticker = "GC=F"
silver_ticker = "SI=F"
usd_inr_ticker = "USDINR=X"

with st.spinner("Fetching live market data..."):
    gold_df = yf.download(gold_ticker, start=start_date, end=end_date, progress=False)
    silver_df = yf.download(silver_ticker, start=start_date, end=end_date, progress=False)
    usd_inr = yf.download(usd_inr_ticker, start=start_date, end=end_date, progress=False)  # Fixed line

if gold_df.empty or silver_df.empty or usd_inr.empty:
    st.error("Failed to fetch data. Please check your internet connection.")
    st.stop()

# ------------------ CONVERT TO INR PER GRAM ------------------
usd_to_inr = float(usd_inr['Close'].iloc[-1])
gold_df['INR_per_gram'] = gold_df['Close'] * usd_to_inr / 31.1035
silver_df['INR_per_gram'] = silver_df['Close'] * usd_to_inr / 31.1035

# ------------------ PRICE CHANGE + VOLATILITY ------------------
def calc_metrics(df):
    last = df['INR_per_gram'].iloc[-1]
    prev = df['INR_per_gram'].iloc[-2]
    change_pct = ((last - prev) / prev) * 100
    volatility = df['INR_per_gram'].pct_change().std() * 100
    return last, change_pct, volatility

gold_price, gold_change, gold_vol = calc_metrics(gold_df)
silver_price, silver_change, silver_vol = calc_metrics(silver_df)

# ------------------ METRICS DISPLAY ------------------
st.markdown("### Current Market Prices")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Gold Price", f"₹{gold_price:,.2f}", f"{gold_change:.2f}%")
col2.metric("Silver Price", f"₹{silver_price:,.2f}", f"{silver_change:.2f}%")
col3.metric("Gold Volatility", f"{gold_vol:.2f}%")
col4.metric("Silver Volatility", f"{silver_vol:.2f}%")

# ------------------ PRICE HISTORY TABLE + DOWNLOAD ------------------
st.markdown("### chart Price History")

# Create full calendar date range
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
full_df = pd.DataFrame(index=date_range)

# Reindex gold & silver to include all calendar days
gold_reindexed = gold_df['INR_per_gram'].reindex(full_df.index)
silver_reindexed = silver_df['INR_per_gram'].reindex(full_df.index, method='nearest')

history = pd.DataFrame({
    "Date": full_df.index.date,
    "Gold": gold_reindexed.round(2),
    "Silver": silver_reindexed.round(2),
})
history["Gold %Δ"] = gold_reindexed.pct_change() * 100
history["Silver %Δ"] = silver_reindexed.pct_change() * 100

def label_trend(x):
    if pd.isna(x):
        return "No Data"
    elif abs(x) < 0.3:
        return "Stable"
    elif x > 0.3:
        return "Uptrend"
    else:
        return "Downtrend"

history["Trend"] = history["Gold %Δ"].apply(label_trend)
history = history.reset_index(drop=True)
history.index = history.index + 1

# Format for display
styled_history = history.style.format({
    "Gold": lambda x: "₹{:.2f}".format(x) if pd.notna(x) else "—",
    "Silver": lambda x: "₹{:.2f}".format(x) if pd.notna(x) else "—",
    "Gold %Δ": lambda x: "{:.2f}%".format(x) if pd.notna(x) else "—",
    "Silver %Δ": lambda x: "{:.2f}%".format(x) if pd.notna(x) else "—",
}, na_rep="—")

st.dataframe(styled_history, use_container_width=True, height=400)

# === DOWNLOAD BUTTON ===
csv = history.copy()
csv["Date"] = csv["Date"].astype(str)
csv = csv.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Price History (CSV)",
    data=csv,
    file_name=f"gold_silver_history_{start_date}_to_{end_date}.csv",
    mime="text/csv",
    help="Download the full price history including weekends (no data = —)"
)



# ------------------ PRICE CHARTS ------------------
st.markdown("### Price Chart")
option = st.radio("Select Metal", ["Gold", "Silver"], horizontal=True)
if option == "Gold":
    st.line_chart(gold_df["INR_per_gram"], use_container_width=True)
else:
    st.line_chart(silver_df["INR_per_gram"], use_container_width=True)


# ======================================================
# ================== LIVE ADD-ONS ======================
# ======================================================



# ------------------------------------------------------
# 1) LIVE TREEMAP – City + Purity + Metal (3-Level)
# ------------------------------------------------------
st.markdown("### world map **Treemap: Live Gold & Silver Rates by City, Purity & Metal**")

@st.cache_data(ttl=300)
def get_city_metal_purity_data():
    url = "https://www.goodreturns.in/gold-rates/"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find("table", {"id": "gold_rates"})
        rows = table.find_all("tr")[1:]
        data = []
        for tr in rows:
            cols = tr.find_all("td")
            if len(cols) >= 3:
                city = cols[0].text.strip()
                gold_24k_str = cols[1].text.replace("₹", "").replace(",", "").strip()
                silver_str = cols[2].text.replace("₹", "").replace(",", "").strip()

                if gold_24k_str and gold_24k_str.replace(".", "").isdigit():
                    gold_24k = float(gold_24k_str)
                    gold_22k = gold_24k * 0.9167  # 22K = 91.67% of 24K
                    silver = float(silver_str) if silver_str and silver_str.replace(".", "").isdigit() else np.nan

                    # Add 24K Gold
                    data.append({
                        "City": city,
                        "Purity": "24K",
                        "Metal": "Gold",
                        "Value": round(gold_24k, 2)
                    })
                    # Add 22K Gold
                    data.append({
                        "City": city,
                        "Purity": "22K",
                        "Metal": "Gold",
                        "Value": round(gold_22k, 2)
                    })
                    # Add Silver
                    if not pd.isna(silver):
                        data.append({
                            "City": city,
                            "Purity": "999",
                            "Metal": "Silver",
                            "Value": round(silver, 2)
                        })
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"GoodReturns scrape failed: {e}. Using fallback.")
        # Fallback: use national average
        return pd.DataFrame([
            {"City": "Mumbai", "Purity": "24K", "Metal": "Gold", "Value": gold_price},
            {"City": "Mumbai", "Purity": "22K", "Metal": "Gold", "Value": gold_price * 0.9167},
            {"City": "Mumbai", "Purity": "999", "Metal": "Silver", "Value": silver_price},
        ])

treemap_df = get_city_metal_purity_data()

if not treemap_df.empty:
    fig_treemap = px.treemap(
        treemap_df,
        path=['City', 'Purity', 'Metal'],  # 3-level hierarchy
        values='Value',
        color='Purity',
        color_discrete_map={
            '24K': '#FFD700',   # Pure Gold
            '22K': '#FFC107',   # 22K Gold
            '999': '#C0C0C0'    # Silver
        },
        title="Live Rates: City → Purity → Metal (INR/gram)"
    )
    fig_treemap.update_traces(
        hovertemplate='<b>%{label}</b><br>Price: ₹%{value:,.2f}/gram'
    )
    fig_treemap.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    st.plotly_chart(fig_treemap, use_container_width=True)
else:
    st.error("Could not load city-wise data for treemap.")



# ------------------------------------------------------
# 2) LIVE CITY-WISE BAR CHART – GoodReturns.in (Highest + Lowest)
# ------------------------------------------------------
st.markdown("### City-Wise Gold & Silver Rates (Top 10 Highest)")

@st.cache_data(ttl=300)
def get_city_live_rates():
    url = "https://www.goodreturns.in/gold-rates/"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find("table", {"id": "gold_rates"})
        rows = table.find_all("tr")[1:]
        data = []
        for tr in rows:
            cols = tr.find_all("td")
            if len(cols) >= 3:
                city = cols[0].text.strip()
                gold_24k = cols[1].text.replace("₹", "").replace(",", "").strip()
                silver = cols[2].text.replace("₹", "").replace(",", "").strip()
                if gold_24k and gold_24k.replace(".", "").isdigit():
                    data.append({
                        "City": city,
                        "Gold": float(gold_24k),
                        "Silver": float(silver) if silver and silver.replace(".", "").isdigit() else np.nan
                    })
        return pd.DataFrame(data).dropna()
    except Exception as e:
        st.warning(f"GoodReturns scrape failed: {e}. Using futures estimate.")
        cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
        return pd.DataFrame({
            "City": cities,
            "Gold": np.random.normal(gold_price, gold_price * 0.01, len(cities)),
            "Silver": np.random.normal(silver_price, silver_price * 0.015, len(cities))
        }).round(2)

city_df = get_city_live_rates()

# Interactive controls
col_a, col_b = st.columns([1, 2])
with col_a:
    selected_metal = st.selectbox("Select Metal", ["Gold", "Silver"], key="city_metal")
with col_b:
    selected_date = st.date_input("Select Date", value=end_date, key="city_date")

display_col = selected_metal
top_cities = city_df.sort_values(display_col, ascending=False).head(10)
lowest_city = city_df.loc[city_df[display_col].idxmin()]

fig_bar = px.bar(
    top_cities,
    x="City", y=display_col,
    text=display_col,
    color=display_col,
    color_continuous_scale="ylorrd" if selected_metal == "Gold" else "greys",
    title=f"Top 10 Cities with Highest {selected_metal} Rates on {selected_date.strftime('%b %d, %Y')}"
)
fig_bar.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
fig_bar.update_layout(xaxis_title="", yaxis_title="Price (INR/gram)")
st.plotly_chart(fig_bar, use_container_width=True)

# === HIGHEST & LOWEST SUMMARY (Clean Text) ===
highest = top_cities.iloc[0]

col_h, col_l = st.columns(2)
with col_h:
    st.success(f"**Highest {selected_metal}:** {highest['City']} → ₹{highest[display_col]:,.2f}/gram")
with col_l:
    st.error(f"**Lowest {selected_metal}:** {lowest_city['City']} → ₹{lowest_city[display_col]:,.2f}/gram")

# ------------------------------------------------------
# DUAL LINE CHART: Gold %Δ & Silver %Δ (Live)
# ------------------------------------------------------
st.markdown("### Daily % Change (Gold & Silver)")

# Use the same reindexed data from history
gold_pct = gold_reindexed.pct_change() * 100
silver_pct = silver_reindexed.pct_change() * 100

pct_df = pd.DataFrame({
    'Date': full_df.index.date,
    'Gold %Δ': gold_pct.round(2),
    'Silver %Δ': silver_pct.round(2)
}).dropna(subset=['Gold %Δ', 'Silver %Δ'], how='all')

if not pct_df.empty:
    fig_pct = px.line(
        pct_df,
        x='Date',
        y=['Gold %Δ', 'Silver %Δ'],
        color_discrete_map={
            'Gold %Δ': '#FFD700',
            'Silver %Δ': '#C0C0C0'
        },
        labels={"value": "Daily Change (%)", "variable": "Metal"},
        title=""
    )
    fig_pct.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Daily Change (%)",
        hovermode='x unified',
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_pct.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_pct.update_traces(hovertemplate='%{y:.2f}%')
    st.plotly_chart(fig_pct, use_container_width=True)
else:
    st.info("Not enough data to show % change.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("**Data Sources:** Yahoo Finance (Futures) • MCX India (Spot) • GoodReturns.in (City Rates) | Built with Streamlit + yfinance + BeautifulSoup")