import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
from bs4 import BeautifulSoup
import plotly.express as px

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Gold & Silver Rates Dashboard", page_icon="chart_with_upwards_trend", layout="wide")
st.markdown("""
    <style>
    body {background-color:#0e1117;color:#fafafa;}
    .stMetric {text-align:center !important;}
    .css-1dp5vir,.stApp {background-color:#0e1117;color:white;}
    table {color:white !important;}
    </style>
""", unsafe_allow_html=True)
st.title("**Live Gold & Silver Rates Dashboard**")

# ------------------ SIDEBAR TIME FILTER ------------------
st.sidebar.header("calendar Time Period")
time_option = st.sidebar.selectbox("Select Period", ["7 days", "30 days", "90 days", "1 year"])
period_map = {
    "7 days": ("7d", "1d"),
    "30 days": ("1mo", "1d"),
    "90 days": ("3mo", "1d"),
    "1 year": ("1y", "1d")
}
period, interval = period_map[time_option]

# ------------------ FETCH LIVE FUTURES & FX ------------------
gold_ticker = "GC=F"
silver_ticker = "SI=F"
usd_inr_ticker = "USDINR=X"

with st.spinner(f"Fetching {time_option} of live market data..."):
    try:
        gold_df = yf.download(gold_ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        silver_df = yf.download(silver_ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        usd_inr = yf.download(usd_inr_ticker, period=period, interval=interval, progress=False, auto_adjust=True)
       
        if gold_df.empty or len(gold_df) < 2:
            st.warning("Period fetch failed. Trying with exact dates...")
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days={"7 days": 7, "30 days": 30, "90 days": 90, "1 year": 365}[time_option])
            gold_df = yf.download(gold_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            silver_df = yf.download(silver_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            usd_inr = yf.download(usd_inr_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

if gold_df.empty or silver_df.empty or usd_inr.empty:
    st.error("No data received. Try a shorter period or check internet.")
    st.stop()

gold_df.index = pd.to_datetime(gold_df.index)
silver_df.index = pd.to_datetime(silver_df.index)
usd_inr.index = pd.to_datetime(usd_inr.index)
st.success(f"Loaded {len(gold_df)} trading days from {gold_df.index[0].date()} to {gold_df.index[-1].date()}")

# ------------------ CONVERT TO INR PER GRAM ------------------
usd_to_inr = float(usd_inr['Close'].iloc[-1])
gold_df['INR_per_gram'] = gold_df['Close'] * usd_to_inr / 31.1035
silver_df['INR_per_gram'] = silver_df['Close'] * usd_to_inr / 31.1035

# ------------------ PRICE CHANGE + VOLATILITY ------------------
def calc_metrics(df):
    last = df['INR_per_gram'].iloc[-1]
    prev = df['INR_per_gram'].iloc[-2] if len(df) > 1 else last
    change_pct = ((last - prev) / prev) * 100 if prev != 0 else 0
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
full_df = pd.DataFrame(index=gold_df.index)
gold_reindexed = gold_df['INR_per_gram']
silver_reindexed = silver_df['INR_per_gram'].reindex(gold_df.index, method='nearest')
history = pd.DataFrame({
    "Date": full_df.index.date,
    "Gold": gold_reindexed.round(2),
    "Silver": silver_reindexed.round(2),
})
history["Gold %Δ"] = gold_reindexed.pct_change() * 100
history["Silver %Δ"] = silver_reindexed.pct_change() * 100

def label_trend(x):
    if pd.isna(x): return "No Data"
    elif abs(x) < 0.3: return "Stable"
    elif x > 0.3: return "Uptrend"
    else: return "Downtrend"

history["Trend"] = history["Gold %Δ"].apply(label_trend)
history = history.reset_index(drop=True)
history.index = history.index + 1

styled_history = history.style.format({
    "Gold": lambda x: "₹{:.2f}".format(x) if pd.notna(x) else "—",
    "Silver": lambda x: "₹{:.2f}".format(x) if pd.notna(x) else "—",
    "Gold %Δ": lambda x: "{:.2f}%".format(x) if pd.notna(x) else "—",
    "Silver %Δ": lambda x: "{:.2f}%".format(x) if pd.notna(x) else "—",
}, na_rep="—")
st.dataframe(styled_history, use_container_width=True, height=400)

csv = history.copy()
csv["Date"] = csv["Date"].astype(str)
csv = csv.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Price History (CSV)",
    data=csv,
    file_name=f"gold_silver_history_{gold_df.index[0].date()}_to_{gold_df.index[-1].date()}.csv",
    mime="text/csv",
    help="Download actual trading days only"
)

# ------------------ PRICE CHARTS ------------------
st.markdown("### Price Chart")
option = st.radio("Select Metal", ["Gold", "Silver"], horizontal=True)
if option == "Gold":
    st.line_chart(gold_reindexed, use_container_width=True)
else:
    st.line_chart(silver_reindexed, use_container_width=True)

# ======================================================
# ================== LIVE ADD-ONS ======================
# ======================================================

# -------------------- CITY SCRAPER --------------------

@st.cache_data(ttl=300)
def get_city_live_rates():
    url = "https://www.goodreturns.in/gold-rates/"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, 'html.parser')

        table = soup.find("table")
        if table is None:
            raise Exception("Table not found")

        rows = table.find_all("tr")[1:12]
        data = []

        for tr in rows:
            cols = tr.find_all("td")
            if len(cols) >= 2:
                city = cols[0].text.strip()
                gold = cols[1].text.replace("₹","").replace(",","").strip()

                if gold.replace('.', "").isdigit():
                    data.append({"City": city, "Gold": float(gold)})

        df = pd.DataFrame(data)
        df["Silver"] = np.nan
        return df.dropna()

    except Exception as e:
        st.info("Fetching city rates failed — using fallback values.")
        cities = ["Mumbai","Delhi","Bangalore","Chennai","Kolkata","Hyderabad","Pune","Ahmedabad","Jaipur","Lucknow"]
        return pd.DataFrame({
            "City": cities,
            "Gold": np.random.normal(gold_price, gold_price * 0.01, 10),
            "Silver": np.nan
        }).round(2)


# -------------------- CITY CHART --------------------
st.markdown("### Top 10 Highest Gold & Silver Rates Today")

city_df = get_city_live_rates()
selected_metal = st.selectbox("Select Metal", ["Gold", "Silver"], key="city")

top_cities = city_df.sort_values(selected_metal, ascending=False).head(10)

fig = px.bar(
    top_cities,
    x="City",
    y=selected_metal,
    text=selected_metal,
    color_discrete_sequence=["#DAA520"] if selected_metal=="Gold" else ["#A9A9A9"],  # Single tone
)

fig.update_traces(
    texttemplate='₹%{y:.0f}',
    textposition='outside'
)

fig.update_layout(
    title=f"Top 10 Cities with Highest {selected_metal} Rates",
    xaxis_title="City",
    yaxis_title="Price (INR/gram)",
    template="plotly_dark",
    showlegend=False,
    margin=dict(l=20,r=20,t=60,b=20)
)

st.plotly_chart(fig, use_container_width=True)

best = top_cities.iloc[0]
worst = city_df.loc[city_df[selected_metal].idxmin()]
st.success(f"Highest: **{best['City']}** → ₹{best[selected_metal]:,.2f}")
st.error(f"Lowest: **{worst['City']}** → ₹{worst[selected_metal]:,.2f}")

# ------------------ CORRELATION HEATMAP ------------------
st.markdown("### Correlation Heatmap: Gold vs Silver ")

corr_df = pd.DataFrame({
    'Gold (INR/g)': gold_df['INR_per_gram'],
    'Silver (INR/g)': silver_df['INR_per_gram']
}).dropna()

corr_matrix = corr_df.corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    labels=dict(color="Correlation Coefficient")
)
fig_corr.update_layout(
    template="plotly_dark",
    margin=dict(l=40, r=40, t=40, b=40),
    xaxis=dict(side="top"),
)
st.plotly_chart(fig_corr, use_container_width=True)

val = corr_matrix.iloc[0,1]
if val>0.8: st.success(f"✅ Strong correlation ({val:.2f})")
elif val>0.4: st.warning(f"⚠️ Moderate correlation ({val:.2f})")
else: st.error(f"❌ Weak correlation ({val:.2f})")

# ------------------ % CHANGE CHART ------------------
st.markdown("### Daily % Change (Gold & Silver)")
pct_df = pd.DataFrame({
    'Date': gold_df.index.date,
    'Gold %Δ': gold_df['INR_per_gram'].pct_change() * 100,
    'Silver %Δ': silver_df['INR_per_gram'].pct_change() * 100
}).dropna()

if not pct_df.empty:
    fig_pct = px.line(
        pct_df,
        x="Date",
        y=["Gold %Δ", "Silver %Δ"],
        color_discrete_map={"Gold %Δ": "#FFD700", "Silver %Δ": "#C0C0C0"},
        labels={"value": "Daily Change (%)"}
    )
    fig_pct.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
    )
    fig_pct.add_hline(y=0,line_dash="dot",line_color="gray",opacity=0.5)
    st.plotly_chart(fig_pct, use_container_width=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("**Data Sources:** Yahoo Finance • GoodReturns.in | Live Dashboard | Built with Streamlit")
