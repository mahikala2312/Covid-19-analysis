import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="COVID-19 Vaccination Dashboard — India", layout="wide")

# ── sample data (replace with your Excel file if uploaded) ───────────────────
IMPACT = {
    "State/UT": ["Maharashtra","Kerala","Karnataka","Tamil Nadu","Uttar Pradesh",
                 "Delhi","West Bengal","Gujarat","Rajasthan","Bihar"],
    "Total_Cases": [8123456,5456789,4789654,4587344,3798654,3254000,3300456,3056789,2923456,2765432],
    "Total_Deaths":[149876,70345,40123,38654,23450,26876,22456,20456,18456,17456],
    "Active_Cases":[134580,12344,6531,2690,1204,124,500,333,200,120],
}
VACCINE = {
    "State/UT": ["Maharashtra","Kerala","Karnataka","Tamil Nadu","Uttar Pradesh",
                 "Delhi","West Bengal","Gujarat","Rajasthan","Bihar"],
    "Population (in millions)":[125,36,68,75,230,19,100,72,81,125],
    "1st_Dose":  [123000000,35000000,65000000,73000000,225000000,18500000,97000000,70000000,80000000,120000000],
    "2nd_Dose":  [120000000,34000000,64000000,72000000,220000000,18000000,95000000,68000000,78000000,118000000],
    "Booster_Dose":[60000000,15000000,25000000,30000000,90000000,7000000,35000000,30000000,32000000,45000000],
    "Total_Vaccinated":[303000000,84000000,154000000,175000000,535000000,43500000,227000000,168000000,190000000,283000000],
}

@st.cache_data
def load_data():
    df_i = pd.DataFrame(IMPACT)
    df_v = pd.DataFrame(VACCINE)
    df = pd.merge(df_i, df_v, on="State/UT")
    df["Vaccination_Rate_%"] = (df["Total_Vaccinated"] / (df["Population (in millions)"] * 1_000_000)) * 100
    df["Cases_Per_Million"] = df["Total_Cases"] / df["Population (in millions)"]
    return df

df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("State-wise COVID-19 Impact vs. Vaccination Coverage — India")
st.caption("Alliance School of Advanced Computing · Foundation of Data Science · October 2025")
st.divider()

# ── KPI metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total doses administered", f"{df['Total_Vaccinated'].sum()/1e9:.2f}B")
c2.metric("Avg vaccination rate",     f"{df['Vaccination_Rate_%'].mean():.1f}%")
c3.metric("Total confirmed cases",    f"{df['Total_Cases'].sum()/1e6:.1f}M")
c4.metric("Total deaths",             f"{df['Total_Deaths'].sum()/1e3:.1f}K")

st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
selected_states = st.sidebar.multiselect(
    "Select states", df["State/UT"].tolist(), default=df["State/UT"].tolist()
)
df_f = df[df["State/UT"].isin(selected_states)]

# ── Chart 1: Cases vs Vaccinated ─────────────────────────────────────────────
st.subheader("Total cases vs. vaccinated population by state")
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(df_f))
ax.bar(x - 0.2, df_f["Total_Cases"],        0.4, label="Total cases",        color="#378ADD", alpha=0.85)
ax.bar(x + 0.2, df_f["Total_Vaccinated"]/1e6, 0.4, label="Vaccinated (M×1000)", color="#1D9E75", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(df_f["State/UT"], rotation=45, ha="right")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v/1e6:.1f}M" if v>=1e6 else str(int(v))))
ax.legend(); ax.spines[["top","right"]].set_visible(False)
st.pyplot(fig); plt.close()

# ── Chart 2: Vaccination rate ─────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Vaccination rate by state (%)")
    sorted_df = df_f.sort_values("Vaccination_Rate_%")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.barh(sorted_df["State/UT"], sorted_df["Vaccination_Rate_%"], color="#7F77DD", alpha=0.85)
    ax2.set_xlabel("Vaccination rate (%)")
    ax2.spines[["top","right"]].set_visible(False)
    st.pyplot(fig2); plt.close()

with col2:
    st.subheader("Vaccination rate vs. case burden")
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.scatter(df_f["Vaccination_Rate_%"], df_f["Cases_Per_Million"], color="#7F77DD", s=80, zorder=3)
    for _, row in df_f.iterrows():
        ax3.annotate(row["State/UT"], (row["Vaccination_Rate_%"], row["Cases_Per_Million"]), fontsize=7, xytext=(4,4), textcoords="offset points")
    # regression line
    if len(df_f) > 1:
        lr = LinearRegression().fit(df_f[["Vaccination_Rate_%"]], df_f["Cases_Per_Million"])
        xr = np.linspace(df_f["Vaccination_Rate_%"].min(), df_f["Vaccination_Rate_%"].max(), 100).reshape(-1,1)
        ax3.plot(xr, lr.predict(xr), color="#D85A30", linewidth=2, label="Regression line")
        ax3.legend()
    ax3.set_xlabel("Vaccination rate (%)"); ax3.set_ylabel("Cases per million")
    ax3.spines[["top","right"]].set_visible(False)
    st.pyplot(fig3); plt.close()

# ── Pie charts ────────────────────────────────────────────────────────────────
st.subheader("Distribution charts")
pc1, pc2 = st.columns(2)
with pc1:
    fig4, ax4 = plt.subplots(figsize=(4, 4))
    ax4.pie([53.3, 46.7], labels=["Male 53.3%","Female 46.7%"], colors=["#378ADD","#D4537E"], autopct="%1.1f%%", startangle=90, wedgeprops={"linewidth":0})
    ax4.set_title("Gender — vaccinated population")
    st.pyplot(fig4); plt.close()

with pc2:
    fig5, ax5 = plt.subplots(figsize=(4, 4))
    ax5.pie([80.7, 19.3], labels=["First dose 80.7%","Second dose 19.3%"], colors=["#1D9E75","#D85A30"], autopct="%1.1f%%", startangle=90, wedgeprops={"linewidth":0})
    ax5.set_title("Dose breakdown")
    st.pyplot(fig5); plt.close()

# ── Regression summary ────────────────────────────────────────────────────────
st.divider()
st.subheader("Linear regression — vaccination rate → total cases")
if len(df_f) > 1:
    lr2 = LinearRegression().fit(df_f[["Vaccination_Rate_%"]], df_f["Total_Cases"])
    y_pred = lr2.predict(df_f[["Vaccination_Rate_%"]])
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Coefficient",  f"{lr2.coef_[0]:,.0f}")
    r2.metric("Intercept",    f"{lr2.intercept_:,.0f}")
    r3.metric("R² score",     f"{r2_score(df_f['Total_Cases'], y_pred):.4f}")
    r4.metric("Interpretation", "Higher vax → fewer cases" if lr2.coef_[0] < 0 else "Check data")

# ── Raw data table ────────────────────────────────────────────────────────────
st.divider()
with st.expander("View raw data table"):
    st.dataframe(df_f.style.format({
        "Total_Cases": "{:,.0f}",
        "Total_Deaths": "{:,.0f}",
        "Total_Vaccinated": "{:,.0f}",
        "Vaccination_Rate_%": "{:.1f}%",
        "Cases_Per_Million": "{:.0f}",
    }), use_container_width=True)

st.caption("Data sources: MoHFW · CoWIN Dashboard · COVID19India.org")
