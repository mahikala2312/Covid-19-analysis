"""
State-wise COVID-19 Impact vs. Vaccination Coverage in India
Foundation of Data Science Project
Authors: Mahi Kala
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import os

# ── output directory ─────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

# ── 1. Load datasets ──────────────────────────────────────────────────────────
print("=" * 60)
print("COVID-19 Vaccination Analysis — India")
print("=" * 60)

df_impact = pd.read_excel(
    "covid_vaccine_statewise_23.xlsx", sheet_name="covid_impact"
)
df_vaccine = pd.read_excel(
    "covid_vaccine_statewise_23.xlsx", sheet_name="vaccination_data"
)

# ── 2. Exploration ────────────────────────────────────────────────────────────
print("\n[1] COVID-19 Impact Data — first 5 rows:")
print(df_impact.head().to_string(index=False))

print("\n[2] Vaccination Data — first 5 rows:")
print(df_vaccine.head().to_string(index=False))

print(f"\n[3] Dataset shapes:")
print(f"    COVID Impact  : {df_impact.shape}")
print(f"    Vaccination   : {df_vaccine.shape}")

print("\n[4] Column names:")
print(f"    COVID Impact  : {df_impact.columns.tolist()}")
print(f"    Vaccination   : {df_vaccine.columns.tolist()}")

print("\n[5] COVID Impact — data info:")
df_impact.info()

print("\n[6] Vaccination — data info:")
df_vaccine.info()

print("\n[7] Null value counts:")
print("    COVID Impact:")
print(df_impact.isnull().sum().to_string())
print("    Vaccination:")
print(df_vaccine.isnull().sum().to_string())

print("\n[8] Descriptive statistics — COVID Impact:")
print(df_impact.describe().to_string())

print("\n[9] Descriptive statistics — Vaccination:")
print(df_vaccine.describe().to_string())

print(f"\n[10] Duplicate rows:")
print(f"     COVID Impact  : {df_impact.duplicated().sum()}")
print(f"     Vaccination   : {df_vaccine.duplicated().sum()}")

print(f"\n[11] Unique states in vaccination data:")
print(df_vaccine["State/UT"].unique())

# ── 3. Merge ──────────────────────────────────────────────────────────────────
df_merged = pd.merge(df_impact, df_vaccine, on="State/UT")
print(f"\n[12] Merged dataset shape: {df_merged.shape}")
print(df_merged.head().to_string(index=False))

# ── 4. Feature engineering ────────────────────────────────────────────────────
df_merged["Vaccination_Rate_%"] = (
    df_merged["Total_Vaccinated"]
    / (df_merged["Population (in millions)"] * 1_000_000)
) * 100

df_merged["Cases_Per_Million"] = (
    df_merged["Total_Cases"]
    / df_merged["Population (in millions)"]
)

df_merged["Deaths_Per_Million"] = (
    df_merged["Total_Deaths"]
    / df_merged["Population (in millions)"]
)

sorted_vaccine = df_merged.sort_values("Vaccination_Rate_%", ascending=False)
print("\n[13] Top 5 states by vaccination coverage:")
print(
    sorted_vaccine[["State/UT", "Vaccination_Rate_%"]]
    .head()
    .to_string(index=False)
)

# ── 5. Correlation ────────────────────────────────────────────────────────────
corr = df_merged[
    ["Total_Cases", "Total_Deaths", "Vaccination_Rate_%",
     "Cases_Per_Million", "Deaths_Per_Million"]
].corr()
print("\n[14] Pearson correlation matrix:")
print(corr.round(4).to_string())

# ── 6. Linear regression: vaccination rate → total cases ─────────────────────
X = df_merged[["Vaccination_Rate_%"]]
y = df_merged["Total_Cases"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n[15] Linear Regression — Vaccination Rate → Total Cases:")
print(f"     Coefficient  : {model.coef_[0]:,.2f}")
print(f"     Intercept    : {model.intercept_:,.2f}")
print(f"     R² Score     : {r2_score(y_test, y_pred):.4f}")
print(f"     MSE          : {mean_squared_error(y_test, y_pred):,.2f}")

# ── 7. State-wise: first dose ─────────────────────────────────────────────────
print("\n[16] First dose by state:")
fd = df_vaccine[["State/UT","1st_Dose"]].sort_values("1st_Dose", ascending=False)
print(fd.to_string(index=False))

print("\n[17] Second dose by state:")
sd = df_vaccine[["State/UT","2nd_Dose"]].sort_values("2nd_Dose", ascending=False)
print(sd.to_string(index=False))

# ── 8. Visualizations ────────────────────────────────────────────────────────
BLUE   = "#378ADD"
GREEN  = "#1D9E75"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
PINK   = "#D4537E"
GRAY   = "#888780"

states = df_merged["State/UT"].tolist()
x_idx  = np.arange(len(states))
width  = 0.35

# -- Fig 1: Cases vs Vaccinated bar --
fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x_idx - width/2, df_merged["Total_Cases"],
       width, label="Total cases", color=BLUE, alpha=0.85)
ax.bar(x_idx + width/2,
       df_merged["Total_Vaccinated"] / 1_000_000,
       width, label="Vaccinated (millions)", color=GREEN, alpha=0.85)
ax.set_xticks(x_idx)
ax.set_xticklabels(states, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v:,.0f}")
)
ax.set_title("State-wise COVID-19 cases vs. vaccination (in millions)", fontsize=13)
ax.set_xlabel("State / UT")
ax.set_ylabel("Count")
ax.legend()
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("output/fig1_cases_vs_vaccinated.png", dpi=150)
plt.close()
print("\n[18] Saved: output/fig1_cases_vs_vaccinated.png")

# -- Fig 2: Vaccination rate bar --
fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.barh(
    df_merged.sort_values("Vaccination_Rate_%")["State/UT"],
    df_merged.sort_values("Vaccination_Rate_%")["Vaccination_Rate_%"],
    color=PURPLE, alpha=0.85
)
ax.set_xlabel("Vaccination rate (%)")
ax.set_title("Vaccination rate by state (% of population)", fontsize=13)
ax.spines[["top","right"]].set_visible(False)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.5, bar.get_y() + bar.get_height()/2,
            f"{w:.1f}%", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("output/fig2_vaccination_rate.png", dpi=150)
plt.close()
print("[19] Saved: output/fig2_vaccination_rate.png")

# -- Fig 3: Gender pie --
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].pie([53.3, 46.7], labels=["Male", "Female"],
            colors=[BLUE, PINK], autopct="%1.1f%%",
            startangle=90, wedgeprops={"linewidth":0})
axes[0].set_title("Gender — vaccinated population")
axes[1].pie([80.7, 19.3], labels=["First dose", "Second dose"],
            colors=[GREEN, CORAL], autopct="%1.1f%%",
            startangle=90, wedgeprops={"linewidth":0})
axes[1].set_title("Dose breakdown")
plt.tight_layout()
plt.savefig("output/fig3_gender_and_dose.png", dpi=150)
plt.close()
print("[20] Saved: output/fig3_gender_and_dose.png")

# -- Fig 4: Scatter regression --
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df_merged["Vaccination_Rate_%"], df_merged["Cases_Per_Million"],
           color=PURPLE, s=80, zorder=3, label="States")

for _, row in df_merged.iterrows():
    ax.annotate(row["State/UT"],
                (row["Vaccination_Rate_%"], row["Cases_Per_Million"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")

# regression line
x_line = np.linspace(df_merged["Vaccination_Rate_%"].min(),
                     df_merged["Vaccination_Rate_%"].max(), 200).reshape(-1, 1)
y_line = LinearRegression().fit(
    df_merged[["Vaccination_Rate_%"]], df_merged["Cases_Per_Million"]
).predict(x_line)
ax.plot(x_line, y_line, color=CORAL, linewidth=2, label="Regression line")

ax.set_xlabel("Vaccination rate (%)")
ax.set_ylabel("COVID-19 cases per million population")
ax.set_title("Vaccination rate vs. case burden — state-level correlation", fontsize=12)
ax.legend()
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("output/fig4_scatter_regression.png", dpi=150)
plt.close()
print("[21] Saved: output/fig4_scatter_regression.png")

# -- Fig 5: Correlation heatmap --
fig, ax = plt.subplots(figsize=(7, 5))
cols = ["Total_Cases","Total_Deaths","Vaccination_Rate_%",
        "Cases_Per_Million","Deaths_Per_Million"]
corr_sub = df_merged[cols].corr()
im = ax.imshow(corr_sub.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
lbls = ["Cases","Deaths","Vax rate","Cases/M","Deaths/M"]
ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(lbls, fontsize=9)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr_sub.values[i,j]:.2f}",
                ha="center", va="center", fontsize=8, color="white" if abs(corr_sub.values[i,j])>0.5 else "black")
ax.set_title("Correlation heatmap", fontsize=12)
plt.tight_layout()
plt.savefig("output/fig5_correlation_heatmap.png", dpi=150)
plt.close()
print("[22] Saved: output/fig5_correlation_heatmap.png")

# ── 9. NLP: simple text analysis ─────────────────────────────────────────────
print("\n[23] NLP Techniques demonstration")
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

headlines = [
    "Kerala reports decline in cases after mass vaccination campaign.",
    "Bihar struggles with vaccine hesitancy in rural areas.",
    "Maharashtra leads COVID-19 vaccination drive with record doses.",
    "Uttar Pradesh intensifies rural outreach for second dose coverage.",
    "Karnataka achieves 70 percent full vaccination of eligible adults.",
    "Delhi records sharp fall in active cases amid booster rollout.",
    "Rajasthan launches mobile vaccination units in remote districts.",
    "Tamil Nadu health ministry reports drop in mortality rates.",
    "West Bengal strengthens cold chain infrastructure for vaccines.",
    "Gujarat's high vaccination rate linked to lower hospitalizations.",
]

stop_words = set(stopwords.words("english"))
stemmer    = PorterStemmer()
all_tokens = []

for text in headlines:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens  = word_tokenize(cleaned)
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    all_tokens.extend(filtered)

from collections import Counter
freq = Counter(all_tokens).most_common(10)
print("     Top 10 stems from news headlines:")
for stem, count in freq:
    print(f"       {stem:20s} {count}")

print("\n[24] Analysis complete — all outputs saved to ./output/")
print("=" * 60)
