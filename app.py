"""
Flask web application — COVID-19 Vaccination Dashboard
Run locally: python app.py
Deploy: gunicorn -w 4 app:app
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os, json

app = Flask(__name__)

# ── Data helpers ──────────────────────────────────────────────────────────────

SAMPLE_IMPACT = {
    "State/UT":          ["Maharashtra","Kerala","Karnataka","Tamil Nadu",
                           "Uttar Pradesh","Delhi","West Bengal","Gujarat",
                           "Rajasthan","Bihar"],
    "Total_Cases":       [8123456,5456789,4789654,4587344,3798654,
                           3254000,3300456,3056789,2923456,2765432],
    "Total_Deaths":      [149876,70345,40123,38654,23450,
                           26876,22456,20456,18456,17456],
    "Total_Recoveries":  [7960000,5370000,4720000,4540000,3770000,
                           3230000,3280000,3030000,2900000,2750000],
    "Active_Cases":      [134580,12344,6531,2690,1204,124,500,333,200,120],
}

SAMPLE_VACCINE = {
    "State/UT":                    ["Maharashtra","Kerala","Karnataka","Tamil Nadu",
                                    "Uttar Pradesh","Delhi","West Bengal","Gujarat",
                                    "Rajasthan","Bihar"],
    "Population (in millions)":    [125,36,68,75,230,19,100,72,81,125],
    "1st_Dose":                    [123000000,35000000,65000000,73000000,
                                    225000000,18500000,97000000,70000000,
                                    80000000,120000000],
    "2nd_Dose":                    [120000000,34000000,64000000,72000000,
                                    220000000,18000000,95000000,68000000,
                                    78000000,118000000],
    "Booster_Dose":                [60000000,15000000,25000000,30000000,
                                    90000000,7000000,35000000,30000000,
                                    32000000,45000000],
    "Total_Vaccinated":            [303000000,84000000,154000000,175000000,
                                    535000000,43500000,227000000,168000000,
                                    190000000,283000000],
}


def load_data():
    """Load from Excel if present, else use sample data."""
    xlsx = "covid_vaccine_statewise_23.xlsx"
    if os.path.exists(xlsx):
        df_i = pd.read_excel(xlsx, sheet_name="covid_impact")
        df_v = pd.read_excel(xlsx, sheet_name="vaccination_data")
    else:
        df_i = pd.DataFrame(SAMPLE_IMPACT)
        df_v = pd.DataFrame(SAMPLE_VACCINE)

    df = pd.merge(df_i, df_v, on="State/UT")
    df["Vaccination_Rate_%"] = (
        df["Total_Vaccinated"] / (df["Population (in millions)"] * 1_000_000)
    ) * 100
    df["Cases_Per_Million"] = df["Total_Cases"] / df["Population (in millions)"]
    return df


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    df = load_data()
    return jsonify({
        "total_doses":      int(df["Total_Vaccinated"].sum()),
        "avg_vax_rate":     round(float(df["Vaccination_Rate_%"].mean()), 1),
        "total_cases":      int(df["Total_Cases"].sum()),
        "total_deaths":     int(df["Total_Deaths"].sum()),
        "states_covered":   len(df),
    })


@app.route("/api/states")
def api_states():
    df = load_data()
    records = df[["State/UT","Total_Cases","Total_Deaths",
                  "Total_Vaccinated","Vaccination_Rate_%",
                  "Cases_Per_Million","1st_Dose","2nd_Dose","Booster_Dose"]].copy()
    records["Vaccination_Rate_%"] = records["Vaccination_Rate_%"].round(2)
    records["Cases_Per_Million"]  = records["Cases_Per_Million"].round(2)
    return jsonify(records.to_dict(orient="records"))


@app.route("/api/regression")
def api_regression():
    df = load_data()
    X = df[["Vaccination_Rate_%"]].values
    y = df["Total_Cases"].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return jsonify({
        "coefficient": round(float(model.coef_[0]), 4),
        "intercept":   round(float(model.intercept_), 4),
        "r2_score":    round(float(r2_score(y, y_pred)), 4),
        "mse":         round(float(mean_squared_error(y, y_pred)), 2),
        "points": [
            {"state": s, "vax_rate": round(float(r), 2), "cases": int(c)}
            for s, r, c in zip(
                df["State/UT"].tolist(),
                df["Vaccination_Rate_%"].tolist(),
                df["Total_Cases"].tolist(),
            )
        ],
    })


@app.route("/api/correlation")
def api_correlation():
    df = load_data()
    cols = ["Total_Cases","Total_Deaths","Vaccination_Rate_%",
            "Cases_Per_Million"]
    corr = df[cols].corr().round(4)
    return jsonify(corr.to_dict())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
