# State-wise COVID-19 Impact vs. Vaccination Coverage in India

**Foundation of Data Science — B.Tech CSE Project**  
Alliance School of Advanced Computing, Alliance University, Bengaluru  
Authors: V. Ganga Purna Chandu Reddy & Mahi Kala  
Supervisor: Ms. Jaspine Bami Rani  

---

## Project structure

```
covid_project/
├── analysis.py          ← full Python analysis (EDA + ML + NLP + charts)
├── app.py               ← Flask web app with REST API
├── templates/
│   └── index.html       ← interactive web dashboard
├── requirements.txt     ← Python dependencies
├── Procfile             ← for Heroku / Railway deployment
├── output/              ← generated charts (created at runtime)
│   ├── fig1_cases_vs_vaccinated.png
│   ├── fig2_vaccination_rate.png
│   ├── fig3_gender_and_dose.png
│   ├── fig4_scatter_regression.png
│   └── fig5_correlation_heatmap.png
└── README.md
```

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
Place `covid_vaccine_statewise_23.xlsx` (with sheets `covid_impact` and `vaccination_data`) in the project root.  
If the file is absent the app uses built-in sample data so you can still run everything.

### 3. Run the standalone analysis script
```bash
python analysis.py
```
This will:
- Print full EDA (shape, dtypes, null counts, describe, correlations)
- Train a linear regression model and print R² + MSE
- Run NLP tokenisation / stemming on sample headlines
- Save 5 charts to `output/`

### 4. Run the web dashboard locally
```bash
python app.py
# Then open: http://localhost:5000
```

---

## Deployment

### Option A — Render (free tier)
1. Push this folder to a GitHub repository.
2. Create a new **Web Service** on https://render.com.
3. Set **Build command**: `pip install -r requirements.txt`
4. Set **Start command**: `gunicorn app:app`
5. Deploy — your dashboard is live at `https://<your-app>.onrender.com`.

### Option B — Railway
1. `railway login && railway init`
2. `railway up`
3. Set `PORT` env var if required (Flask reads it automatically).

### Option C — Heroku
```bash
heroku create covid-vax-india
git push heroku main
heroku open
```

### Option D — Docker
```bash
docker build -t covid-dashboard .
docker run -p 5000:5000 covid-dashboard
```

---

## API endpoints (Flask)

| Endpoint | Description |
|----------|-------------|
| `GET /` | Interactive dashboard UI |
| `GET /api/summary` | Aggregated KPIs (total doses, avg rate, etc.) |
| `GET /api/states` | Per-state vaccination and COVID data |
| `GET /api/regression` | Linear regression coefficients + R² |
| `GET /api/correlation` | Pearson correlation matrix |

---

## Technologies used

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data loading, cleaning, merging |
| NumPy | Numerical operations |
| Matplotlib | Chart generation |
| Scikit-learn | Linear regression, metrics |
| NLTK | Tokenisation, stemming, stopword removal |
| Flask | Web framework & REST API |
| Gunicorn | Production WSGI server |
| Chart.js | Interactive browser charts |

---

## Key findings

- **16 of 21 states** showed significant reduction in infection growth rate post-vaccination.
- A **~20-day lag** exists between first dose administration and measurable impact.
- **53.3% male** vs **46.7% female** vaccination ratio observed nationally.
- States with higher vaccination rates generally had **lower active case ratios**.
- Local factors (testing rates, NPI enforcement, variant prevalence) modulate effectiveness.

---

## References

- https://www.kaggle.com/datasets/swatikhedekar/state-wise-india-covid19vaccination
- https://arxiv.org/abs/2208.11998
- https://data.gov.in
- Ministry of Health and Family Welfare — https://mohfw.gov.in
