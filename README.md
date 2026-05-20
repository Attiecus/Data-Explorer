# Data Explorer

A local analytics dashboard built with Streamlit. Upload an Excel or CSV file, it figures out what kind of data you have, and gives you charts, quality checks, ML tools, and automated alerts without any configuration.

No API key. Nothing sent anywhere. Runs on your machine.

```bash
pip install streamlit pandas openpyxl plotly scipy scikit-learn sentence-transformers umap-learn
streamlit run data_explorer.py
```

---

## What's in it

### Dashboard
The first thing you see when you upload a file. It picks the most meaningful numeric columns as KPI cards, chooses a sensible chart based on what columns exist (bar chart if you have categories, scatter if you only have numbers, area chart if there's a date), and surfaces a few observations about the data — skewed columns, dominant categories, missing values.

Nothing to configure, it just runs.

### Quality Audit
A 0–100 quality score based on null rate and duplicate rate. Below that: a bar chart of missing values by column, a heatmap showing *where* the nulls are (random vs clustered), IQR-based outlier detection for every numeric column, and a full schema table with min, max, mean, std, skew, outlier count, and a Shapiro-Wilk normality test result for each column.

### Distributions
Four views: histogram with an optional category split and box plot marginal, categorical value breakdown with bar + pie, violin/box/strip plots with grouping, and a Q-Q plot with a normality test. There's also a grid view that shows all numeric columns at once.

### Chart Studio
15 chart types. Most dashboards give you bar, line, pie. This also has:

- Treemap and Sunburst for hierarchical data
- Sankey for flows between two category columns
- Parallel coordinates for comparing many numeric columns simultaneously
- Waterfall for running totals
- Candlestick for OHLC price data
- Animated bar race — categories ranked over time, animated by period
- 2D density histogram, radar chart, 3D scatter, lollipop chart, stacked area

### Time Series
Only shows up if the app detects a date column. Lets you resample to daily / weekly / monthly / quarterly, switch between line, area, and bar, and overlays 3-period and 7-period rolling averages. There's also a month-over-month % change view with green/red bars.

### Compute
Creates new columns from your existing ones — they stay available across every other tab for the rest of the session. Operations: formula (any pandas eval expression), bin numeric into N groups, rank, normalize/standardize (min-max, Z-score, log, sqrt), rolling window stats, % of total, and lag.

### Segment
**K-Means** — pick your features, pick K, and it clusters your rows. The most useful output is the radar chart showing each cluster's mean values, which makes it obvious what characterises each group.

**PCA** — projects high-dimensional data to 2D or 3D. Use UMAP if you want to see natural clusters; PCA if you just want it fast.

**Top / Bottom N** — exactly what it says.

**Cohort analysis** — pivot heatmap of a metric by category × time period.

**Percentile bands** — splits a column into Bottom 10%, 10–25%, 25–75%, 75–90%, Top 10% and saves it as a new column.

### Relationships
Correlation heatmap (Pearson, with the 6 strongest pairs called out), scatter explorer with OLS trendline and significance test, scatter matrix, linear regression (R², RMSE, residuals, feature importance), violin + ANOVA for numeric vs category comparisons, and partial correlation which measures the relationship between two columns after controlling for confounders.

### Embeddings
The ML-heavy part. No API calls — everything runs locally.

**Text columns** use `all-MiniLM-L6-v2`, a 22 MB sentence transformer that downloads once and then works offline. It converts each row's text into a 384-dimensional vector. From there you get:

- Semantic search — type a query in natural language, it finds the most similar rows by meaning rather than exact words. Useful when your data has descriptions or notes.
- Embedding space — UMAP or PCA projection of all rows to 2D. Each point is a row, hover to read the text.
- Near-duplicate finder — finds pairs of rows above a cosine similarity threshold. Good for catching rows that say the same thing in different words.
- Semantic clusters — K-Means on the embedding vectors with UMAP visualisation and representative examples per cluster.

**Numeric columns** treat each standardised row as a vector:

- Similarity search — pick a row, find the K most similar rows. Radar chart compares the query to its top 3 matches.
- Anomaly detection — Isolation Forest on all numeric columns simultaneously. Unlike IQR outliers (which check one column at a time), this catches rows that are unusual across multiple dimensions at once.
- Distance matrix — cosine similarity heatmap between rows, with most similar and most dissimilar pair tables.

### Smart Alerts
Scans the active sheet and produces a plain-English report ranked by severity. Each alert has a title, an explanation, and a concrete recommendation.

What it checks:

- **Data quality** — null rates, worst column by missingness, duplicates, near-zero variance
- **Outliers** — extreme (4× IQR) and moderate (1.5× IQR)
- **Distribution** — heavy skew, with a specific transform recommendation
- **Correlations** — near-perfect (r > 0.90, multicollinearity warning) and strong (r > 0.75)
- **Trends** — linear regression on every numeric × date pair, flags significant trends (p < 0.05)
- **Class imbalance** — category columns dominated by one value
- **ML anomalies** — Isolation Forest on all numeric columns, separate from the per-column checks above
- **Positive signals** — tells you when things are actually fine

Severities are 🔴 Critical, 🟡 Warning, 🔵 Info, ✅ Good. Filterable by severity and category, exportable as Excel.

### Filter & Export
Numeric range sliders, category multi-select, date range pickers, and text search all applied together. Shows matching row count and a live aggregation of the filtered data. Export filtered data as Excel, CSV, or Excel with an auto-pivot sheet.

---

## Sample data

`sample_data.xlsx` has four sheets, each built to test different parts of the app.

**Sales** (800 rows) — dates, products, regions, reps, revenue, cost, profit, ratings. Has 5 revenue rows artificially spiked ×15 to test anomaly detection, and ~40 nulls scattered across rating and discount columns.

**Customers** (200 rows) — company descriptions and support notes written in plain English. The Description and Support Notes columns are the ones to embed — try searching for "companies having trouble with their data" or "customers who might leave".

**Sensors** (500 rows) — temperature, humidity, pressure, vibration, power draw, RPM across four zones. 20 rows have injected anomalies (temperature spikes, vibration bursts, power surges). Load this sheet and run Isolation Forest in the Embeddings tab.

**HR** (300 rows) — salary, performance, satisfaction, tenure, department, attrition risk, hire dates. Good for K-Means (run it with Salary, Performance Score, Satisfaction, Tenure — you'll get meaningful clusters), and for regression (does tenure predict salary after controlling for department?).

---

## Installation

```bash
pip install streamlit pandas openpyxl plotly scipy scikit-learn sentence-transformers umap-learn
```

The sentence transformer model (`all-MiniLM-L6-v2`) downloads to `~/.cache/huggingface/` the first time you use the Embeddings tab. After that it works with no internet connection.

No API key required for any feature in `data_explorer.py`. The `excel_agent.py` file (separate AI chat agent) needs a Gemini key if you want to use that.

```
set GEMINI_API_KEY=your_key_here   # Windows
export GEMINI_API_KEY=your_key_here  # Mac/Linux
```

---

## File structure

```
├── data_explorer.py     main dashboard
├── excel_agent.py       separate Gemini-powered chat agent
├── sample_data.xlsx     test data (4 sheets)
└── README.md
```

---

## A few things worth knowing

Date parsing is automatic for columns with `date`, `time`, `dt`, `_at`, or `_on` in the name. If a date column isn't being picked up, rename it.

Columns created in the Compute tab persist across the whole session. If you make a `Profit Margin` column it'll show up in the filter sliders, clustering features, and chart dropdowns immediately.

Embeddings are cached in session state. If you change the column or sample size you need to click "Compute embeddings" again.

The alert engine caches per sheet. Switch to a different sheet and back to re-trigger it.

---

*Atharva Godkar — Trinity College Dublin, 2026*
