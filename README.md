# 📊 Google Analytics Customer Revenue Prediction

A machine learning dashboard that analyses Google Merchandise Store visitor data 
to predict revenue and identify high-value customers.

🔗 **Live Demo**: [Click here to view the dashboard](https://quantumcoder-18-ga-revenue-prediction.streamlit.app)

---

## 📌 What this project does

Most website visitors never buy anything — only about 1% actually make a purchase.
This project answers the question: **which visitors are likely to spend money, and how much?**

It analyses 100,000 real visitor sessions from the Google Merchandise Store and builds
a machine learning model to predict revenue per visitor.

---

## 📷 Dashboard Pages

| Page | Description |
|---|---|
| Overview | Total revenue, visits, conversion rate, revenue trend chart |
| Traffic Sources | Where visitors come from — Google, YouTube, Direct, Social |
| Device Analytics | Desktop vs mobile vs tablet, top browsers and operating systems |
| Revenue Prediction | Enter visitor details and get a live revenue prediction |
| Customer Segments | High / medium / low value visitors based on the 80-20 rule |

---

## 🧠 How it works
Raw Data → Clean & Prepare → Train LightGBM Model → Streamlit Dashboard
1. **Data exploration** — loaded 100,000 rows of Google Analytics data
2. **Data cleaning** — fixed missing values, converted text to numbers
3. **Feature engineering** — extracted day, month, weekday from dates
4. **Model training** — trained a LightGBM model to predict revenue
5. **Dashboard** — built an interactive Streamlit app with 5 pages

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Algorithm | LightGBM |
| RMSE | 0.3497 |
| R² Score | 0.2459 |
| Training rows | 80,000 |
| Test rows | 20,000 |

**Top features the model found most useful:**
- `totals.hits` — more interactions = more likely to buy
- `totals.pageviews` — more pages viewed = higher intent
- `geoNetwork.city` — location is a strong revenue signal
- `month` — seasonality affects purchasing behaviour

---

## 🛠️ Tech stack

| Category | Tool |
|---|---|
| Language | Python 3.12 |
| Dashboard | Streamlit |
| ML Model | LightGBM |
| Data processing | Pandas, NumPy |
| Visualisation | Plotly |
| Model saving | Joblib |
| Dataset | Google Analytics — Kaggle |

---

## 🚀 Run it locally
```bash
# 1. Clone the repository
git clone https://github.com/QuantumCoder-18/ga-revenue-prediction.git
cd ga-revenue-prediction

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data and train model
python prepare_data.py
python train_model.py

# 5. Launch dashboard
streamlit run main.py
```

---

## 📂 Project structure
ga-revenue-prediction/
├── main.py              ← Streamlit dashboard (5 pages)
├── prepare_data.py      ← Data cleaning and feature engineering
├── train_model.py       ← LightGBM model training
├── explore_data.py      ← Data exploration and analysis
├── requirements.txt     ← Python dependencies
├── models/
│   └── lgbm_model.pkl   ← Saved trained model
└── data/
└── clean_data.csv   ← Processed dataset (auto-generated)
---

## 💡 Key findings

- **99% of visitors spend ₹0** — only 1% make a purchase
- **United States** generates the most visits and revenue
- **Desktop** dominates at 68.9% of all visits
- **Google and YouTube** are the top traffic sources
- **Page views and hits** are the strongest predictors of purchase intent

---

## 👤 About

Built by **Aruthra** as a data science portfolio project.

- GitHub: [@QuantumCoder-18](https://github.com/QuantumCoder-18)
- Dataset: [Kaggle — GA Customer Revenue Prediction](https://www.kaggle.com/c/ga-customer-revenue-prediction)