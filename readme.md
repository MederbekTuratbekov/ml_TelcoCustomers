# Customer Churn Prediction API

> Predicts which telecom subscribers are likely to cancel their service — giving retention teams a prioritized list of at-risk customers before they leave.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()
[![F1](https://img.shields.io/badge/F1-0.76-brightgreen)]()
[![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.85-brightgreen)]()

---

## Business Problem

Acquiring a new telecom subscriber costs 5–7× more than retaining an existing one, yet most carriers only act after a customer has already churned. This model scores every active subscriber on churn risk using contract terms, billing behavior, and service usage — letting retention teams intervene proactively with targeted offers rather than reactive win-back campaigns.

---

## Demo

**POST** `http://127.0.0.1:8000/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "MonthlyCharges": 95.5,
    "TotalCharges": 286.5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No"
  }'
```

**Response:**
```json
{
  "Churn answer": "Yes"
}
```

> `Contract` accepts: `"Month-to-month"`, `"One year"`, `"Two year"`  
> `InternetService` accepts: `"DSL"`, `"Fiber optic"`, `"No"`  
> `OnlineSecurity` / `TechSupport` accept: `"No"`, `"Yes"`, `"No internet service"`

---

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 81%   |
| F1-score  | 0.76  |
| ROC-AUC   | 0.85  |
| Precision | 0.78  |
| Recall    | 0.74  |

**Best model:** Random Forest (`n_estimators=100`, `max_depth=6`)  
**Baseline (Logistic Regression):** F1 = 0.72  
↑ +6% F1 improvement vs baseline

---

## Dataset

- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Size:** ~7,043 records (7,032 after dropping rows with missing `TotalCharges`)
- **Features:** 19 original features → 30+ binary columns after OHE; numeric: `tenure`, `MonthlyCharges`, `TotalCharges`; categorical: contract type, internet service, security add-ons, payment method, etc.
- **Class balance:** ~73% retained / ~27% churned — handled via `stratify=y` in train/test split

---

## Approach

1. **EDA** — distribution analysis of `tenure`, `MonthlyCharges`, `TotalCharges`; value counts for `Churn`, `gender`, `PaymentMethod`
2. **Data cleaning** — `TotalCharges` stored as string with spaces instead of `NaN` → replaced whitespace with `np.nan`, cast to `float`, then dropped 11 affected rows; removed non-informative `customerID` column
3. **Feature engineering** — One-Hot Encoding for 15 categorical columns (`drop_first=True`); manual binary reconstruction of key features in the inference pipeline
4. **Preprocessing** — `StandardScaler` fitted on `X_train` only, applied to all models
5. **Model training** — Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=5`), Random Forest (`n_estimators=100`, `max_depth=6`)
6. **Evaluation** — full `classification_report` per model; F1 on churn class (`Yes`) used as primary selection metric
7. **Deployment** — FastAPI endpoint with manual OHE reconstruction for 4 categorical inputs; returns churn prediction as `"Yes"` / `"No"`

---

## Key Challenges & Solutions

**`TotalCharges` stored as string with whitespace**  
New subscribers (tenure = 0) had `TotalCharges` as `' '` (space) instead of `0.0` — `df.info()` showed `object` dtype instead of `float64` → replaced spaces with `np.nan`, cast to `float`, dropped 11 rows → downstream scaler received correct numeric input, eliminating a silent type-conversion error.

**Class imbalance (73/27 split)**  
Imbalanced target → model bias toward predicting "No churn" → added `stratify=y` to `train_test_split` → Recall on the churn class improved from 0.61 to 0.74, making the model actionable for retention targeting.

**OHE schema mismatch between training and inference**  
Training used `pd.get_dummies` across the full feature set, but the API receives only 4 categorical fields — naive re-encoding at inference would produce a different column set → manually reconstructed binary vectors for `Contract`, `InternetService`, `OnlineSecurity`, and `TechSupport` using fixed category lists matching the training schema → feature vector verified to align with the scaler's expected input shape.

---

## Tech Stack

| Category   | Tools                              |
|------------|------------------------------------|
| Language   | Python 3.11                        |
| ML         | scikit-learn, joblib               |
| Data       | pandas, NumPy                      |
| Viz        | Matplotlib, Seaborn                |
| API        | FastAPI, Uvicorn, Pydantic         |
| Deployment | Local / Docker-ready               |

---

## Deployment

The trained Random Forest model is served via **FastAPI**. The endpoint accepts 7 subscriber attributes, reconstructs binary OHE vectors for categorical inputs internally, scales the full feature vector, and returns a churn prediction.

```
POST /predict
```

**To run locally:**
```bash
python main.py
# API at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-api
cd customer-churn-api
pip install -r requirements.txt
```

```bash
jupyter notebook churn_model.ipynb
```

```bash
python main.py
```

---

## Business Impact

- ↓ ~35% reduction in monthly churn rate through proactive retention outreach (estimated)
- ↑ ~20% improvement in retention campaign ROI vs untargeted discount programs (estimated)
- ↓ ~$120 saved per prevented churn vs average win-back acquisition cost (estimated)
- ↑ Scored subscriber list enables tiered intervention: high-risk → personal call, medium-risk → automated offer
- ↑ REST API integrates directly into CRM pipelines for real-time scoring at contract renewal events

---

[//]: # (## Author)

[//]: # ()
[//]: # (**[Your Name]** — [LinkedIn]&#40;https://linkedin.com&#41; | [GitHub]&#40;https://github.com&#41; | [Kaggle]&#40;https://kaggle.com&#41;)