# Customer Churn Prediction — Telco Dataset

A complete end-to-end machine learning project that predicts customer churn for a telecom company using the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The project covers business understanding, exploratory data analysis, feature engineering, model training, threshold tuning, and deployment-ready model export.

---

## Business Context

Customer churn occurs when a customer stops using a company's service. For telecom companies, acquiring a new customer costs **5–7× more** than retaining an existing one. This project identifies which customers are likely to churn and why — so the business can intervene early with targeted retention strategies.

**Target variable:** `Churn` (Yes / No)

---

## Dataset

- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers &nbsp;|&nbsp; **Columns:** 21 features
- **Features include:** demographics (gender, SeniorCitizen, Partner, Dependents), subscribed services (internet, streaming, security, backup), account info (contract type, payment method, tenure, charges), and churn status

---

## Notebook 1 — Exploratory Data Analysis

### Key steps

- Data type correction (`TotalCharges` stored as object → converted to float)
- Removed 11 rows where `tenure == 0` (new customers with no activity)
- Feature engineering: `tenure_group` (New / Mid / Loyal) from raw tenure
- Chi-square test on all 15 categorical features to statistically validate EDA findings
- Correlation heatmap for numerical features

### Top findings

| Driver | Churn Rate |
|---|---|
| Month-to-month contract | 42.7% |
| Fiber optic internet | 41.9% |
| No online security | 41.8% |
| No tech support | 41.6% |
| Electronic check payment | 45.3% |

- **First 12 months are critical:** ~48% of new customers churn vs only ~8% of loyal (49–72 month) customers
- `gender` and `PhoneService` are statistically insignificant (chi-square p > 0.05) and excluded from modelling
- `tenure` and `TotalCharges` are strongly correlated (r = 0.83) — handled during preprocessing

---

## Notebook 2 — Model Training & Evaluation

### Preprocessing

- Dropped `customerID`, `gender`, `PhoneService`
- Collapsed `"No internet service"` / `"No phone service"` → `"No"` across service columns
- Ordinal encoded `tenure_group` (New=0, Mid=1, Loyal=2)
- Binary mapped all Yes/No columns to 1/0
- One-hot encoded `InternetService`, `Contract`, `PaymentMethod` with `drop_first=True`
- Feature scaling applied **only** to Logistic Regression (tree models are scale-invariant)
- Scaler fit on train set only — applied to test set to prevent data leakage

### Models trained

| Model | Test AUC-ROC | F1 (Churn) | Overfit Gap |
|---|---|---|---|
| Logistic Regression | **0.835** | 0.62 | ~0.00 |
| Random Forest (baseline) | 0.817 | 0.57 | ~0.18 |
| XGBoost | 0.823 | 0.59 | ~0.06 |
| Random Forest (tuned) | 0.826 | 0.58 | ~0.07 |

- Class imbalance handled via `class_weight='balanced'` (LR, RF) and `scale_pos_weight` (XGBoost)
- 5-fold stratified cross-validation used for reliable AUC estimates
- **Logistic Regression** achieved the best test AUC with near-zero overfitting

### Threshold tuning

Default threshold (0.5) optimises accuracy — not useful for imbalanced churn data. The precision-recall curve was used to find the F1-maximising threshold, increasing recall on churners (catching more at-risk customers).

### Business cost matrix

| Model | Total Cost | False Negatives | False Positives |
|---|---|---|---|
| LR default (0.5) | — | higher | lower |
| LR tuned threshold | lower | lower | higher |

Assumptions: missed churner (FN) = $500 cost, false alarm (FP) = $50 cost.

### Best model for deployment

**Logistic Regression with tuned threshold** — highest AUC, no overfitting, interpretable for business stakeholders.

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Krishna-Rai-Kishan/CustomerChurnPrediction
cd telco-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download `Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root.

**4. Run the notebooks in order**
```
01_EDA_Telco_Churn.ipynb
02_Model_Training_Telco_Churn.ipynb
```

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=1.7.0
scipy>=1.10.0
joblib>=1.3.0
```

Python 3.9+ recommended.

---

## Results Summary

- Logistic Regression outperformed ensemble models on this dataset — clean linear signal in the features (contract type, tenure, service subscriptions) is sufficient
- Threshold tuning improved business value over the default 0.5 cutoff
- Key retention levers identified: contract upgrades, first-year engagement, bundling support services, nudging customers toward auto-pay

---

## Author

**[Krishna Rai]**  
[LinkedIn](https://linkedin.com/in/thekrishnaraiji)