# Customer Churn Prediction — Telco Dataset

An end-to-end churn analysis and prediction project that predicts customer churn for a telecom company using the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It covers data cleaning, exploratory analysis with statistical testing, feature engineering, model comparison (Logistic Regression, Random Forest, XGBoost), hyperparameter tuning, threshold tuning, and a business cost evaluation.

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn, XGBoost, Jupyter Notebook

---

## Business Context

Churn is when a customer stops using a company's service. Retaining an existing telecom customer is generally cheaper than acquiring a new one, so the goal here is to find **which customers are likely to churn and why**, so that retention effort can be targeted.

**Target variable:** `Churn` (Yes / No)

---

## Dataset

| Item | Value |
|---|---|
| Source | [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Raw size | 7,043 rows, 21 columns |
| After cleaning | **7,032 rows** (11 rows with `tenure = 0` removed) |
| Churn rate (cleaned) | **26.6%** (1,869 churned / 5,163 retained) |
| Feature groups | Demographics, subscribed services, account details (contract, billing, payment), charges, tenure |

The class split is moderately imbalanced (about 73% / 27%), so accuracy alone is misleading. A model that always predicts "No churn" would be about 73% accurate and catch no churners. Models are compared on **ROC-AUC, F1 and recall for the churn class**.

---

## Repository Structure

```
CustomerChurnPrediction/
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_EDA_Telco_Churn.ipynb
│   └── 02_Model_Training_Telco_Churn.ipynb
├── requirements.txt
└── README.md
```

---

## Notebook 1: Exploratory Data Analysis

### Data cleaning

- Dropped `customerID` (identifier, no predictive value).
- Converted `TotalCharges` from text to numeric. This exposed **11 missing values**, all belonging to customers with `tenure = 0` (brand-new accounts with no billing history). These 11 rows were removed.
- After dropping `customerID`, 22 rows appear identical. They are new customers (tenure of 1 month) with the same plan and charges, not true duplicates, so they were kept.
- Outlier check with box plots: no outlier removal was needed. `TotalCharges` is right-skewed because long-tenure customers accumulate charges.

### Feature engineering

`tenure_group`, created from `tenure`:

| Group | Tenure | Customers | Churn rate |
|---|---|---:|---:|
| New | 0 to 12 months | 2,175 | **47.7%** |
| Mid | 13 to 48 months | 2,618 | 23.6% |
| Loyal | 49 to 72 months | 2,239 | **9.5%** |

### Churn rate by category

| Driver | Churn rate | Comparison |
|---|---:|---|
| Month-to-month contract | **42.7%** | One-year 11.3%, two-year 2.8% |
| Electronic check payment | **45.3%** | Bank transfer 16.7%, credit card 15.3%, mailed check 19.2% |
| Fiber optic internet | 41.9% | DSL 19.0%, no internet 7.4% |
| No online security | 41.8% | With online security 14.6% |
| No tech support | 41.6% | With tech support 15.2% |
| No online backup | 39.9% | With online backup 21.6% |
| No partner | 33.0% | With partner 19.7% |
| No dependents | 31.3% | With dependents 15.5% |
| Paperless billing | 33.6% | Without 16.4% |

### Statistical validation (Chi-square test of independence)

Chi-square tests were run for all **15 categorical features** against `Churn`.

- **13 of 15** features have a statistically significant relationship with churn (p < 0.05).
- `Contract` has the largest statistic (χ² = 1,179.55), followed by `OnlineSecurity` (846.68), `TechSupport` (824.93) and `InternetService` (728.70).
- `MultipleLines` is significant but weak (χ² = 11.27, p = 0.0036).
- `gender` (p = 0.49) and `PhoneService` (p = 0.35) are **not significant** and were excluded from modelling.

### Correlation (numerical features)

| Pair | Correlation |
|---|---:|
| Tenure vs Churn | -0.35 |
| TotalCharges vs Churn | -0.20 |
| MonthlyCharges vs Churn | 0.19 |
| SeniorCitizen vs Churn | 0.15 |
| Tenure vs TotalCharges | **0.83** |
| MonthlyCharges vs TotalCharges | 0.65 |

Tenure and TotalCharges are strongly correlated, so `TotalCharges` is dropped for Logistic Regression in the modelling notebook.

---

## Notebook 2: Model Training and Evaluation

### Preprocessing

- Dropped `customerID`, `gender` and `PhoneService`.
- Collapsed "No internet service" and "No phone service" into "No" across the service columns.
- Ordinal-encoded `tenure_group` (New = 0, Mid = 1, Loyal = 2).
- Mapped Yes/No columns to 1/0.
- One-hot encoded `InternetService`, `Contract` and `PaymentMethod` with `drop_first=True`.
- Stratified 80/20 train-test split (`random_state=42`): **5,625 train / 1,407 test**, 22 features.
- `StandardScaler` applied only to Logistic Regression, **fit on the training set only** to avoid data leakage. Tree models use unscaled data.
- Class imbalance handled with `class_weight='balanced'` (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost).

### Baseline model comparison (test set)

| Model | Test AUC | F1 (Churn) | Recall (Churn) | Train AUC | Overfit gap | 5-fold CV AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.833 | 0.606 | 0.781 | 0.849 | 0.016 | 0.845 ± 0.005 |
| Random Forest (default) | 0.818 | 0.539 | 0.484 | 1.000 | 0.182 | 0.825 ± 0.007 |
| XGBoost (default) | 0.809 | 0.594 | 0.693 | 0.984 | 0.175 | 0.825 ± 0.005 |

The default Random Forest overfits heavily (train AUC about 1.00 vs test 0.818).

### Hyperparameter tuning (Random Forest)

`RandomizedSearchCV` with **30 parameter configurations** and 5-fold cross-validation, scored on ROC-AUC.

- **Best parameters:** `n_estimators=300`, `max_depth=10`, `min_samples_leaf=4`, `max_features='sqrt'`
- **Best cross-validated AUC:** 0.8455

| Random Forest | Test AUC | Train AUC | Overfit gap |
|---|---:|---:|---:|
| Default | 0.818 | 1.000 | 0.182 |
| **Tuned** | **0.838** | 0.920 | **0.082** |

Tuning improved test AUC by about 0.02 and **cut the overfit gap by more than half (0.182 to 0.082)**.

### Final model comparison

| Model | Test AUC | F1 (Churn) | Recall (Churn) | Overfit gap |
|---|---:|---:|---:|---:|
| Logistic Regression (threshold 0.5) | 0.833 | 0.606 | 0.781 | 0.016 |
| Logistic Regression (threshold 0.622) | 0.833 | 0.627 | 0.701 | 0.016 |
| Random Forest (default) | 0.818 | 0.539 | 0.484 | 0.182 |
| **Random Forest (tuned)** | **0.838** | 0.627 | 0.759 | 0.082 |
| XGBoost (default) | 0.809 | 0.594 | 0.693 | 0.175 |

**Reading the results:** the tuned Random Forest has the highest test AUC, but Logistic Regression is within 0.005 AUC and its cross-validated AUC is essentially tied (0.845 vs 0.8455). Logistic Regression has the smallest overfit gap and is the most interpretable, so the two are close in practice. XGBoost was run with default parameters and was not tuned.

### Threshold tuning (Logistic Regression)

The decision threshold was moved from 0.5 to **0.622**, the value that maximises F1 for the churn class on the precision-recall curve.

| Metric (Churn) | Threshold 0.50 | Threshold 0.622 |
|---|---:|---:|
| Precision | 0.495 | 0.567 |
| Recall | 0.781 | 0.701 |
| F1 | 0.606 | 0.627 |

Raising the threshold improved precision and F1 but **reduced recall**, meaning fewer churners are caught.

### Business cost evaluation

Illustrative assumptions: a missed churner (false negative) costs **$500** and an unnecessary retention offer (false positive) costs **$50**.

| Model | Total cost | False positives | False negatives |
|---|---:|---:|---:|
| **Logistic Regression (threshold 0.5)** | **$55,900** | 298 | 82 |
| Random Forest (tuned) | $57,400 | 248 | 90 |
| Logistic Regression (threshold 0.622) | $66,000 | 200 | 112 |
| XGBoost (default) | $69,450 | 239 | 115 |

Because a missed churner is assumed to cost 10 times a false alarm, the F1-optimal threshold (0.622) is **not** the cheapest option: it lowers false positives but raises false negatives, increasing total cost versus the default 0.5. The cost figures use assumed values and are for comparison only.

### Best model for deployment

**Logistic Regression with tuned threshold** — highest AUC, no overfitting, interpretable for business stakeholders.

---

## Key Findings and Retention Ideas

These are statistical associations in historical data, not proof of cause.

1. **Contract type is the strongest signal.** Month-to-month customers churn at 42.7% vs 2.8% on two-year plans. Incentives for moving to longer contracts are worth testing.
2. **The first year is the riskiest period.** Customers with 12 months or less churn at 47.7% vs 9.5% for those with 49+ months. Onboarding and early-engagement offers are the natural focus.
3. **Fiber optic customers churn at more than double the DSL rate** (41.9% vs 19.0%), which may point to price or service-quality issues worth investigating.
4. **Customers without add-on support services churn far more** (online security 41.8%, tech support 41.6%). Bundling these services is a candidate retention lever.
5. **Electronic check users churn most (45.3%).** Moving customers to automatic payment methods (15 to 17% churn) could help.
6. **Gender and phone service are not related to churn** (chi-square p = 0.49 and 0.35).

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
**Before running the notebooks please move the dataset in the notebooks directory**
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

## Author

**[Krishna Rai]**  
[LinkedIn](https://linkedin.com/in/thekrishnaraiji)