# 📊 QMTRY — Loan Default Risk Model

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## 📌 Project Overview
This project applies **deep learning techniques** to predict whether a borrower will **default** on a loan using Lending Club’s historical dataset (2007–2015).

Why it matters:
- Reduces **loss exposure** by flagging high-risk borrowers  
- Improves **credit decisioning**  
- Mirrors **healthcare revenue cycle** risk prediction (default ≈ claim denial/non-payment)

The dataset is **highly imbalanced** (most borrowers repay), making it ideal to demonstrate practical handling of **imbalanced, high-stakes financial/healthcare data**.

---

## 🎯 Objectives
- Build a **deep learning classifier** for default risk  
- Apply **class-imbalance strategies** (oversampling, undersampling, SMOTE)  
- Engineer features for **interpretability & auditability**  
- Evaluate beyond accuracy (focus on **recall** & **ROC-AUC**)  
- Produce a **generalizable pipeline** transferable to denial prediction

---

## 🏦 Domain Context
- **Finance** → loan risk modeling, underwriting  
- **Healthcare Parallel** → denial prediction, payment risk  
- **Audit-Ready** → reproducible scripts, tracked metrics, explainability

---

## 📊 Dataset Description
| Feature            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `credit.policy`    | 1 if customer meets Lending Club credit criteria, 0 otherwise               |
| `purpose`          | Loan purpose (credit_card, debt_consolidation, educational, etc.)           |
| `int.rate`         | Interest rate (e.g., 0.11 for 11%)                                          |
| `installment`      | Monthly loan installment amount                                             |
| `log.annual.inc`   | Log of self-reported annual income                                          |
| `dti`              | Debt-to-income ratio                                                        |
| `fico`             | Borrower FICO score                                                         |
| `days.with.cr.line`| Number of days borrower has had a credit line                               |
| `revol.bal`        | Revolving balance                                                           |
| `revol.util`       | Revolving utilization rate                                                  |
| `inq.last.6mths`   | Number of creditor inquiries in last 6 months                               |
| `delinq.2yrs`      | 30+ day delinquencies in past 2 years                                       |
| `pub.rec`          | Derogatory public records                                                    |
| `not.fully.paid`   | **Target** → 1 if loan not fully paid (default), 0 if repaid                |

---

## ⚙️ Methodology
1. **Ingest & Inspect** → types, nulls, target ratio  
2. **Clean** → impute/trim, encode categorical (`purpose`)  
3. **EDA** → distributions, correlations, imbalance visualizations  
4. **Balance** → oversample/undersample/SMOTE (imbalanced-learn)  
5. **Engineer** → scaling numeric, one-hot categorical  
6. **Model** → Keras/TensorFlow DNN for tabular data  
7. **Evaluate** → Recall, ROC-AUC, confusion matrix, PR curve  
8. **Explain** → SHAP or permutation importance  
9. **Package** → save artifacts, seed for reproducibility

---

## 🚀 Tech Stack
- **Python 3.9+**  
- **Pandas, NumPy** — preprocessing  
- **Matplotlib, Seaborn** — visualization  
- **scikit-learn** — transforms, metrics, model utils  
- **imbalanced-learn** — SMOTE & resampling  
- **TensorFlow/Keras** — deep learning

---

## 📈 Key Metrics
- **Recall (Sensitivity)** — priority: catch defaults  
- **ROC-AUC** — overall separability  
- **Precision, F1** — balance false positives/negatives  
- **Confusion Matrix, PR Curve** — error analysis

---

## 📂 Repository Structure
