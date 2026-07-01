# Credit Card Fraud Detection using SMOTE & Random Forest

An end-to-end machine learning pipeline built to detect fraudulent credit card transactions. This project tackles severe class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) and optimizes a Random Forest Classifier to achieve high precision and recall.

## 📌 Project Overview
Credit card fraud datasets are notoriously imbalanced. In this dataset, fraudulent transactions account for only **0.1727%** of the total data. A naive model would yield high accuracy but fail to catch actual fraud. This project focuses on optimizing the **F1-Score** for the minority class to ensure fraudulent activities are caught accurately without triggering excessive false alarms.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Environment:** Google Colab / Jupyter Notebook
- **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (SMOTE), `joblib`, `matplotlib`

## 📊 Pipeline & Workflow

1. **Data Ingestion:** Automated dataset download using `kagglehub` from the MLG-ULB Kaggle dataset.
2. **Exploratory Data Analysis:** Evaluated the severe class imbalance ($284,315$ normal vs. $492$ fraud transactions).
3. **Baseline Modeling:** Trained a balanced Logistic Regression model as a performance benchmark.
4. **Advanced Modeling:** Evaluated a Random Forest Classifier (`class_weight='balanced'`).
5. **Resampling (SMOTE):** Oversampled the minority class in the training set to create a perfectly balanced $50/50$ distribution for training.
6. **Threshold Optimization:** Plotted the Precision-Recall curve and fine-tuned the classification threshold to **0.7** to maximize the fraud class F1-Score.
7. **Model Export:** Serialized the final optimized model into `fraud_model.pkl` using `joblib`.

## 📈 Final Model Performance (Test Set)

By tuning the decision threshold to **0.7** on the SMOTE-trained Random Forest model, the following metrics were achieved for the fraud class (`Class 1`):

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Precision** | `0.93` | 93% of flagged transactions are genuinely fraudulent. |
| **Recall** | `0.80` | The model successfully captures 80% of all actual fraud cases. |
| **F1-Score** | `0.86` | A robust, balanced metric for highly skewed data. |
| **ROC-AUC** | `0.96` | Strong overall discriminative ability. |

## 📁 Repository Structure
```text
├── CreditcardFraudDetection.ipynb   # Main Google Colab / Jupyter Notebook
├── fraud_model.pkl                  # Serialized trained model ready for deployment
└── README.md                        # Project documentation
