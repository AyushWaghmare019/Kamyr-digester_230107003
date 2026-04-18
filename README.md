# 🧪 Kappa Number Prediction in a Continuous Kamyr Digester

### CL653 Final Project — Chemical Engineering + Machine Learning

**Kappa Soft Sensor for Kraft Pulping Process**

---

## 📌 Overview

This project develops a **machine-learning-based soft sensor** to predict the **Kappa number** — a key indicator of residual lignin in pulp — in a continuous **Kamyr digester** used in the kraft pulping process.

Traditionally, Kappa number is measured offline using laboratory titration, introducing delays of 1–4 hours. This project replaces that delay with a **real-time predictive model**, enabling faster and more efficient process control.

---

## 🎯 Objective

* Predict Kappa number using process variables
* Build a **robust ML pipeline** for industrial data
* Integrate **chemical engineering principles** (reaction kinetics, transport phenomena)
* Deploy as an interactive **Streamlit dashboard (soft sensor)**

---

## ⚙️ Core Chemical Engineering Concepts

* **Reaction Engineering** → Delignification kinetics
* **Mass Transfer** → Chemical penetration into wood chips
* **Transport Phenomena** → Heat + mass coupling
* **Process Control** → Real-time quality monitoring

---

## 📊 Dataset

* Source: https://openmv.net/info/kamyr-digester
* Observations: **301 rows × 22 variables**
* Includes **time-lagged variables** (residence time effect)
* Target: **Kappa number (lignin content)**

---

## 🧠 Methodology

### 🔹 Data Preprocessing

* Drop columns with >45% missing values
* Median imputation (robust to industrial noise)
* Duplicate removal

### 🔹 Outlier Detection (Multi-layer)

* Hampel Filter (robust univariate)
* IQR method (cross-check)
* Isolation Forest (multivariate anomalies)

### 🔹 Feature Engineering

* H-factor proxy (temperature-based kinetics)
* AA charge rate (chemical reaction driver)

### 🔹 Feature Scaling

* StandardScaler (required for ML models)

---

### 🔹 Dimensionality Reduction

* PCA → handles multicollinearity
* t-SNE → nonlinear structure visualization

---

### 🔹 Models Trained

| Model                           | Type              |
| ------------------------------- | ----------------- |
| Linear Regression               | Baseline          |
| Partial Least Squares (PLS)     | Chemometric       |
| Support Vector Regression (SVR) | Nonlinear         |
| Random Forest                   | Ensemble          |
| Gradient Boosting               | Advanced Ensemble |

---

### 🔹 Hyperparameter Tuning

* GridSearchCV
* 5-fold cross-validation

---

## 📈 Results

| Model             | R²       | RMSE     |
| ----------------- | -------- | -------- |
| **SVR (Best)**    | **0.63** | **1.79** |
| Gradient Boosting | 0.60     | 1.85     |
| Random Forest     | 0.56     | 1.95     |
| Linear Regression | 0.48     | 2.12     |
| PLS               | 0.46     | 2.15     |

👉 SVR achieved performance comparable to **lab measurement uncertainty**

---

## 🔍 Key Insights

Top variables affecting Kappa:

* White liquor flow
* Steam flow & heat
* Chip rate
* Alkali concentration

👉 Matches **first-principles chemical engineering theory**

---

## 🚀 Deployment

* Built using **Streamlit**
* Features:

  * Real-time prediction
  * What-if analysis
  * Model comparison
  * Feature importance visualization

👉 Acts as an **inferential soft sensor**

---

## 📁 Project Structure

```
kamyr-digester/
│
├── kamyr_pipeline.py        # Training pipeline
├── streamlit_app.py         # Deployment app
├── kappa_model.pkl          # Trained model
├── data/
│   └── kamyr_digester.csv
├── outputs/                 # Plots & results
├── artifacts/               # Saved models & scaler
└── README.md
```

---

## 🧪 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit app

```bash
python -m streamlit run app.py
```

---

## 💡 Applications

* Real-time process monitoring
* Reduced lab dependency
* Improved pulp quality control
* Cost optimization in bleaching

---

## ⚠️ Limitations

* Small dataset (~300 samples)
* Model drift over time
* Not suitable for startup/shutdown conditions

---

## 🔮 Future Work

* Hybrid physics + ML models
* Online retraining (sliding window)
* Uncertainty quantification
* Integration with Digital Twin

---

## 📚 References

* Dayal et al. (1994) — Kamyr Digester Modeling
* OpenMV Dataset — Kevin Dunn
* TAPPI T236 Standard (Kappa measurement)
* Gustafson et al. (Kraft kinetics model)

---

## 👨‍💻 Author

**Aayush Waghmare**
Chemical Engineering | AI/ML in Process Systems

---

## 🌐 Links

* Dataset: https://openmv.net/info/kamyr-digester
* (Add your Streamlit deployment link here)

---

## ⭐ Final Note

This project demonstrates how **Chemical Engineering + Machine Learning** can be combined to build real-world industrial solutions like **soft sensors for process optimization**.
