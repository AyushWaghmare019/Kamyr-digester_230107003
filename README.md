<img width="359" height="500" alt="image" src="https://github.com/user-attachments/assets/a17c082c-ab28-4015-96a1-a03bc29395d5" /># 🧪 Kappa Number Prediction in a Continuous Kamyr Digester

### CL653 Final Project — Chemical Engineering × Machine Learning

---

## 📌 Overview

In the kraft pulping process, the **Kappa number** indicates the amount of residual lignin in pulp and directly impacts bleaching cost, energy consumption, and product quality.

In practice, Kappa is measured offline using laboratory titration every 1–4 hours. This delay limits process control, as operators must rely on outdated information.

This project develops a **data-driven soft sensor** that predicts the Kappa number in near real time using routinely measured process variables from a continuous Kamyr digester.

---

## 🎯 Objectives

- Predict Kappa number from process measurements  
- Build a robust pipeline for **industrial sensor data**  
- Incorporate **chemical engineering principles** into modelling  
- Deploy an interactive **Streamlit dashboard** for real-time use  

---

## ⚙️ Engineering Context

The problem combines multiple core areas of chemical engineering:

- **Reaction Engineering** → Delignification kinetics  
- **Mass Transfer** → Chemical penetration into wood chips  
- **Transport Phenomena** → Heat–mass coupling  
- **Process Control** → Real-time quality monitoring  

---

## 📊 Dataset

- Source: https://openmv.net/info/kamyr-digester  
- 301 hourly observations × 22 variables  
- Includes **time-lagged variables** aligned with residence time  
- Target variable: **Kappa number**  

---

## 🧠 Methodology

### Data Preprocessing
- Removed columns with >45% missing values  
- Median imputation for remaining data  
- Duplicate removal  

### Outlier Detection
- Hampel filter (robust univariate)  
- IQR rule (cross-check)  
- Isolation Forest (multivariate anomalies)  

### Feature Engineering
- H-factor proxy (temperature severity)  
- Active alkali (AA) charge rate  

### Scaling
- StandardScaler (fit on training data only)  

### Dimensionality Reduction
- PCA (handles multicollinearity)  
- t-SNE (visualizes nonlinear structure)  

---

## 🤖 Models Evaluated

- Linear Regression  
- Partial Least Squares (PLS)  
- Support Vector Regression (SVR)  
- Random Forest  
- Gradient Boosting  

Hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**.

---

## 📈 Results

| Model             | R²   | RMSE |
|------------------|------|------|
| **SVR (Best)**   | 0.63 | 1.79 |
| Gradient Boosting| 0.60 | 1.85 |
| Random Forest    | 0.56 | 1.95 |
| Linear Regression| 0.48 | 2.12 |
| PLS              | 0.46 | 2.15 |

The best model (SVR) achieves performance comparable to the **intrinsic uncertainty of laboratory Kappa measurement (±1–1.5 units)**.

---

## 🔍 Key Insights

Important variables influencing Kappa:

- White liquor flow  
- Steam flow and heat input  
- Chip feed rate  
- Alkali concentration  

These results align with **first-principles understanding of kraft pulping**.

---

## 🚀 Deployment

The model is deployed using **Streamlit**, providing:

- Real-time prediction  
- What-if analysis  
- Model comparison  
- Feature importance visualization  

This functions as an **inferential soft sensor** for process monitoring.

---

## 📁 Project Structure


<img width="359" height="500" alt="image" src="https://github.com/user-attachments/assets/ef73e59a-6ed0-49c8-a341-041f5f550e1a" />


---

## 🧪 How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Run the training pipeline
```
        python kamyr_pipeline.py
```
3. Launch the Streamlit dashboard
```
   python -m streamlit run app.py
```


💡 Applications
Real-time process monitoring
Reduced dependence on lab measurements
Improved control of pulp quality
Lower chemical and energy costs

Author:-
Aayush Waghmare
Chemical Engineering | AI/ML in Process Systems

⭐ Final Note

This project shows how combining chemical engineering fundamentals with machine learning can lead to practical, deployable solutions such as real-time soft sensors for industrial processes.
