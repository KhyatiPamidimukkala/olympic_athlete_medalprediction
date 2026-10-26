# 🏅 Olympic Medal Data Analysis Using Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aditipaulolympicmedalanalysis-t3qvjwlsmzfkzxhnee4avm.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🚀 **Live Dashboard**: Experience the full interactive web application directly in your browser at **[aditipaulolympicmedalanalysis-t3qvjwlsmzfkzxhnee4avm.streamlit.app](https://aditipaulolympicmedalanalysis-t3qvjwlsmzfkzxhnee4avm.streamlit.app/)**

An end-to-end Machine Learning research project and interactive web dashboard analyzing **120+ years of modern Olympic Games data (1896–2016)** across **271,116 athlete records**.

This project implements missing data imputation, anti-leakage feature scaling, simple and multiple linear regression, supervised classification under class imbalance, morphological clustering (K-Means & GMM), ensemble learning, cross-validation, hyperparameter tuning, and a live AI prediction interface.

---

## ⚡ Quick Start

### 🌐 1. Live Deployment (Instant Access)
No setup required! Open the live web app:
👉 **[Olympic Medal Analysis - ML Dashboard](https://aditipaulolympicmedalanalysis-t3qvjwlsmzfkzxhnee4avm.streamlit.app/)**

### 💻 2. Local Setup

```bash
# 1. Clone repository
git clone https://github.com/Aditipaul17/AditiPaul_olympic_medal_analysis.git
cd AditiPaul_olympic_medal_analysis

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the high-performance interactive web dashboard (http://localhost:8501)
streamlit run app.py

# Or execute the complete standalone ML pipeline via CLI
python main.py
```

---

## 📁 Project Structure

```
├── app.py                          # High-performance Streamlit web dashboard (8 interactive sections)
├── main.py                         # Standalone end-to-end ML pipeline execution script
├── olympic_medal_analysis.ipynb    # Executed master Jupyter Notebook with detailed mathematical derivations
├── requirements.txt                # Project dependencies (pandas, scikit-learn, streamlit, plotly, etc.)
├── README.md                       # Project documentation
├── data/
│   └── olympics.csv                # Historical Olympic Games dataset (271,116 rows x 15 columns)
└── outputs/
    └── figures/                    # 13 Generated high-resolution evaluation plots (PNG)
```

---

## 🖥️ Interactive Web Dashboard (`app.py`)

Access the deployed dashboard live at **[aditipaulolympicmedalanalysis-t3qvjwlsmzfkzxhnee4avm.streamlit.app](https://aditipaulolympicmedalanalysis-t3qvjwlsmzfkzxhnee4avm.streamlit.app/)** or run locally with `streamlit run app.py` (**`http://localhost:8501`**). The dashboard is styled with a sleek Navy Blue & Gold aesthetic and divided into 8 modules:

1. **🔴 Dashboard Overview**: Key Olympic metrics (records, sports, medals, years), medal breakdown charts, top sports participation rankings, and an instant-search dataset explorer with CSV download.
2. **🧹 Missing Value Intelligence & Preprocessing**: Missingness by feature, interactive imputation lab comparing Mean, Median, and Mode histograms before/after filling, and anti-leakage scaling (`StandardScaler` vs. `MinMaxScaler`).
3. **📈 Simple & Multiple Linear Regression**: Interactive height-weight OLS regression line, residual distribution plots, and a multi-predictor laboratory with dynamic coefficient visualization.
4. **🤖 Supervised Classification Studio**: Compare **KNN**, **Gaussian Naive Bayes**, and **Logistic Regression** with probability threshold tuning, confusion matrices, prediction probability histograms, and multi-model ROC curves.
5. **🧩 Morphological Clustering**: Unsupervised grouping of athlete biotypes using **K-Means** and **Gaussian Mixture Models (GMM)** with silhouette score comparisons, 2D scatter projections, and feature heatmaps.
6. **🌲 Ensemble Learning**: Side-by-side performance comparison of **Decision Trees**, **Bagging**, and **Random Forests**, feature importance rankings, and metric heatmaps.
7. **🧪 Cross-Validation & Hyperparameter Tuning**: Stability analysis across folds (**Stratified K-Fold** vs. **K-Fold**), score distributions, and automated parameter optimization (**Grid Search** vs. **Randomized Search**).
8. **🔮 Interactive Athlete Predictor (Live AI)**: Input height, age, sex, season, and sport to obtain live real-time estimates of athlete body weight, BMI, and calibrated medal probability with an interactive confidence gauge.

---

## ⚡ Performance & Optimization Highlights

The interactive dashboard includes several architectural optimizations to ensure instant responsiveness:
- **Lazy Model Initialization**: Heavy model training is scoped strictly to the Live Inference module, eliminating startup freezes and allowing immediate navigation.
- **Resource Hash Bypass**: Utilizes Streamlit's underscore parameter convention (`_df_data`) to prevent redundant hashing of the 36 MB DataFrame on reruns.
- **Vectorized Search Filtering**: Search queries use vectorized NumPy operations across text columns for fast sub-second filtering across 270,000+ rows.
- **Stratified Subsampling for Real-time Interactivity**: CPU-intensive algorithms (KNN, 150-tree ensembles, 5-fold cross-validation) dynamically subsample stratified batches when working with massive cohorts, delivering sub-second execution while preserving statistical accuracy.
- **DOM & Render Caching**: Data table payloads and matplotlib scatter plots are capped for smooth browser scrolling without memory spikes.

---

## 📊 Machine Learning Topics & Results

| Topic | Algorithm / Method | Target / Feature | Key Metric | Key Finding |
|---|---|---|---|---|
| **Missing Data Imputation** | Mean, Median & Mode | Biometrics (`height`, `weight`, `age`) | Variance preservation | Median preserves natural IQR ($70.0\text{ kg}$) without artificial variance spikes. |
| **Preprocessing & Scaling** | `StandardScaler`, `MinMaxScaler` | `age`, `height`, `weight` | Train vs. Test scale | Train $\mu=0.00$, Test $\mu=-0.005$ with strict fit-on-train anti-leakage protocol. |
| **Simple Linear Regression** | Ordinary Least Squares (OLS) | Height $\to$ Weight | $R^2 = 0.635$, $\text{RMSE} = 8.50\text{ kg}$ | Athlete mass increases $+1.08\text{ kg}$ per cm of height. |
| **Multiple Linear Regression** | Multivariate OLS | Height, Age, Sex, Sport $\to$ Weight | $R_{\text{adj}}^2 = 0.708$, $\text{RMSE} = 7.60\text{ kg}$ | Male sex adds $+7.61\text{ kg}$ controlling for height, age, and sport. |
| **K-Nearest Neighbors** | KNN Classifier ($K=5$) | Medal Victory ($0/1$) | $\text{ROC-AUC} = 0.613$, F1 $= 0.218$ | Standardized features required; distance weighting prevents majority-class dominance. |
| **Naive Bayes** | Gaussian Naive Bayes | Medal Victory ($0/1$) | $\text{ROC-AUC} = 0.612$, Fast fit | Fast $\mathcal{O}(Nd)$ baseline; robust despite height-weight covariance. |
| **Logistic Regression** | Cost-Balanced Logistic Reg | Medal Victory ($0/1$) | Recall: $0.1\% \to 58.2\%$, $\text{OR}_{\text{height}} = 1.15$ | Class balancing corrects majority-class bias; height is a positive predictor. |
| **K-Means Clustering** | K-Means ($K=3$) | Morphological Archetypes | Inertia Elbow, Silhouette | Discovers 3 distinct biotypes: Power ($95\text{kg}$), Compact ($57\text{kg}$), Tall ($78\text{kg}$). |
| **Gaussian Mixture Models** | GMM (EM, Full Covariance) | Soft Morphology Clustering | BIC Minimum ($K=4$) | Anisotropic covariance captures natural height-weight elongation. |
| **Ensemble Classifiers** | Bagging & Random Forest | Medal Victory ($0/1$) | $\text{ROC-AUC}: 0.590 \to 0.662$ | Ensemble bagging and feature subsampling reduce classification variance by $>7\%$. |
| **Cross-Validation** | K-Fold vs. Stratified K-Fold | Validation Stability | Score std: $\pm 0.017 \to \pm 0.010$ | Stratification reduces variance by $40\%$ and preserves class balance ($13.58\%$). |
| **Hyperparameter Tuning** | GridSearchCV vs. RandomizedSearch | Random Forest Tuning | Search Time: $13.8\text{s} \to 6.1\text{s}$ | Randomized search finds matching AUC ($0.6056$) in $56\%$ less computation time. |

---

## 📦 Dataset Information

- **Source:** [TidyTuesday Olympic Games Dataset](https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-07-27/olympics.csv)
- **Size:** 271,116 athlete records across 120 years (1896–2016).
- **Attributes:** 15 columns (demographics, physical measurements, event categories, medal achievements).

---

## 👤 Author
**Aditi Paul**  
- GitHub: [@Aditipaul17](https://github.com/Aditipaul17)  
- Repository: [AditiPaul_olympic_medal_analysis](https://github.com/Aditipaul17/AditiPaul_olympic_medal_analysis)
