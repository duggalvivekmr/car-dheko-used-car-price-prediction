# 🚗 Car Dheko - Used Car Price Prediction

## 📝 Project Overview

This project develops a machine learning model to predict used car prices based on features like brand, model, year, fuel type, and kilometers driven. The final model is deployed via a **Streamlit** application for real-time usage by customers and sales agents.

---

## 🎯 Project Objectives

- Predict used car prices using supervised regression techniques.
- Clean, unify, and process multi-city datasets into a structured format.
- Visualize insights with EDA techniques.
- Build and deploy a Streamlit-based price prediction interface.
- Ensure reproducibility and version control using **Git** and **Git LFS**.

---

## 🧠 Skills Demonstrated

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Selection & Evaluation  
- Hyperparameter Tuning  
- Streamlit App Development  
- Git Version Control + Git LFS  
- Technical Documentation  

---

## 🗃️ Dataset & Features

The dataset combines used car listings from:

`Kolkata`, `Delhi`, `Bangalore`, `Chennai`, `Hyderabad`, and `Jaipur`.

Each row represents a car and includes:

| Feature         | Description                                   |
|----------------|-----------------------------------------------|
| `oem`          | Original Equipment Manufacturer               |
| `model`        | Car Model                                     |
| `modelYear`    | Year of Manufacture                           |
| `ft`           | Fuel Type (Petrol, Diesel, etc.)              |
| `transmission` | Manual/Automatic                              |
| `km`           | Kilometers Driven (cleaned to int)            |
| `ownerNo`      | Number of Previous Owners                     |
| `priceActual`  | Actual Sale Price                             |
| `city`         | City of Listing                               |

📎 Full feature dictionary: [Feature Description](https://docs.google.com/document/d/1hxW7IvCX5806H0IsG2Zg9WnVIpr2ZPueB4AElMTokGs/edit?tab=t.0)

---

## 🔧 Data Pipeline

1. **Import & Concatenate**
   - All city-wise datasets were merged after tagging with a `City` column.
   - Cleaned unstructured fields (e.g., stripped 'kms', converted prices to float).

2. **Handling Missing Values**
   - Numerical: Imputed using median/mean.
   - Categorical: Used mode or added "Unknown" category.

3. **Encoding & Scaling**
   - One-Hot Encoding for nominal categories.
   - Label/Ordinal Encoding for ordinal features.
   - MinMaxScaler/StandardScaler used based on model requirements.

4. **Outlier Removal**
   - IQR and Z-score techniques applied on numerical fields.

---

## 📊 Exploratory Data Analysis

- Visuals: Histograms, Boxplots, Correlation Heatmaps, Pairplots.
- Correlation insights on how fuel type, year, and OEM affect price.
- Identified high-variance features and performed dimensionality reduction where needed.

---

## 🤖 Model Development & Evaluation

| Step                  | Description |
|-----------------------|-------------|
| Train-Test Split      | 70:30 or 80:20 split |
| Algorithms Used       | Linear Regression, Decision Tree, Random Forest, XGBoost |
| Cross-Validation      | 5-fold CV |
| Evaluation Metrics    | MAE, MSE, R² Score |
| Hyperparameter Tuning | GridSearchCV / RandomizedSearchCV |

---

## 🛠 Optimization

- Regularization: L1 (Lasso), L2 (Ridge) to mitigate overfitting.
- Feature Engineering: Derived interaction terms and ratios.
- Model comparison based on test scores and bias-variance tradeoff.

---

## 🚀 Deployment

- **Streamlit App**
  - Allows users to select inputs via dropdowns and sliders.
  - Returns real-time price prediction with confidence indicator.

- **Deployment Requirements**
  - `requirements.txt` for environment setup.
  - GitHub-integrated CI/CD with version tracking.
  - App can be deployed on **Streamlit Cloud** or **Heroku**.

---

## 📦 Deliverables

- 🗂 Source Code: Preprocessing, Modeling, EDA, Deployment Scripts  
- 📊 EDA Report (HTML/Notebook)  
- 🧠 Trained ML Model (.pkl or .joblib)  
- 🌐 Streamlit App & Deployment Files  
- 📄 Final Documentation (README + Methodology)

---

## 💻 Tools & Technologies

- Python, Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn, XGBoost, LightGBM  
- Streamlit, Jupyter Notebook  
- Git & GitHub for version control  
- **Git LFS** for handling large dataset files and model binaries

---

## 📌 **Git LFS Setup**:

git lfs install
git lfs track "*.xlsx" "*.pkl" "*.joblib"
git add .gitattributes

---

## 📈 Results Summary 

- High accuracy model with R² > 0.85 on test set.
- Feature importance revealed modelYear, km, fuel type as top predictors.
- Deployed app is responsive and user-friendly.

---

## 📅 Project Timeline (10 Days)

- Days    Activities
- Day 1-2  Data Cleaning, Feature Extraction
- Day 3-4  EDA + Visual Insights
- Day 5-6  Model Training + Evaluation
- Day 7    Hyperparameter Tuning + Optimization
- Day 8-9  Streamlit App Development + Integration
- Day 10   Final Documentation + Submission

---
  
## 🔗 Resources

📁 Dataset [Google Drive](https://drive.google.com/drive/folders/16U7OH7URsCW0rf91cwyDqEgd9UoeZAJh)
📑 [Capstone Guidelines](https://drive.google.com/drive/folders/1QPn24zlTJVS94YtxXkUg70AzKuOCPCp6)

---

### ✍️ Author

Vivek Duggal
[LinkedIn](https://www.linkedin.com/in/vivekkduggal/)
GUVI HCL Data Science Trainee

🙏 Acknowledgments
Thanks to CarDekho for the dataset and GUVI for mentorship and project structure.
