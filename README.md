# **🚗 Car Dheko - Used Car Price Prediction**

## 📝 Project Overview

This project aims to build an accurate and interactive machine learning model that predicts the price of used cars based on several features such as brand, model, fuel type, transmission, kilometers driven, and more. The model is deployed using **Streamlit** for real-time predictions, making it accessible for both customers and sales representatives.

---

## 📊 Project Objectives

- Predict the price of a used car using machine learning techniques.
- Clean, process, and analyze data from various cities in India.
- Build a user-friendly **Streamlit** web application for price prediction.
- Optimize model performance and present results through visualizations.

---

## 🧠 Skills Demonstrated

- Data Cleaning and Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Machine Learning (Regression Models)  
- Model Evaluation and Optimization  
- Streamlit App Development  
- Documentation & Reporting  

---

### 1. Data Processing

- Merged datasets from **Kolkata, Delhi, Bangalore, Chennai, Hyderabad, Jaipur**.
- Added a new `City` column to each dataset for origin tagging.
- Cleaned and transformed features (e.g., stripped units from numerical strings).
- Handled missing values using statistical imputation techniques.
- Encoded categorical variables using One-Hot and Label Encoding.
- Scaled numerical features using StandardScaler/MinMaxScaler.
- Removed outliers using IQR and Z-score methods.

### 2. Exploratory Data Analysis (EDA)

- Performed descriptive statistics on all features.
- Used visualizations like histograms, boxplots, scatter plots, and heatmaps.
- Identified key features affecting car prices (e.g., model year, fuel type, kms driven).

### 3. Model Development

- Train-Test Split (80-20 or 70-30).
- Models tried:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Cross-validated for stability.

### 4. Model Evaluation

- Evaluation Metrics: 
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
- Compared models and selected the best performer based on metric scores.

### 5. Optimization

- Applied Feature Engineering and Regularization (L1, L2).
- Performed Hyperparameter Tuning using GridSearchCV.

### 6. Deployment

- Developed a Streamlit app to take user input and return price predictions.
- Ensured a smooth, user-friendly experience.

---

## 🧾 Dataset Details

- Sourced from **CarDekho** for multiple cities.
- Contains features like:
  - Manufacturer (`oem`)
  - Car Model (`model`)
  - Year (`modelYear`)
  - Fuel Type (`ft`)
  - Transmission (`transmission`)
  - Kilometers Driven (`km`)
  - Number of Owners (`ownerNo`)
  - Price (`priceActual`)
- For full feature description, refer [Feature Description](https://docs.google.com/document/d/1hxW7IvCX5806H0IsG2Zg9WnVIpr2ZPueB4AElMTokGs/edit?usp=sharing)

---

## 📈 Results

- Built and deployed a high-performing used car price prediction model.
- Streamlit app functional for public and internal use.
- Achieved low error scores on evaluation metrics.
- Provided comprehensive documentation and visual reports.

---

## 🚀 Streamlit App

> Users can input car details like brand, year, fuel type, and get an instant prediction.

Features:

- Real-time predictions
- Clean interface
- Mobile and desktop-friendly

---

## 📦 Deliverables

- Source code for:
  - Data cleaning and preprocessing
  - Model training and tuning
  - Streamlit app
- Streamlit deployment files
- Visual EDA report
- Documentation (this README, methodology)
- Final trained ML model

---

## 📊 Evaluation Metrics

- **MAE**, **MSE**, **R² Score**
- Quality of data preprocessing
- Usability and UI of Streamlit app
- Clarity of documentation

---

## 🧰 Tools and Technologies

- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, LightGBM
- Streamlit for deployment
- Jupyter Notebooks
- Git for version control

---

## 📆 Timeline

- **Total Duration:** 10 Days  
  - Day 1-2: Data Cleaning & Merging  
  - Day 3-4: EDA  
  - Day 5-6: Model Development & Evaluation  
  - Day 7: Model Optimization  
  - Day 8-9: Streamlit App & Testing  
  - Day 10: Documentation & Submission  

---

## 🔗 Useful Links

- [Dataset (Google Drive)](https://drive.google.com/drive/folders/16U7OH7URsCW0rf91cwyDqEgd9UoeZAJh)
- [Capstone Project Guidelines](https://docs.google.com/document/d/1gbhLvJYY7J73lu1g9c6C9LRJvYemiDOdRDAEMe632w8/edit)

---

## ✍️ Author

### Vivek Duggal

[linkedin](https://www.linkedin.com/in/vivekkduggal/)  
*GUVI HCL Data Science Trainee*

---

## 📢 Acknowledgments

Thanks to **CarDekho** for the dataset and **GUVI** for mentoring and project structure.
