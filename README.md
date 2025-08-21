Heart Disease Prediction Project  

This project aims to analyze and predict heart disease using Machine Learning models.  
It includes data preprocessing, feature selection, model training, hyperparameter tuning, clustering, and deployment with Streamlit.  



 File Structure  

Heart_Disease_Project/
│── data/
│ └── heart_disease.csv
│
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ └── 06_hyperparameter_tuning.ipynb
│
│── models/
│ └── final_model.pkl
│
│── ui/
│ └── app.py (Streamlit UI)
│
│
│── requirements.txt
│── README.md
│── .gitignore

markdown
Copy
Edit

---

 Steps in the Project  

1. **Data Preprocessing**  
   - Handle missing values  
   - Encode categorical variables  
   - Normalize numerical features  

2. **Dimensionality Reduction (PCA)**  
   - Reduce features while keeping variance  

3. **Feature Selection**  
   - Random Forest / XGBoost feature importance  
   - Recursive Feature Elimination (RFE)  
   - Chi-Square test  

4. **Supervised Learning**  
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - Support Vector Machine (SVM)  
   - Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC  

5. **Unsupervised Learning (Clustering)**  
   - K-Means (Elbow method)  
   - Hierarchical clustering (Dendrogram)  
   - Compare clusters with disease labels  

6. **Hyperparameter Tuning**  
   - GridSearchCV  
   - RandomizedSearchCV  

7. **Model Export & Deployment**  
   - Save best model as `.pkl`  
   - Build interactive app with **Streamlit**  
   - Deployment via **Ngrok / Streamlit Cloud**  

---

Installation  

Clone the repository:  
bash
git clone https://github.com/101-shosho/Heart-Disease-Prediction-Project.git
cd Heart-Disease-Prediction-Project
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Project
Run Streamlit UI:

bash
Copy
Edit
streamlit run HEART_PROJECT.py

 Results
The models were compared using multiple metrics.

The best model achieved high accuracy and robust performance after hyperparameter tuning.

Detailed evaluation can be found in results/evaluation_metrics.txt.

Deployment with Ngrok
Run the Streamlit app:

bash
Copy
Edit
streamlit run HEART_PROJECT.py
Start ngrok:


 Author
Shahd Ashraf Ghazy
AI Student | Data & ML Enthusiast

GitHub: 101-shosho

