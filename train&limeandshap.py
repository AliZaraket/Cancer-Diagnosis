# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
import joblib  # To save/load models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
df = pd.read_csv(url, names=column_names)

# Drop ID column
df.drop(columns=['ID'], inplace=True)

# Convert Diagnosis to binary (M=1, B=0)
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

# Step 2: Data Preprocessing
X = df.drop(columns=['Diagnosis'])  # Features
y = df['Diagnosis']  # Target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train SVM Models with Different Kernels
svm_linear = SVC(kernel='linear', probability=True, random_state=42)
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
svm_sigmoid = SVC(kernel='sigmoid', probability=True, random_state=42)

# Train models
svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)
svm_sigmoid.fit(X_train, y_train)

# Train Gradient Boosting Classifier
gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gradient_boosting.fit(X_train, y_train)

# Step 4: Train Ensemble Model (Voting Classifier)
ensemble_model = VotingClassifier(
    estimators=[
        ('svm_linear', svm_linear),
        ('svm_rbf', svm_rbf),
        ('svm_sigmoid', svm_sigmoid),
        ('gradient_boosting', gradient_boosting)
    ],
    voting='soft'  # Soft voting for probability averaging
)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(ensemble_model, "ensemble_model.pkl")
print("✅ Model trained and saved as 'ensemble_model.pkl'")

# Step 5: Evaluate Model on Test Set
y_pred_ensemble = ensemble_model.predict(X_test)
print("✅ Ensemble Model Performance:\n", classification_report(y_test, y_pred_ensemble))

# Step 6: SHAP Analysis (Global Feature Importance)
shap.initjs()
explainer = shap.Explainer(gradient_boosting, X_train)
shap_values = explainer(X_test)

# Save SHAP Summary Plot (Whole Test Set)
shap.summary_plot(shap_values, X_test, feature_names=df.columns[1:], show=False)
plt.savefig("shap_summary_plot.png")
plt.show()
print("✅ SHAP summary plot saved as 'shap_summary_plot.png'")

# Step 7: SHAP Force Plot for a Single Prediction
i = 10  # Choose a sample
shap_values_exp = shap.Explanation(
    values=shap_values.values,  
    base_values=shap_values.base_values,  
    data=X_test,  
    feature_names=df.columns[1:]
)

# Save SHAP Force Plot
shap.save_html("shap_force_plot.html", shap.force_plot(
    shap_values_exp.base_values[i],  
    shap_values_exp.values[i, :],  
    X_test[i, :],  
    feature_names=df.columns[1:]
))
print("✅ SHAP force plot saved as 'shap_force_plot.html'")

# Step 8: LIME Explanation (Single Test Sample)
explainer = lime.lime_tabular.LimeTabularExplainer(
    scaler.inverse_transform(X_train),  # Transform back to original range
    feature_names=df.columns[1:], 
    class_names=['Benign', 'Malignant'], 
    discretize_continuous=True
)

# Choose a test sample
i = 3
exp = explainer.explain_instance(
    scaler.inverse_transform(X_test[i, :].reshape(1, -1)).flatten(),
    gradient_boosting.predict_proba,
    num_features=10
)

# Save LIME Explanation Plot
lime_fig = exp.as_pyplot_figure()
lime_fig.savefig("lime_explanation.png")
plt.show()
print("✅ LIME explanation saved as 'lime_explanation.png'")
