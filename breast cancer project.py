#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Step 1: Data Preparation and Exploration
import pandas as pd

# Load the dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis", "Mean Radius", "Mean Texture", ...]  # Define column names
df = pd.read_csv(data_url, names=column_names)

# Explore the dataset
print(df.head())
print(df.info())
print(df.describe())

# Step 2: Data Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Encode the target variable
label_encoder = LabelEncoder()
df['Diagnosis'] = label_encoder.fit_transform(df['Diagnosis'])

# Split the data into features and target variable
X = df.drop(['ID', 'Diagnosis'], axis=1)
y = df['Diagnosis']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Model Selection and Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation and Optimization
from sklearn.metrics import classification_report

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Model Deployment
import joblib

# Save the model
joblib.dump(model, 'breast_cancer_model.pkl')


# In[ ]:




