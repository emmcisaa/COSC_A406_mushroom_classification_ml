# mushroom_classification.py

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from xgboost import XGBClassifier

# Loading Data
df = pd.read_csv("/Users/erinmcisaac/Desktop/STEM/COSC_A406/COSC_A406_mushroom_classification_ml/mushrooms.csv")


# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Class distribution
print("\nClass distribution:\n", df["class"].value_counts())

# Ensure plot directory exists
os.makedirs("plots", exist_ok=True)

# Plot class distribution
sns.countplot(data=df, x="class")
plt.title("Mushroom Class Distribution (Edible vs Poisonous)")
plt.xlabel("Class (e = Edible, p = Poisonous)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/class_distribution.png")
plt.close()

# Encode Categorical Features
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models and Evaluate

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acc_knn = accuracy_score(y_test, knn.predict(X_test))
print("KNN Accuracy:", acc_knn)

# Logistic Regression
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
acc_logreg = accuracy_score(y_test, logreg.predict(X_test))
print("Logistic Regression Accuracy:", acc_logreg)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", acc_rf)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
acc_xgb = accuracy_score(y_test, xgb.predict(X_test))
print("XGBoost Accuracy:", acc_xgb)

# Neural Network (Keras)
nn = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
loss_nn, acc_nn = nn.evaluate(X_test, y_test, verbose=0)
print("Neural Network Accuracy:", acc_nn)


# Confusion Matrix (Random Forest)
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders["class"].classes_)
disp.plot()
plt.title("Confusion Matrix – Random Forest")
plt.savefig("plots/rf_confusion_matrix.png")
plt.close()


# Feature Importances (Random Forest)
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=features[indices][:10])
plt.title("Top 10 Feature Importances – Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("plots/rf_feature_importance.png")
plt.close()

# Final Accuracy Summary
print("\nFinal Accuracy Summary:")
print(f"KNN: {acc_knn:.4f}")
print(f"Logistic Regression: {acc_logreg:.4f}")
print(f"Random Forest: {acc_rf:.4f}")
print(f"XGBoost: {acc_xgb:.4f}")
print(f"Neural Network: {acc_nn:.4f}")