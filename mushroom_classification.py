# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("/Users/erinmcisaac/Desktop/STEM/COSC_A406/COSC_A406_mushroom_classification_ml/mushrooms.csv")

# Reading dataset
print(df.head())
print(df.info())
print(df['class'].value_counts())

# Step 4: Encode all categorical variables
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Step 5: Split data into X and y
X = df.drop('class', axis=1)
y = df['class']

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preprocessing complete.")
