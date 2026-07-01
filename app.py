import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

# Load Dataset

df = pd.read_csv("iris.csv")

# # Check Data
print(df.head())

# # Shape
print(df.shape)

# # Columns
print(df.columns)

# # Information
print(df.info())

# # Statistics
print(df.describe())

# # Missing Values
print(df.isnull().sum())

# # Species Count
print(df["Species"].value_counts())

# # Histogram

df.hist(figsize=(10,8))
plt.show()

# # Scatter Plot

plt.scatter(
    df["SepalLengthCm"],
    df["PetalLengthCm"]
)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")

plt.show()

# # Features

X = df.drop(["Id", "Species"], axis=1)

# # Target

y = df["Species"]

# # Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# # Model

model = RandomForestClassifier(random_state=42)

# # Train

model.fit(X_train, y_train)

# Prediction

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

# # Accuracy

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# # Confusion Matrix

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix")
print(cm)

# # Classification Report

print("\nClassification Report")

print(classification_report(y_test, predictions))   