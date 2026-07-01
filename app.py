import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add Target Column
df["species"] = iris.target

# Convert Number to Name
df["species"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# Check Data
print(df.head())

# Missing Values
print(df.isnull().sum())

# Species Count
print(df["species"].value_counts())

# Histogram
df.hist(figsize=(10,8))
plt.show()

# Scatter Plot
plt.scatter(
    df["sepal length (cm)"],
    df["petal length (cm)"]
)
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Sepal Length vs Petal Length")
plt.show()

# Features
X = df.drop("species", axis=1)

# Target
y = df["species"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report")

print(classification_report(y_test, predictions))

print(predictions[:10])