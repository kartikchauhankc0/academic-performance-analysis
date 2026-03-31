import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_excel("data.xlsx")

dataset = dataset.drop(columns=["Any additional comments or suggestions?", "Timestamp"], errors='ignore')
dataset.columns = dataset.columns.str.strip()

dataset["Year of Study"] = dataset["Year of Study"].fillna("4th year")
dataset["Living Situation"] = dataset["Living Situation"].fillna(dataset["Living Situation"].mode()[0])

dataset["Rate your sleep quality"] = dataset["Rate your sleep quality"].fillna(
    dataset["Rate your sleep quality"].mode()[0]
)

dataset["How would you rate your stress levels?"] = dataset["How would you rate your stress levels?"].fillna(
    dataset["How would you rate your stress levels?"].mode()[0]
)

encoder = OrdinalEncoder()
categorical_cols = dataset.select_dtypes(include="object").columns
dataset[categorical_cols] = encoder.fit_transform(dataset[categorical_cols])

plt.figure(figsize=(6,4))
sns.histplot(dataset["What is your current overall CGPA/GPA?"], kde=True)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(dataset.corr(), cmap="coolwarm")
plt.show()

target = "What is your current overall CGPA/GPA?"

dataset["cgpa_category"] = pd.cut(
    dataset[target],
    bins=[0, 6, 7.5, 10],
    labels=["Low", "Medium", "High"]
)

X = dataset.drop(columns=[target, "cgpa_category"])
y = dataset["cgpa_category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.show()