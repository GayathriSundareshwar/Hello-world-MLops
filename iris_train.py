from sklearn.datasets import load_iris

import pandas as pd
import joblib
import os
import json

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
print(df.describe())
print("attempt 2 test")
print("\nTragetNames:", iris.target_names)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


X,y = iris.data, iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

#save model
base_dir=os.path.dirname(__file__)
artifacts_dir=os.path.join(base_dir, "artifacts")

os.makedirs(artifacts_dir, exist_ok=True)
model_path = os.path.join(artifacts_dir,"model.pkl")
joblib.dump(model, model_path)


#save tiny metrics file
acc=model.score(X_test,y_test)
metrics = {"accuracy": float(acc)}
with open(os.path.join(artifacts_dir,"metrics.json"), "w") as f:
    json.dump(metrics, f)
print(f"Model trained and saved to {model_path}")
print(f"Test Accuracy: {acc:.4f}")
