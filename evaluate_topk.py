import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split

print("Loading model and data...")
model = joblib.load("betterhealth_model.pkl")

with open("symptom_columns.json") as f:
    SYMPTOM_COLUMNS = json.load(f)

df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
X = df.drop(columns=["disease"])
y = df["disease"]

# Same filtering and split as training
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 2].index
mask = y.isin(valid_classes)
X, y = X[mask], y[mask]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Evaluating on {len(X_test)} test samples...")
probas = model.predict_proba(X_test)
classes = model.classes_

top1 = top3 = top5 = 0
for i, true_label in enumerate(y_test):
    top_indices = np.argsort(probas[i])[::-1]
    top_predictions = classes[top_indices]
    if true_label == top_predictions[0]:
        top1 += 1
    if true_label in top_predictions[:3]:
        top3 += 1
    if true_label in top_predictions[:5]:
        top5 += 1

total = len(y_test)
print(f"\n{'='*40}")
print(f"  Top-1 accuracy : {top1/total:.2%}  (exact match)")
print(f"  Top-3 accuracy : {top3/total:.2%}  (correct in top 3)")
print(f"  Top-5 accuracy : {top5/total:.2%}  (correct in top 5)")
print(f"{'='*40}")
print("\nTop-5 is the key metric for BetterHealth —")
print("the doctor just needs to see the right answer in the list.")
