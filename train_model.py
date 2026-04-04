import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# ── 2. Separate features and target ──────────────────────────────────────────
X = df.drop(columns=["disease"])  # all symptom columns
y = df["disease"]  # target label

# Save symptom column names for use in the API later
symptom_columns = list(X.columns)
print(f"Symptoms: {len(symptom_columns)} features")
print(f"Diseases: {y.nunique()} unique classes")

# ── 3. Remove diseases with only 1 sample (can't split them) ─────────────────
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 2].index
removed = len(class_counts) - len(valid_classes)
if removed > 0:
    print(f"Removing {removed} disease(s) with only 1 sample...")
    mask = y.isin(valid_classes)
    X, y = X[mask], y[mask]
    print(f"Remaining: {y.nunique()} diseases, {len(y)} rows")

# ── 4. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

# ── 4. Train Random Forest ────────────────────────────────────────────────────
print("Training model... (this may take 2-5 minutes on 246k rows)")
model = RandomForestClassifier(
    n_estimators=100,  # 100 trees — good balance of speed vs accuracy
    max_depth=20,  # prevents overfitting
    min_samples_leaf=2,
    n_jobs=-1,  # use all CPU cores
    random_state=42,
    verbose=1,
)
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\nEvaluating...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# Overall accuracy
accuracy = (y_pred == y_test).mean()
print(f"\nOverall accuracy: {accuracy:.2%}")

# ── 6. Save model ─────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_DIR / "betterhealth_model.pkl")
print("\nModel saved to model/betterhealth_model.pkl")

# ── 7. Build symptom frequency map (for follow-up question engine) ────────────
# For each disease, find which symptoms are most commonly present (value=1)
print("\nBuilding symptom frequency map...")
freq_map = {}
for disease in df["disease"].unique():
    disease_rows = df[df["disease"] == disease].drop(columns=["disease"])
    # Get symptoms present in >50% of cases for this disease
    common_symptoms = disease_rows.mean()
    common_symptoms = common_symptoms[common_symptoms > 0.5].index.tolist()
    freq_map[disease] = common_symptoms

with open(MODEL_DIR / "symptom_frequency_map.json", "w") as f:
    json.dump(freq_map, f)
with open(MODEL_DIR / "symptom_columns.json", "w") as f:
    json.dump(symptom_columns, f)
print("Symptom frequency map saved to model/symptom_frequency_map.json")
print("\nDone! Files created:")
print("  - model/betterhealth_model.pkl")
print("  - model/symptom_columns.json")
print("  - model/symptom_frequency_map.json")
