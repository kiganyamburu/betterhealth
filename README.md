# BetterHealth

BetterHealth is a disease prediction project that combines:

- A machine learning model (Random Forest) trained on symptom vectors
- A FastAPI backend for inference workflows
- Gemini-assisted symptom extraction from free-text clinical notes

The API supports a 3-step flow:

1. Analyze initial free-text description
2. Process follow-up answers
3. Save finalized clinical cases for future retraining

## Quick Start (Windows / PowerShell)

1. Activate virtual environment:

```powershell
.\env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set Gemini key for the current terminal:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY_HERE"
```

4. Train and evaluate:

```powershell
.\tasks.ps1 retrain
```

5. Start API:

```powershell
.\tasks.ps1 serve
```

6. Open docs:

- http://127.0.0.1:8000/docs

## Project Structure

- api.py: FastAPI app and Gemini-assisted symptom parsing
- train_model.py: model training and artifact generation
- evaluate_topk.py: Top-1, Top-3, Top-5 evaluation
- tasks.ps1: simple task runner (train/evaluate/serve/test/retrain)
- tests/test_api.py: automated API endpoint tests
- requirements.txt: Python dependencies
- data/Final_Augmented_dataset_Diseases_and_Symptoms.csv: training dataset
- model/: model artifacts for API runtime
- cases/: saved finalized cases (JSONL)

## Requirements

- Python 3.10+
- pip
- A Gemini API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configure Gemini Key (Environment Variable)

The API reads the key from environment variable `GEMINI_API_KEY`.

PowerShell (current terminal session):

```powershell
$env:GEMINI_API_KEY="YOUR_KEY_HERE"
```

PowerShell (persist for your user):

```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "YOUR_KEY_HERE", "User")
```

After setting a persistent user variable, restart terminal/VS Code so it is picked up.

## Standardized Data and Model Paths

Scripts now use consistent locations:

- Dataset: data/Final_Augmented_dataset_Diseases_and_Symptoms.csv
- Model outputs: model/betterhealth_model.pkl
- Symptom columns: model/symptom_columns.json
- Frequency map: model/symptom_frequency_map.json

## Run with Task Runner

Use the included PowerShell task runner:

```powershell
.\tasks.ps1 train
.\tasks.ps1 evaluate
.\tasks.ps1 serve
.\tasks.ps1 test
.\tasks.ps1 retrain
```

tasks.ps1 automatically uses env/Scripts/python.exe when it exists.

Task meanings:

- train: train model and write artifacts to model/
- evaluate: compute Top-1/Top-3/Top-5 metrics
- serve: start FastAPI with reload
- test: run automated API tests
- retrain: run train then evaluate in sequence

## Manual Commands

Train model:

```bash
python train_model.py
```

Evaluate Top-K:

```bash
python evaluate_topk.py
```

Run API:

```bash
uvicorn api:app --reload
```

API URLs:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs

## Automated Tests

Run tests:

```bash
pytest tests -q
```

The test suite currently covers core API behavior in tests/test_api.py:

- Root health endpoint
- Analyze flow when too few symptoms are found
- Follow-up flow symptom merge behavior
- Save-case JSONL persistence

## Typical Workflow

1. Train or retrain model artifacts
2. Evaluate Top-K quality
3. Start API and test with /docs
4. Save finalized cases via /save-case
5. Periodically retrain to include new saved cases

## API Endpoints

### GET /

Health check endpoint.

### GET /symptoms

Returns all symptom columns used by the model.

### POST /analyze

Step 1: Analyze initial free-text description.

Sample request body:

```json
{
  "text": "patient has chest pain and shortness of breath",
  "session_id": "abc123"
}
```

### POST /followup

Step 2: Process follow-up answers and refine predictions.

Sample request body:

```json
{
  "session_id": "abc123",
  "existing_symptoms": { "chest pain": 1, "fatigue": 1 },
  "follow_up_answers": "yes to chills, no vomiting",
  "questions_asked": ["Does the patient have chills?", "Any vomiting?"]
}
```

### POST /save-case

Step 3: Save finalized diagnosis to cases/new_cases.jsonl.

Sample request body:

```json
{
  "session_id": "abc123",
  "symptoms": { "chest pain": 1, "fatigue": 1 },
  "confirmed_diagnosis": "Angina",
  "notes": "Patient improves with rest",
  "medication": "As prescribed",
  "test_results": "ECG pending"
}
```

## Troubleshooting

- ImportError for google.genai:
  Use the project virtual environment and reinstall requirements.

```powershell
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

- scikit-learn InconsistentVersionWarning when loading model:
  This means the saved model was trained with a different scikit-learn version.
  Run `.\tasks.ps1 retrain` to regenerate artifacts with your current environment.
