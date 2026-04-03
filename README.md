# BetterHealth

BetterHealth is a disease prediction project that combines:

- A machine learning model (Random Forest) trained on symptom vectors
- A FastAPI backend for inference workflows
- Gemini-assisted symptom extraction from free-text clinical notes

The API supports a 3-step flow:

1. Analyze initial free-text description
2. Process follow-up answers
3. Save finalized clinical cases for future retraining

## Project Structure

- api.py: FastAPI app and Gemini-assisted symptom parsing
- train_model.py: model training and artifact generation
- evaluate_topk.py: Top-1, Top-3, Top-5 evaluation
- requirements.txt: Python dependencies
- data/Final_Augmented_dataset_Diseases_and_Symptoms.csv: training dataset
- model/: model artifacts for API runtime
- cases/: saved finalized cases (JSONL)

## Requirements

- Python 3.10+
- pip
- A Gemini API key

Install dependencies:

pip install -r requirements.txt

## Configure Gemini Key

In api.py, set:

GEMINI_API_KEY = "YOUR_KEY_HERE"

You can also refactor this later to use environment variables for better security.

## Train the Model

Run:

python train_model.py

By default, train_model.py currently expects the dataset file in the project root:

- Final_Augmented_dataset_Diseases_and_Symptoms.csv

If your dataset is only in data/, either:

- Copy it to project root before training, or
- Update the path in train_model.py to data/Final_Augmented_dataset_Diseases_and_Symptoms.csv

Training outputs created by train_model.py:

- betterhealth_model.pkl
- symptom_columns.json
- symptom_frequency_map.json

To run the API with the current api.py implementation, place these files under model/:

- model/betterhealth_model.pkl
- model/symptom_columns.json
- model/symptom_frequency_map.json

## Evaluate Top-K Accuracy

Run:

python evaluate_topk.py

evaluate_topk.py also currently expects:

- betterhealth_model.pkl in project root
- symptom_columns.json in project root
- Final_Augmented_dataset_Diseases_and_Symptoms.csv in project root

If your files are in model/ and data/, update the paths in evaluate_topk.py accordingly.

## Run the API

Start FastAPI:

uvicorn api:app --reload

API will be available at:

- http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

## API Endpoints

### GET /

Health check endpoint.

### GET /symptoms

Returns all symptom columns used by the model.

### POST /analyze

Step 1: Analyze initial free-text description.

Sample request body:

{
"text": "patient has chest pain and shortness of breath",
"session_id": "abc123"
}

### POST /followup

Step 2: Process follow-up answers and refine predictions.

Sample request body:

{
"session_id": "abc123",
"existing_symptoms": {"chest pain": 1, "fatigue": 1},
"follow_up_answers": "yes to chills, no vomiting",
"questions_asked": ["Does the patient have chills?", "Any vomiting?"]
}

### POST /save-case

Step 3: Save finalized diagnosis to cases/new_cases.jsonl.

Sample request body:

{
"session_id": "abc123",
"symptoms": {"chest pain": 1, "fatigue": 1},
"confirmed_diagnosis": "Angina",
"notes": "Patient improves with rest",
"medication": "As prescribed",
"test_results": "ECG pending"
}

## Suggested Next Improvements

- Move Gemini key to environment variables
- Standardize file paths across scripts (all from data/ and model/)
- Add a simple Makefile or task runner for train/evaluate/serve commands
- Add automated tests for API endpoints
