from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import numpy as np
from google import genai
from datetime import datetime, timezone
import os
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CASES_DIR = BASE_DIR / "cases"
MODEL_FILE = MODEL_DIR / "betterhealth_model.pkl"
SYMPTOM_COLUMNS_FILE = MODEL_DIR / "symptom_columns.json"
FREQ_MAP_FILE = MODEL_DIR / "symptom_frequency_map.json"
CASES_FILE = CASES_DIR / "new_cases.jsonl"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CONFIDENCE_THRESHOLD = 5.0  # min % to include in results
MIN_SYMPTOMS_TO_PREDICT = 3  # need at least 3 symptoms to predict

app = FastAPI(title="BetterHealth AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model and supporting files on startup ────────────────────────────────
print("Loading model...")
model = None
if MODEL_FILE.exists():
    model = joblib.load(MODEL_FILE)
else:
    print(f"Warning: Model file not found at {MODEL_FILE}")

SYMPTOM_COLUMNS = []
if SYMPTOM_COLUMNS_FILE.exists():
    with open(SYMPTOM_COLUMNS_FILE) as f:
        SYMPTOM_COLUMNS = json.load(f)
else:
    print(f"Warning: Symptom columns file not found at {SYMPTOM_COLUMNS_FILE}")

FREQ_MAP = {}
if FREQ_MAP_FILE.exists():
    with open(FREQ_MAP_FILE) as f:
        FREQ_MAP = json.load(f)
else:
    print(f"Warning: Symptom frequency map file not found at {FREQ_MAP_FILE}")

# Setup Gemini
gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

if not GEMINI_API_KEY:
    print(
        "Warning: GEMINI_API_KEY not set. Symptom extraction will return empty results."
    )

model_classes_count = len(model.classes_) if model is not None else 0
print(f"Ready. {len(SYMPTOM_COLUMNS)} symptoms, {model_classes_count} diseases.")


# ── Request / Response models ─────────────────────────────────────────────────


class InitialInput(BaseModel):
    text: str  # free text from assistant
    session_id: str  # unique ID per patient session


class FollowUpInput(BaseModel):
    session_id: str
    existing_symptoms: dict[str, int]  # symptoms collected so far
    follow_up_answers: str  # text OR converted from yes/no buttons
    questions_asked: list[str]  # the questions that were asked


class SaveCaseInput(BaseModel):
    session_id: str
    symptoms: dict[str, int]
    confirmed_diagnosis: str
    notes: str = ""
    medication: str = ""
    test_results: str = ""


# ── Helper: Extract symptoms from text using Gemini ───────────────────────────


def extract_symptoms_with_gemini(text: str, symptom_list: list[str]) -> list[str]:
    """
    Ask Gemini to read natural language and return matching symptoms
    from our known symptom list.
    """
    if gemini is None:
        return []

    prompt = f"""
You are a medical symptom extraction assistant.

A medical assistant has described a patient's condition in plain English.
Your job is to extract symptoms from their description and match them to
the closest terms from the official symptom list below.

PATIENT DESCRIPTION:
"{text}"

OFFICIAL SYMPTOM LIST (these are the only valid outputs):
{json.dumps(symptom_list, indent=2)}

INSTRUCTIONS:
- Read the description carefully
- Match any mentioned symptoms to the closest term in the official list
- Only return symptoms that are clearly present or described
- Return ONLY a valid JSON array of matched symptom strings, exactly as they appear in the list
- If nothing matches, return an empty array []
- Do not include explanations, markdown, or any text outside the JSON array

Example output format:
["chest pain", "shortness of breath", "fatigue"]
"""
    try:
        response = gemini.models.generate_content(  # FIX 2: added model + contents
            model="gemini-1.5-flash", contents=prompt
        )
        raw = response.text.strip()

        # Clean up in case Gemini adds markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        extracted = json.loads(raw)

        # Validate — only keep symptoms that actually exist in our list
        valid = [s for s in extracted if s in symptom_list]
        return valid

    except Exception as e:
        print(f"Gemini extraction error: {e}")
        return []


def parse_followup_answers_with_gemini(
    questions: list[str], answers_text: str, symptom_list: list[str]
) -> dict[str, int]:
    """
    Given follow-up questions and the assistant's answers (free text or
    converted from yes/no buttons), extract which symptoms are present (1)
    or absent (0).
    """
    if gemini is None:
        return {}

    prompt = f"""
You are a medical symptom extraction assistant.

A medical assistant was asked these follow-up questions about a patient:
{json.dumps(questions, indent=2)}

The assistant's answers were:
"{answers_text}"

OFFICIAL SYMPTOM LIST:
{json.dumps(symptom_list, indent=2)}

INSTRUCTIONS:
- For each question, determine if the answer confirms the symptom is PRESENT or ABSENT
- Match each question's symptom to the closest term in the official symptom list
- Return ONLY a valid JSON object mapping symptom name to 1 (present) or 0 (absent)
- Only include symptoms from the official list
- Do not include explanations or any text outside the JSON object

Example output format:
{{"palpitations": 1, "irregular heartbeat": 0, "dizziness": 1}}
"""
    try:
        response = gemini.models.generate_content(  # FIX 3: added model + contents
            model="gemini-1.5-flash", contents=prompt
        )
        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        # Validate keys are real symptoms
        valid = {k: int(v) for k, v in parsed.items() if k in symptom_list}
        return valid

    except Exception as e:
        print(f"Gemini follow-up parse error: {e}")
        return {}


# ── Helper: Run Random Forest prediction ─────────────────────────────────────


def run_prediction(symptoms_dict: dict[str, int]) -> tuple[list[dict], list[str]]:
    """
    Given a symptoms dict {symptom_name: 0/1}, run the Random Forest
    and return (top_conditions, follow_up_questions).
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model and place files in model/.",
        )

    input_vector = np.array(
        [symptoms_dict.get(col, 0) for col in SYMPTOM_COLUMNS]
    ).reshape(1, -1)

    probas = model.predict_proba(input_vector)[0]
    classes = model.classes_

    top_indices = np.argsort(probas)[::-1][:5]
    top_conditions = [
        {"disease": classes[i], "confidence": round(float(probas[i]) * 100, 1)}
        for i in top_indices
        if probas[i] * 100 >= CONFIDENCE_THRESHOLD
    ]

    # Follow-up questions based on top disease's common symptoms
    follow_up_questions = []
    if top_conditions:
        top_disease = top_conditions[0]["disease"]
        common = FREQ_MAP.get(top_disease, [])
        reported_present = {k for k, v in symptoms_dict.items() if v == 1}
        follow_up_questions = [
            s.replace("_", " ").capitalize()
            for s in common
            if s not in reported_present
        ][:5]

    return top_conditions, follow_up_questions


# ── Helper: Decide response status ───────────────────────────────────────────


def decide_status(symptoms_dict: dict[str, int], top_conditions: list[dict]) -> str:
    """
    need_more_info           — too few symptoms, don't predict yet
    prediction_with_followup — predicted but follow-ups will help refine
    prediction_ready         — confident enough result
    """
    present_count = sum(1 for v in symptoms_dict.values() if v == 1)

    if present_count < MIN_SYMPTOMS_TO_PREDICT or not top_conditions:
        return "need_more_info"

    top_confidence = top_conditions[0]["confidence"]

    if top_confidence >= 60:
        return "prediction_ready"
    else:
        return "prediction_with_followup"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "BetterHealth AI API is running"}


@app.get("/symptoms")
def get_all_symptoms():
    return {"symptoms": SYMPTOM_COLUMNS}


@app.post("/analyze")
def analyze_initial_input(input: InitialInput):
    """
    STEP 1 — Called when assistant submits the initial patient description.

    Request:
      { "text": "patient has chest pain and is short of breath", "session_id": "abc123" }

    Response (need_more_info):
      { "status": "need_more_info", "extracted_symptoms": {}, "top_conditions": [],
        "follow_up_questions": [...], "message": "..." }

    Response (prediction_with_followup):
      { "status": "prediction_with_followup", "extracted_symptoms": {...},
        "top_conditions": [...], "follow_up_questions": [...], "message": "..." }

    Response (prediction_ready):
      { "status": "prediction_ready", "extracted_symptoms": {...},
        "top_conditions": [...], "follow_up_questions": [], "message": "..." }
    """
    # 1. Extract symptoms from natural language
    extracted = extract_symptoms_with_gemini(input.text, SYMPTOM_COLUMNS)

    # 2. If Gemini found nothing at all
    if not extracted:
        return {
            "status": "need_more_info",
            "session_id": input.session_id,
            "extracted_symptoms": {},
            "top_conditions": [],
            "follow_up_questions": [
                "Where exactly is the discomfort or pain located?",
                "Does the patient have a fever or chills?",
                "How long have they been experiencing these symptoms?",
                "Are they experiencing any pain? If so, rate it 1-10.",
                "Any nausea, vomiting, or digestive issues?",
            ],
            "message": "Description was too vague. Please gather more specific information.",
        }

    symptoms_dict = {s: 1 for s in extracted}
    present_count = len(extracted)

    # 3. Too few symptoms — ask follow-ups before predicting
    if present_count < MIN_SYMPTOMS_TO_PREDICT:
        return {
            "status": "need_more_info",
            "session_id": input.session_id,
            "extracted_symptoms": symptoms_dict,
            "top_conditions": [],
            "follow_up_questions": [
                "Does the patient have a fever?",
                "Is there any pain? Where is it located?",
                "How long have these symptoms been present?",
                "Any difficulty breathing?",
                "Any nausea or vomiting?",
            ],
            "message": f"Found {present_count} symptom(s) — need at least {MIN_SYMPTOMS_TO_PREDICT} to predict.",
        }

    # 4. Run prediction
    top_conditions, follow_up_questions = run_prediction(symptoms_dict)
    status = decide_status(symptoms_dict, top_conditions)

    return {
        "status": status,
        "session_id": input.session_id,
        "extracted_symptoms": symptoms_dict,
        "top_conditions": top_conditions,
        "follow_up_questions": (
            follow_up_questions if status != "prediction_ready" else []
        ),
        "message": f"Extracted {present_count} symptom(s).",
    }


@app.post("/followup")
def process_followup(input: FollowUpInput):
    """
    STEP 2 — Called when assistant answers follow-up questions.

    The frontend can send answers as:
      A) Free text:  "yes to chills and sweating, no vomiting"
      B) Yes/No buttons converted to text: "Chills: yes, Sweating: yes, Vomiting: no"
         (frontend just joins the button answers into a string before sending)

    Request:
      {
        "session_id": "abc123",
        "existing_symptoms": {"chest pain": 1, "fatigue": 1},
        "follow_up_answers": "yes to chills, no vomiting",
        "questions_asked": ["Does the patient have chills?", "Any vomiting?"]
      }
    """
    # 1. Parse answers using Gemini
    new_symptoms = parse_followup_answers_with_gemini(
        input.questions_asked, input.follow_up_answers, SYMPTOM_COLUMNS
    )

    # 2. Merge with existing symptoms
    merged = {**input.existing_symptoms, **new_symptoms}

    # 3. Re-run prediction with fuller picture
    top_conditions, follow_up_questions = run_prediction(merged)
    status = decide_status(merged, top_conditions)

    present_count = sum(1 for v in merged.values() if v == 1)

    return {
        "status": status,
        "session_id": input.session_id,
        "extracted_symptoms": merged,
        "top_conditions": top_conditions,
        "follow_up_questions": (
            follow_up_questions if status == "prediction_with_followup" else []
        ),
        "message": f"Updated with follow-up answers. Now have {present_count} symptom(s).",
    }


@app.post("/save-case")
def save_case(input: SaveCaseInput):
    """
    STEP 3 — Called when doctor finalizes and saves the diagnosis.
    Stores the full case for future model retraining.
    """
    case = {
        "session_id": input.session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symptoms": input.symptoms,
        "confirmed_diagnosis": input.confirmed_diagnosis,
        "notes": input.notes,
        "medication": input.medication,
        "test_results": input.test_results,
    }

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    with open(CASES_FILE, "a") as f:
        f.write(json.dumps(case) + "\n")

    return {"status": "Case saved successfully", "session_id": input.session_id}
