import io
import os
import re
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Aspect-Based Sentiment Analysis AI Engine")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Templates
templates = Jinja2Templates(directory="templates")

# Sentiment Map
LABELS_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Model Paths
MODEL_PATH = "absa_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ ML Model & Vectorizer loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning loading pkl files: {e}")

# Aspect extraction dictionary based on app feedback domain
ASPECT_KEYWORDS = {
    "Ads / Premium / Subscription": ["ads", "ad", "pay", "money", "subscription", "price", "premium", "cost", "paid", "buy", "charge"],
    "Speaking & Pronunciation": ["speech", "voice", "pronounce", "pronunciation", "accent", "mic", "speaking", "audio", "listen", "sound"],
    "Learning Effectiveness": ["learn", "easy", "helpful", "effective", "understand", "practice", "grammar", "vocabulary", "lesson", "teach"],
    "UI / App Performance": ["crash", "bug", "slow", "interface", "ui", "design", "freeze", "error", "load", "lag", "update"],
    "Language & Course Variety": ["language", "course", "content", "words", "spanish", "french", "german", "level", "variety"]
}

def extract_aspect(text: str) -> str:
    """Extracts aspect automatically from text if none is provided."""
    text_lower = text.lower()
    for aspect_name, keywords in ASPECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return aspect_name
    return "General / Overall"


def predict_sentiment_logic(text: str, aspect: str = ""):
    """Predicts sentiment using ML model if loaded, else fallback rule-based logic."""
    text_clean = str(text).strip()
    
    # Auto-extract aspect if empty or not provided
    extracted_aspect = aspect.strip() if aspect and aspect.strip() else extract_aspect(text_clean)

    if not text_clean:
        return {
            "aspect": "General / Overall",
            "sentiment": "Neutral",
            "confidence": 50.0,
            "breakdown": {"Positive": 0.0, "Neutral": 100.0, "Negative": 0.0},
        }

    # Model Prediction Logic
    if model is not None and vectorizer is not None:
        try:
            full_text = f"{extracted_aspect} {text_clean}".strip()
            vec_text = vectorizer.transform([full_text])

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vec_text)[0]
                pos_pct = round(float(probs[2]) * 100, 1) if len(probs) > 2 else 0.0
                neu_pct = round(float(probs[1]) * 100, 1) if len(probs) > 1 else 0.0
                neg_pct = round(float(probs[0]) * 100, 1)

                pred_idx = int(np.argmax(probs))
                sentiment = LABELS_MAP.get(pred_idx, "Neutral")
                confidence = round(float(probs[pred_idx]) * 100, 1)
            else:
                pred = model.predict(vec_text)[0]
                sentiment = (
                    LABELS_MAP.get(int(pred), "Neutral")
                    if isinstance(pred, (int, np.integer))
                    else str(pred).capitalize()
                )
                confidence = 90.0
                pos_pct, neu_pct, neg_pct = (
                    (100.0, 0.0, 0.0)
                    if sentiment == "Positive"
                    else (
                        (0.0, 100.0, 0.0)
                        if sentiment == "Neutral"
                        else (0.0, 0.0, 100.0)
                    )
                )

            return {
                "aspect": extracted_aspect,
                "sentiment": sentiment,
                "confidence": confidence,
                "breakdown": {
                    "Positive": pos_pct,
                    "Neutral": neu_pct,
                    "Negative": neg_pct,
                },
            }
        except Exception as e:
            print(f"Inference Error: {e}")

    # Fallback Rule-Based Logic with Breakdown
    text_lower = text_clean.lower()
    if any(
        w in text_lower
        for w in ["good", "great", "excellent", "love", "awesome", "best", "fast"]
    ):
        return {
            "aspect": extracted_aspect,
            "sentiment": "Positive",
            "confidence": 85.0,
            "breakdown": {"Positive": 85.0, "Neutral": 10.0, "Negative": 5.0},
        }
    elif any(
        w in text_lower
        for w in [
            "bad",
            "worst",
            "terrible",
            "slow",
            "crash",
            "error",
            "hate",
            "bug",
        ]
    ):
        return {
            "aspect": extracted_aspect,
            "sentiment": "Negative",
            "confidence": 85.0,
            "breakdown": {"Positive": 5.0, "Neutral": 10.0, "Negative": 85.0},
        }

    return {
        "aspect": extracted_aspect,
        "sentiment": "Neutral",
        "confidence": 60.0,
        "breakdown": {"Positive": 20.0, "Neutral": 60.0, "Negative": 20.0},
    }


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict_single(
    app_name: str = Form(""),
    aspect: str = Form(""),
    review: str = Form(...),
):
    try:
        if not review.strip():
            return JSONResponse(
                status_code=400, content={"error": "Review text cannot be empty"}
            )

        res = predict_sentiment_logic(review, aspect)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Single Prediction Error: {str(e)}"}
        )


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file format. Upload .csv or .xlsx"},
            )

        if df.empty:
            return JSONResponse(
                status_code=400, content={"error": "Uploaded file is empty"}
            )

        # Dynamic Review Column Detection
        review_col = None
        for col in df.columns:
            if re.search(
                r"(review|comment|text|feedback|description|content)",
                str(col),
                re.IGNORECASE,
            ):
                review_col = col
                break

        if review_col is None:
            string_cols = df.select_dtypes(include=["object"]).columns
            review_col = string_cols[0] if len(string_cols) > 0 else df.columns[0]

        predictions = []
        extracted_aspects = []
        
        for val in df[review_col]:
            res = predict_sentiment_logic(str(val))
            predictions.append(res["sentiment"])
            extracted_aspects.append(res["aspect"])

        df["Extracted_Aspect"] = extracted_aspects
        df["Predicted_Sentiment"] = predictions

        pos_count = int((df["Predicted_Sentiment"] == "Positive").sum())
        neu_count = int((df["Predicted_Sentiment"] == "Neutral").sum())
        neg_count = int((df["Predicted_Sentiment"] == "Negative").sum())

        return JSONResponse(
            content={
                "total_rows": len(df),
                "summary": {
                    "Positive": pos_count,
                    "Neutral": neu_count,
                    "Negative": neg_count,
                },
                "table_html": df.head(10).to_html(
                    classes="styled-table", index=False
                ),
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"File Processing Error: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
