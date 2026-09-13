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

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template Setup
templates = Jinja2Templates(directory="templates")

# Sentiment Label Map
LABELS_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# --- Load ML Model & Vectorizer Safely ---
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


def predict_sentiment_logic(text: str, aspect: str = ""):
    """Predicts sentiment using ML model if loaded, else fallback rule-based."""
    text_clean = str(text).strip()
    if not text_clean:
        return {"sentiment": "Neutral", "confidence": 50.0}

    # Model Prediction
    if model is not None and vectorizer is not None:
        try:
            full_text = f"{aspect} {text_clean}".strip()
            vec_text = vectorizer.transform([full_text])
            pred = model.predict(vec_text)[0]

            if isinstance(pred, (int, np.integer)):
                sentiment = LABELS_MAP.get(int(pred), "Neutral")
            else:
                sentiment = str(pred).capitalize()

            return {"sentiment": sentiment, "confidence": 90.0}
        except Exception as e:
            print(f"Inference Error: {e}")

    # Fallback Rule-Based Logic
    text_lower = text_clean.lower()
    if any(
        w in text_lower
        for w in ["good", "great", "excellent", "love", "awesome", "best", "fast"]
    ):
        return {"sentiment": "Positive", "confidence": 85.0}
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
        return {"sentiment": "Negative", "confidence": 85.0}

    return {"sentiment": "Neutral", "confidence": 60.0}


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    # Python 3.14 & Latest Starlette compatibility syntax
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

        # Detect Review Column Dynamic Syntax
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
        for val in df[review_col]:
            res = predict_sentiment_logic(str(val))
            predictions.append(res["sentiment"])

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
