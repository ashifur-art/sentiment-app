import io
import re
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Aspect-Based Sentiment Analysis AI Engine")

# CORS Middleware (Frontend API call issue fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template Setup
templates = Jinja2Templates(directory="templates")

# Sentiment Labels Map
LABELS_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


# --- Dummy AI Inference Engine (Replace with your actual ML model code) ---
def dummy_predict_sentiment(text: str, aspect: str = ""):
    """Mock ML prediction function returning class probabilities."""
    text_lower = text.lower()

    # Rule-based fallback logic for demonstration
    if any(
        w in text_lower
        for w in ["good", "great", "excellent", "love", "awesome", "best"]
    ):
        probs = [0.05, 0.15, 0.80]
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
        probs = [0.85, 0.10, 0.05]
    else:
        probs = [0.20, 0.60, 0.20]

    pred_idx = int(np.argmax(probs))
    sentiment = LABELS_MAP[pred_idx]
    confidence = round(float(probs[pred_idx]) * 100, 2)

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "breakdown": {
            "Positive": round(probs[2] * 100, 1),
            "Neutral": round(probs[1] * 100, 1),
            "Negative": round(probs[0] * 100, 1),
        },
    }


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_single(
    app_name: str = Form(""),
    aspect: str = Form(""),
    review: str = Form(...),
):
    if not review.strip():
        return JSONResponse(
            status_code=400, content={"error": "Review text cannot be empty"}
        )

    res = dummy_predict_sentiment(review, aspect)
    return JSONResponse(content=res)


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename.lower()

        # Reading Dataset based on extension
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Unsupported file format. Please upload a .csv or .xlsx file."
                },
            )

        if df.empty:
            return JSONResponse(
                status_code=400, content={"error": "The uploaded file is empty."}
            )

        # Flexible Review Column Identification
        review_col = None
        for col in df.columns:
            if re.search(
                r"(review|comment|text|feedback|description|content)",
                str(col),
                re.IGNORECASE,
            ):
                review_col = col
                break

        # Fallback to first string/object column if no named review column found
        if review_col is None:
            string_cols = df.select_dtypes(include=["object"]).columns
            if len(string_cols) > 0:
                review_col = string_cols[0]
            else:
                review_col = df.columns[0]

        predictions = []

        # Predict sentiment row by row
        for val in df[review_col]:
            text_str = str(val) if pd.notnull(val) else ""
            res = dummy_predict_sentiment(text_str)
            # Storing string sentiment label directly to avoid index errors
            predictions.append(res["sentiment"])

        # Safely assign predicted sentiment list to DataFrame
        df["Predicted_Sentiment"] = predictions

        # Summary Metrics Calculation
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
            status_code=500, content={"error": f"Error processing file: {str(e)}"}
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)