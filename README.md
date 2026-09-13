# 🚀 Aspect-Based Sentiment Analysis (ABSA) for Language Learning Apps

An end-to-end, state-of-the-art **Aspect-Based Sentiment Analysis (ABSA)** application designed to predict fine-grained sentiments (Positive, Neutral, Negative) along with confidence probability distributions for specific target aspects within user reviews.

---

## 📌 1. Project Overview & Background

Traditional Sentiment Analysis assigns a single sentiment label to an entire review. However, user reviews are often complex and contain mixed opinions. 

*Example Review:* **"The lessons are great, but the subscription cost is too high."**
- **Lessons (Aspect):** Positive 🟢
- **Subscription (Aspect):** Negative 🔴

This project solves this limitation by introducing a **Context-Aware Aspect-Based Sentiment Analysis** pipeline trained on a real-world dataset of **6,174 reviews** collected across 10 popular Language Learning Applications (e.g., Duolingo, Babbel, Busuu, Memrise).

---

## 📊 2. Dataset Specification

- **Total Records:** 6,174 rows
- **Columns:**
  - `App Name`: The application name (e.g., Duolingo, Memrise)
  - `clean_review`: Preprocessed textual user review
  - `Aspect`: The specific aspect category being evaluated (16 unique categories)
  - `Sentiments`: Target sentiment label (`Positive`, `Neutral`, `Negative`)
- **Class Distribution:**
  - **Positive:** 2,569 instances
  - **Neutral:** 1,870 instances
  - **Negative:** 1,735 instances

---

## 🛠️ 3. Input Engineering & Preprocessing Pipeline

To enable models to evaluate sentiments *conditioned* on a specific target aspect and app, we implemented a **Context Framing Transformation**:

$$\text{Input String} = \text{"App Name: \{App\} \vert{} Target Aspect: \{Aspect\} \vert{} User Review: \{Review\}"}$$

### Preprocessing Steps:
1. **Text Sanitization:** Removal of irrelevant HTML tags, unusual emojis, and redundant spaces.
2. **Context Reconstruction:** Concatenating `App Name`, `Aspect`, and `clean_review`.
3. **Feature Extraction (for Baseline ML Models):** `TfidfVectorizer` (max_features=5000, unigrams + bigrams).
4. **Subword Tokenization (for Transformer Models):** Pre-trained subword tokenizers matching model architectures (e.g., DistilBERT, BERT-base).

---

## 🧠 4. Model Training & Performance Evaluation

We conducted extensive benchmark evaluations comparing **Deep Learning Transformer Architectures** against **13 Traditional & Ensemble Machine Learning Models**.

### 🌟 High-Performing Transformer Models (Primary Benchmark)

| Model Architecture | Accuracy | Precision | Recall | Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Base** | **81.12%** | **80.07%** | **79.82%** | **79.90%** | **Selected Winner (Production)** |
| **BERT + BiGRU Layer** | **81.12%** | **80.14%** | **79.80%** | **79.93%** | Benchmark Leader |

---

### 📉 Traditional & Ensemble ML Model Experiments (13 Baseline Models)

All 13 ML models were trained using **TF-IDF Feature Representation** with an 85/15 Stratified Train-Test split.

| # | Model Name | Accuracy | Precision | Recall | Macro F1 |
| - | :--- | :---: | :---: | :---: | :---: |
| 1 | **Logistic Regression** | **71.31%** | 70.37% | 68.81% | **69.07%** |
| 2 | **SGD Classifier (Log Loss)** | **71.09%** | 69.95% | 68.51% | 68.71% |
| 3 | **Support Vector Machine (RBF SVC)** | **70.87%** | 69.95% | 67.97% | 68.15% |
| 4 | **Gradient Boosting Classifier** | 70.01% | 69.80% | 66.70% | 66.93% |
| 5 | **Linear Support Vector (LinearSVC)** | 69.04% | 67.33% | 66.64% | 66.72% |
| 6 | **Extra Trees Classifier** | 68.18% | 67.05% | 64.89% | 64.63% |
| 7 | **Random Forest Classifier** | 67.31% | 66.16% | 64.00% | 63.82% |
| 8 | **AdaBoost Classifier** | 67.21% | 65.26% | 64.92% | 64.98% |
| 9 | **Multi-Layer Perceptron (MLP)** | 66.45% | 64.58% | 64.14% | 64.23% |
| 10 | **Multinomial Naive Bayes** | 65.37% | 65.14% | 61.69% | 61.56% |
| 11 | **Bernoulli Naive Bayes** | 61.92% | 61.55% | 58.02% | 57.79% |
| 12 | **Decision Tree Classifier** | 61.70% | 59.81% | 59.86% | 59.81% |
| 13 | **K-Nearest Neighbors (KNN)** | 56.31% | 56.42% | 54.81% | 54.84% |

---

## 📁 5. Repository File Structure
absa-project/
├── dataset/
│   └── Merged_Final_Dataset1 (2).xlsx   # Dataset File
├── saved_models/
│   ├── absa_model.pkl                   # Saved Model Artifacts
│   └── tfidf_vectorizer.pkl             # TF-IDF Vectorizer Artifacts
├── templates/
│   └── index.html                       # Frontend Interactive UI
├── app.py                               # FastAPI Web Server Application
├── train_and_save.py                    # Training & Model Serialization Script
├── requirements.txt                     # Python Package Dependencies
└── README.md                            # Documentation


---

## 🏃 6. Detailed Project Execution & Run Guide (Step-by-Step)

Follow these detailed steps to set up, train, and run the project locally on your machine using **VS Code**.

### 🔹 Step 1: Open Project in VS Code
Open VS Code, go to `File` -> `Open Folder...`, and select your project root folder (`absa-project`). 
Open the built-in Terminal (`Ctrl + ~` on Windows or `Cmd + ~` on Mac).

### 🔹 Step 2: Create & Activate Virtual Environment
Creating an isolated Python virtual environment prevents package conflicts.

* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  uvicorn app:app --reload --port 8000