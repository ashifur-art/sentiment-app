import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def train():
  print("Loading dataset...")
  file_path = "Merged_Final_Dataset1 (2).xlsx"
  df = pd.read_excel(file_path)

  # কলামগুলোর খালি ডাটা বাদ দেওয়া
  df.dropna(
      subset=["App Name", "Aspect", "clean_review", "Sentiments"], inplace=True
  )

  # Context Framing: App Name, Aspect এবং Review একসাথে কম্বাইন করা
  print("Formatting feature inputs (App Name + Aspect + Review)...")
  df["text"] = (
      "App Name: "
      + df["App Name"].astype(str)
      + " | Target Aspect: "
      + df["Aspect"].astype(str)
      + " | User Review: "
      + df["clean_review"].astype(str)
  )

  X = df["text"]
  y = df["Sentiments"]

  # Train-Test Split (80% Train, 20% Test)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Vectorization using TF-IDF
  print("Extracting features with TF-IDF...")
  tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
  X_train_tfidf = tfidf.fit_transform(X_train)
  X_test_tfidf = tfidf.transform(X_test)

  # Model Training
  print("Training Logistic Regression Model...")
  model = LogisticRegression(max_iter=1000, C=1.5, random_state=42)
  model.fit(X_train_tfidf, y_train)

  # Evaluation Report
  y_pred = model.predict(X_test_tfidf)
  print("\n--- Model Evaluation Report ---")
  print(classification_report(y_test, y_pred))

  # Save Pickled Models (app.py এর সাথে মিল রেখে)
  joblib.dump(model, "absa_model.pkl")
  joblib.dump(tfidf, "tfidf_vectorizer.pkl")

  print(
      "\n✅ Model and Vectorizer saved successfully as 'absa_model.pkl' and"
      " 'tfidf_vectorizer.pkl'!"
  )


if __name__ == "__main__":
  train()