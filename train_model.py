import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/jobs.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("AI RESUME ANALYZER - MODEL TRAINING")
print("=" * 60)

print("\n[1/7] Loading dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please create data/jobs.csv first."
    )


df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded successfully: "
    f"{len(df)} records"
)


# ============================================================
# VALIDATE DATASET
# ============================================================

print("\n[2/7] Validating dataset...")

required_columns = [
    "job_title",
    "description",
    "skills"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# Remove empty rows

df = df.dropna(
    subset=[
        "job_title",
        "description",
        "skills"
    ]
)


# ============================================================
# CREATE TRAINING TEXT
# ============================================================

print("\n[3/7] Preparing training data...")


# Combine job description and skills.
#
# Example:
#
# Description:
# "Build machine learning models"
#
# Skills:
# "Python, Pandas, Scikit-learn"
#
# Combined:
# "Build machine learning models
#  Python Pandas Scikit-learn"

df["training_text"] = (
    df["description"].astype(str)
    + " "
    + df["skills"].astype(str)
)


X = df["training_text"]

y = df["job_title"]


print(
    f"Number of samples: {len(df)}"
)

print(
    f"Number of job roles: {y.nunique()}"
)

print("\nJob roles:")

for role in sorted(y.unique()):

    count = (y == role).sum()

    print(
        f"  • {role}: {count} samples"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n[4/7] Splitting dataset...")

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

except ValueError:

    # Fallback for very small datasets

    print(
        "Warning: Dataset is too small for stratified splitting."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

print("\n[5/7] Creating TF-IDF features...")


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    f"Training feature matrix: "
    f"{X_train_tfidf.shape}"
)


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n[6/7] Training ML classifier...")


model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


model.fit(
    X_train_tfidf,
    y_train
)


print(
    "Model training completed."
)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


y_pred = model.predict(
    X_test_tfidf
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n[7/7] Saving trained model...")


joblib.dump(
    model,
    MODEL_PATH
)


joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


print(
    f"\nClassifier saved to:\n"
    f"  {MODEL_PATH}"
)


print(
    f"\nTF-IDF vectorizer saved to:\n"
    f"  {VECTORIZER_PATH}"
)


# ============================================================
# FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print(
    f"Algorithm: Logistic Regression"
)

print(
    f"Features: TF-IDF"
)

print(
    f"Vocabulary size: {len(vectorizer.vocabulary_)}"
)

print(
    f"Number of classes: {len(model.classes_)}"
)

print(
    "\nClasses:"
)

for class_name in model.classes_:

    print(
        f"  • {class_name}"
    )


print("\n" + "=" * 60)

print(
    "TRAINING COMPLETE ✅"
)

print("=" * 60)

print(
    "\nYou can now run:"
)

print(
    "    streamlit run app.py"
)