"""
text_processor.py

Purpose:
    Clean and preprocess resume text before it is passed
    to the skill extraction and machine learning modules.

Pipeline:

    Raw Resume Text
          ↓
    Normalize Text
          ↓
    Remove Noise
          ↓
    Clean Whitespace
          ↓
    Processed Text
"""

import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean and normalize resume text.

    Parameters
    ----------
    text : str
        Raw text extracted from a resume.

    Returns
    -------
    str
        Cleaned text.
    """

    if not text:
        return ""

    # --------------------------------------------------------
    # Convert to string
    # --------------------------------------------------------

    text = str(text)

    # --------------------------------------------------------
    # Convert to lowercase
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # Replace common separators with spaces
    #
    # Example:
    # Python|SQL|Pandas
    #
    # becomes:
    # Python SQL Pandas
    # --------------------------------------------------------

    text = re.sub(
        r"[|•●▪►→]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Preserve useful programming symbols
    #
    # We intentionally DON'T remove:
    #
    # C++
    # C#
    # .NET
    # Node.js
    #
    # because they can be important skills.
    # --------------------------------------------------------

    # Replace URLs

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Replace email addresses

    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove unwanted punctuation
    #
    # Keep:
    # + # . -
    #
    # because they can appear in:
    # C++
    # C#
    # .NET
    # Node.js
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-zA-Z0-9+#.\-\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove standalone punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"(?<![a-zA-Z0-9])[.\-](?![a-zA-Z0-9])",
        " ",
        text
    )

    # --------------------------------------------------------
    # Normalize repeated dots
    # --------------------------------------------------------

    text = re.sub(
        r"\.{2,}",
        ".",
        text
    )

    # --------------------------------------------------------
    # Normalize repeated hyphens
    # --------------------------------------------------------

    text = re.sub(
        r"-{2,}",
        "-",
        text
    )

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Strip leading/trailing spaces
    # --------------------------------------------------------

    text = text.strip()

    return text


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize_text(text):
    """
    Convert cleaned text into individual tokens.

    Example:

        "python data science sql"

    becomes:

        ["python", "data", "science", "sql"]
    """

    if not text:
        return []

    return text.split()


# ============================================================
# REMOVE STOP WORDS
# ============================================================

def remove_stopwords(text):
    """
    Remove common English stop words.

    This is mainly useful for ML text representation.

    Example:

        "I am a Python developer"

    becomes approximately:

        "python developer"
    """

    if not text:
        return ""

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "we",
        "were",
        "with",
        "you",
        "your"
    }

    tokens = tokenize_text(text)

    filtered_tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(filtered_tokens)


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_resume(text):
    """
    Run the complete preprocessing pipeline.

    Steps:
        1. Clean text
        2. Remove stop words
        3. Normalize whitespace

    Returns
    -------
    str
        Final processed text.
    """

    text = clean_text(text)

    text = remove_stopwords(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TF-IDF VECTORIZER
# ============================================================

def create_tfidf_vectorizer(
    max_features=5000,
    ngram_range=(1, 2)
):
    """
    Create a TF-IDF vectorizer.

    Parameters
    ----------
    max_features : int
        Maximum number of vocabulary features.

    ngram_range : tuple
        Range of n-grams.

        (1,1) → single words
        (1,2) → single words + two-word phrases

    Returns
    -------
    TfidfVectorizer
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True
    )

    return vectorizer


# ============================================================
# CREATE TF-IDF FEATURES
# ============================================================

def create_tfidf(
    documents,
    max_features=5000,
    ngram_range=(1, 2)
):
    """
    Convert text documents into TF-IDF features.

    Parameters
    ----------
    documents : list
        List of text documents.

    Returns
    -------
    vectorizer
        Trained TF-IDF vectorizer.

    matrix
        TF-IDF feature matrix.
    """

    vectorizer = create_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    return vectorizer, matrix


# ============================================================
# TRANSFORM NEW TEXT
# ============================================================

def transform_text(
    vectorizer,
    text
):
    """
    Transform new text using an already trained
    TF-IDF vectorizer.

    Important:
        We use transform(), NOT fit_transform(),
        for new resume data.
    """

    processed_text = preprocess_resume(
        text
    )

    matrix = vectorizer.transform(
        [processed_text]
    )

    return matrix


# ============================================================
# TEXT STATISTICS
# ============================================================

def get_text_statistics(text):
    """
    Return basic statistics about the text.
    """

    if not text:

        return {
            "characters": 0,
            "words": 0,
            "sentences": 0
        }

    characters = len(text)

    words = len(
        text.split()
    )

    sentences = len(
        re.findall(
            r"[.!?]+",
            text
        )
    )

    return {
        "characters": characters,
        "words": words,
        "sentences": sentences
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "TEXT PROCESSOR MODULE"
    )

    print("=" * 60)


    sample_text = """
    John Doe is a Python Developer with experience
    in Machine Learning, Pandas, NumPy, SQL and
    Scikit-learn.

    Email: john@example.com

    Portfolio: https://example.com
    """


    print("\nOriginal text:")
    print(sample_text)


    cleaned = clean_text(
        sample_text
    )


    print("\nCleaned text:")
    print(cleaned)


    processed = preprocess_resume(
        sample_text
    )


    print("\nProcessed text:")
    print(processed)


    print("\nTokens:")

    print(
        tokenize_text(processed)
    )


    print("\nStatistics:")

    print(
        get_text_statistics(
            sample_text
        )
    )


    print("\nCreating TF-IDF...")

    documents = [
        "python machine learning pandas numpy",
        "java spring boot backend development",
        "sql power bi data analytics",
        "tensorflow deep learning neural networks"
    ]


    vectorizer, matrix = create_tfidf(
        documents
    )


    print(
        f"TF-IDF matrix shape: {matrix.shape}"
    )


    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_)}"
    )


    print(
        "\nText processor ready ✅"
    )

    print("=" * 60)