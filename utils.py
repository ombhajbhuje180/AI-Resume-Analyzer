"""
utils.py

Purpose:
    Common reusable helper functions for the
    AI Resume Analyzer & Job Matcher.

This module contains:
    - File validation
    - Percentage formatting
    - Model loading
    - Data loading
    - Text helpers
    - Safe conversions

It should NOT contain:
    - ML training
    - Resume parsing
    - Skill extraction
    - Job matching
    - Streamlit UI logic
"""

import os
from typing import Any, Optional

import joblib
import pandas as pd


# ============================================================
# FILE UTILITIES
# ============================================================

def file_exists(path: str) -> bool:
    """
    Check whether a file exists.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    bool
        True if file exists, otherwise False.
    """

    return os.path.isfile(path)


def directory_exists(path: str) -> bool:
    """
    Check whether a directory exists.
    """

    return os.path.isdir(path)


def ensure_directory(path: str) -> str:
    """
    Create a directory if it does not already exist.

    Returns the same path.
    """

    os.makedirs(
        path,
        exist_ok=True
    )

    return path


def get_file_extension(filename: str) -> str:
    """
    Get the lowercase file extension.

    Example:

        resume.PDF → pdf
        candidate.docx → docx
    """

    if not filename:
        return ""

    return (
        filename
        .split(".")[-1]
        .lower()
    )


def is_supported_resume(filename: str) -> bool:
    """
    Check whether the uploaded file is a supported
    resume format.
    """

    extension = get_file_extension(
        filename
    )

    return extension in {
        "pdf",
        "docx"
    }


# ============================================================
# NUMBER / SCORE UTILITIES
# ============================================================

def clamp_percentage(
    value: float
) -> float:
    """
    Keep a percentage between 0 and 100.

    Example:

        120 → 100
        -5  → 0
        87  → 87
    """

    try:

        value = float(value)

    except (
        TypeError,
        ValueError
    ):

        return 0.0


    return max(
        0.0,
        min(
            100.0,
            value
        )
    )


def format_percentage(
    value: float,
    decimals: int = 1
) -> str:
    """
    Convert a numeric value into a percentage string.

    Example:

        91.456 → "91.5%"
    """

    value = clamp_percentage(
        value
    )

    return f"{value:.{decimals}f}%"


def similarity_to_percentage(
    similarity: float
) -> float:
    """
    Convert cosine similarity from 0-1
    into a 0-100 percentage.

    Example:

        0.91 → 91.0
    """

    try:

        similarity = float(
            similarity
        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0


    return round(
        clamp_percentage(
            similarity * 100
        ),
        2
    )


# ============================================================
# MODEL UTILITIES
# ============================================================

def load_pickle_model(
    path: str
) -> Any:
    """
    Load a joblib/pickle model.

    Used for:
        classifier.pkl
        tfidf_vectorizer.pkl
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model file not found: {path}"
        )


    try:

        return joblib.load(
            path
        )

    except Exception as error:

        raise RuntimeError(
            f"Unable to load model "
            f"'{path}': {error}"
        )


def save_pickle_model(
    model: Any,
    path: str
) -> None:
    """
    Save a model using joblib.
    """

    directory = os.path.dirname(
        path
    )


    if directory:

        ensure_directory(
            directory
        )


    joblib.dump(
        model,
        path
    )


# ============================================================
# DATA UTILITIES
# ============================================================

def load_csv(
    path: str
) -> pd.DataFrame:
    """
    Load a CSV file safely.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )


    try:

        return pd.read_csv(
            path
        )

    except Exception as error:

        raise RuntimeError(
            f"Unable to read CSV file "
            f"'{path}': {error}"
        )


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list
) -> None:
    """
    Check whether a DataFrame contains
    all required columns.

    Raises ValueError if columns are missing.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )


# ============================================================
# TEXT UTILITIES
# ============================================================

def safe_text(
    value: Any
) -> str:
    """
    Convert a value safely to text.

    None becomes an empty string.
    """

    if value is None:

        return ""

    return str(
        value
    ).strip()


def normalize_spaces(
    text: str
) -> str:
    """
    Replace repeated whitespace with
    a single space.
    """

    if not text:

        return ""

    return " ".join(
        str(text).split()
    )


def truncate_text(
    text: str,
    max_length: int = 200
) -> str:
    """
    Shorten text for UI display.

    Example:

        truncate_text(long_text, 50)
    """

    text = safe_text(
        text
    )


    if len(text) <= max_length:

        return text


    return (
        text[:max_length - 3]
        + "..."
    )


# ============================================================
# LIST UTILITIES
# ============================================================

def unique_list(
    items: list
) -> list:
    """
    Remove duplicate values while preserving order.
    """

    if not items:

        return []


    result = []

    seen = set()


    for item in items:

        key = str(
            item
        ).lower().strip()


        if key not in seen:

            seen.add(
                key
            )

            result.append(
                item
            )


    return result


def sort_skills(
    skills: list
) -> list:
    """
    Sort skills alphabetically.
    """

    if not skills:

        return []


    return sorted(
        unique_list(
            skills
        ),
        key=lambda x: str(
            x
        ).lower()
    )


# ============================================================
# JOB MATCH UTILITIES
# ============================================================

def get_best_match_score(
    job_matches: Optional[pd.DataFrame]
) -> float:
    """
    Get the highest job match score from
    a job matching DataFrame.
    """

    if job_matches is None:

        return 0.0


    if job_matches.empty:

        return 0.0


    possible_columns = [
        "match_score",
        "match_percentage",
        "score",
        "similarity"
    ]


    for column in possible_columns:

        if column in job_matches.columns:

            try:

                value = float(
                    job_matches[column].max()
                )

            except (
                TypeError,
                ValueError
            ):

                return 0.0


            # Convert 0-1 similarity
            # to percentage.

            if value <= 1:

                value *= 100


            return clamp_percentage(
                value
            )


    return 0.0


def get_best_job_title(
    job_matches: Optional[pd.DataFrame]
) -> Optional[str]:
    """
    Get the title of the highest-ranked job.
    """

    if job_matches is None:

        return None


    if job_matches.empty:

        return None


    possible_columns = [
        "job_title",
        "title",
        "role",
        "job_role"
    ]


    for column in possible_columns:

        if column in job_matches.columns:

            return str(
                job_matches.iloc[0][
                    column
                ]
            )


    return None


# ============================================================
# RESUME STATISTICS
# ============================================================

def get_word_count(
    text: str
) -> int:
    """
    Count words in text.
    """

    if not text:

        return 0


    return len(
        str(text).split()
    )


def get_character_count(
    text: str
) -> int:
    """
    Count characters in text.
    """

    if not text:

        return 0


    return len(
        str(text)
    )


# ============================================================
# ERROR HANDLING
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0
) -> float:
    """
    Safely convert a value to float.
    """

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return default


def safe_int(
    value: Any,
    default: int = 0
) -> int:
    """
    Safely convert a value to integer.
    """

    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# APPLICATION INFORMATION
# ============================================================

APP_NAME = (
    "AI Resume Analyzer & Job Matcher"
)

APP_VERSION = "1.0.0"

SUPPORTED_RESUME_FORMATS = [
    "pdf",
    "docx"
]


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "UTILITIES MODULE"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Percentage
    # --------------------------------------------------------

    print(
        "\nPercentage:"
    )

    print(
        format_percentage(
            91.456
        )
    )


    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    print(
        "\nSimilarity:"
    )

    print(
        similarity_to_percentage(
            0.874
        )
    )


    # --------------------------------------------------------
    # File extension
    # --------------------------------------------------------

    print(
        "\nFile Extension:"
    )

    print(
        get_file_extension(
            "resume.PDF"
        )
    )


    # --------------------------------------------------------
    # Supported file
    # --------------------------------------------------------

    print(
        "\nSupported Resume:"
    )

    print(
        is_supported_resume(
            "resume.pdf"
        )
    )


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    print(
        "\nText Utilities:"
    )

    sample = (
        "   Python    Machine Learning   SQL   "
    )

    print(
        normalize_spaces(
            sample
        )
    )


    # --------------------------------------------------------
    # Unique list
    # --------------------------------------------------------

    print(
        "\nUnique List:"
    )

    skills = [
        "Python",
        "SQL",
        "Python",
        "Pandas",
        "SQL"
    ]

    print(
        unique_list(
            skills
        )
    )


    # --------------------------------------------------------
    # Constants
    # --------------------------------------------------------

    print(
        "\nApplication:"
    )

    print(
        APP_NAME
    )

    print(
        APP_VERSION
    )


    print(
        "\nUtils module ready ✅"
    )

    print("=" * 60)