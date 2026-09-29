"""
job_matcher.py

Purpose:
    Match a candidate resume against available jobs.

Approach:
    TF-IDF + Cosine Similarity

Input:
    Resume text
    data/jobs.csv

Output:
    Ranked job matches with similarity scores.

This module does NOT:
    - Read PDF/DOCX files
    - Extract resume files
    - Train the ML classifier
    - Generate Streamlit UI
"""

import os

import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from text_processor import preprocess_resume


# ============================================================
# CONFIGURATION
# ============================================================

JOBS_PATH = "data/jobs.csv"

MODEL_DIR = "models"

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD JOB DATA
# ============================================================

def load_jobs(
    path=JOBS_PATH
):
    """
    Load available jobs from CSV.

    Expected columns:

        job_title
        description
        skills

    Returns
    -------
    pandas.DataFrame
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Job dataset not found: {path}"
        )


    jobs = pd.read_csv(path)


    required_columns = [
        "job_title",
        "description",
        "skills"
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in jobs.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing columns in jobs.csv: "
            + ", ".join(missing_columns)
        )


    # Remove rows with missing information

    jobs = jobs.dropna(
        subset=[
            "job_title",
            "description",
            "skills"
        ]
    ).copy()


    return jobs


# ============================================================
# PREPARE JOB TEXT
# ============================================================

def prepare_job_text(jobs):
    """
    Combine job description and skills
    into one text field.

    Example:

        Description:
        "Build machine learning models"

        Skills:
        "Python, SQL, Pandas"

        ↓

        Combined job text.
    """

    jobs = jobs.copy()


    jobs["job_text"] = (
        jobs["description"].astype(str)
        + " "
        + jobs["skills"].astype(str)
    )


    # Apply same preprocessing used for resumes

    jobs["processed_text"] = (
        jobs["job_text"]
        .apply(preprocess_resume)
    )


    return jobs


# ============================================================
# LOAD TRAINED TF-IDF VECTORIZER
# ============================================================

def load_vectorizer(
    path=VECTORIZER_PATH
):
    """
    Load the TF-IDF vectorizer created by train_model.py.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {path}\n"
            "Run train_model.py first."
        )


    vectorizer = joblib.load(
        path
    )


    return vectorizer


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

def calculate_similarity(
    resume_text,
    job_texts,
    vectorizer=None
):
    """
    Calculate cosine similarity between a resume
    and multiple job descriptions.

    Parameters
    ----------
    resume_text : str
        Processed resume text.

    job_texts : list
        List of processed job texts.

    vectorizer :
        Trained TF-IDF vectorizer.

    Returns
    -------
    list
        Similarity scores between 0 and 1.
    """

    if not resume_text:

        return []


    if not job_texts:

        return []


    # Load saved vectorizer if not provided

    if vectorizer is None:

        vectorizer = load_vectorizer()


    # Process resume

    processed_resume = preprocess_resume(
        resume_text
    )


    # Convert resume to TF-IDF

    resume_vector = vectorizer.transform(
        [processed_resume]
    )


    # Convert jobs to TF-IDF

    job_vectors = vectorizer.transform(
        job_texts
    )


    # Calculate cosine similarity

    similarities = cosine_similarity(
        resume_vector,
        job_vectors
    )


    return similarities[
        0
    ].tolist()


# ============================================================
# CONVERT SIMILARITY TO PERCENTAGE
# ============================================================

def similarity_to_percentage(
    similarity
):
    """
    Convert similarity value from 0-1
    to percentage.
    """

    percentage = float(
        similarity
    ) * 100


    return round(
        percentage,
        2
    )


# ============================================================
# MATCH JOBS
# ============================================================

def match_jobs(
    resume_text,
    top_n=10
):
    """
    Match a resume against all jobs.

    Parameters
    ----------
    resume_text : str
        Resume text.

    top_n : int
        Number of top matches to return.

    Returns
    -------
    pandas.DataFrame

        Ranked job matches.
    """

    # --------------------------------------------------------
    # Validate resume
    # --------------------------------------------------------

    if not resume_text:

        return pd.DataFrame()


    # --------------------------------------------------------
    # Load jobs
    # --------------------------------------------------------

    jobs = load_jobs()


    if jobs.empty:

        return pd.DataFrame()


    # --------------------------------------------------------
    # Prepare job text
    # --------------------------------------------------------

    jobs = prepare_job_text(
        jobs
    )


    # --------------------------------------------------------
    # Load vectorizer
    # --------------------------------------------------------

    vectorizer = load_vectorizer()


    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    similarities = calculate_similarity(
        resume_text,
        jobs["processed_text"].tolist(),
        vectorizer
    )


    # --------------------------------------------------------
    # Add similarity score
    # --------------------------------------------------------

    jobs["similarity"] = similarities


    jobs["match_score"] = (
        jobs["similarity"]
        .apply(similarity_to_percentage)
    )


    jobs["match_percentage"] = (
        jobs["match_score"]
    )


    # --------------------------------------------------------
    # Sort highest match first
    # --------------------------------------------------------

    jobs = jobs.sort_values(
        by="match_score",
        ascending=False
    )


    # --------------------------------------------------------
    # Keep top matches
    # --------------------------------------------------------

    jobs = jobs.head(
        top_n
    )


    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    jobs = jobs.reset_index(
        drop=True
    )


    return jobs


# ============================================================
# GET BEST JOB MATCH
# ============================================================

def get_best_match(
    resume_text
):
    """
    Return the single best matching job.
    """

    matches = match_jobs(
        resume_text,
        top_n=1
    )


    if matches.empty:

        return None


    return matches.iloc[0].to_dict()


# ============================================================
# GET TOP JOB TITLES
# ============================================================

def get_top_job_titles(
    resume_text,
    top_n=5
):
    """
    Return top matching job titles and scores.
    """

    matches = match_jobs(
        resume_text,
        top_n=top_n
    )


    if matches.empty:

        return []


    results = []


    for _, row in matches.iterrows():

        results.append(
            {
                "job_title": row["job_title"],
                "match_score": row["match_score"]
            }
        )


    return results


# ============================================================
# JOB MATCH SUMMARY
# ============================================================

def get_match_summary(
    resume_text
):
    """
    Generate a simple summary of job matches.
    """

    matches = match_jobs(
        resume_text,
        top_n=5
    )


    if matches.empty:

        return {
            "best_job": None,
            "best_score": 0,
            "total_jobs": 0
        }


    best_job = matches.iloc[0]


    return {
        "best_job": best_job["job_title"],
        "best_score": best_job["match_score"],
        "total_jobs": len(matches)
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "JOB MATCHER MODULE"
    )

    print("=" * 60)


    sample_resume = """
    Python developer with experience in data science,
    machine learning, pandas, numpy, SQL and
    scikit-learn.

    Built machine learning projects using Python
    and worked with data visualization.
    """


    print(
        "\nAnalyzing sample resume..."
    )


    try:

        matches = match_jobs(
            sample_resume,
            top_n=5
        )


        if matches.empty:

            print(
                "\nNo matches found."
            )

        else:

            print(
                "\nTop Job Matches:"
            )


            for index, row in matches.iterrows():

                print(
                    f"\n{index + 1}. "
                    f"{row['job_title']}"
                )

                print(
                    f"   Match: "
                    f"{row['match_score']}%"
                )


            print(
                "\n\nDetailed Results:"
            )

            print(
                matches[
                    [
                        "job_title",
                        "match_score"
                    ]
                ].to_string(
                    index=False
                )
            )


    except Exception as error:

        print(
            f"\nError: {error}"
        )


    print(
        "\nJob matcher ready ✅"
    )

    print("=" * 60)