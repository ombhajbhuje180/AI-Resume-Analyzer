"""
recommender.py

Purpose:
    Generate career and skill recommendations based on:

    1. Candidate skills
    2. Target job role
    3. Job matching results
    4. Missing skills

This module does NOT:
    - Read resumes
    - Train ML models
    - Calculate job similarity
    - Handle Streamlit UI
"""

from skill_extractor import (
    JOB_ROLE_SKILLS,
    format_skill_name,
    find_missing_skills,
    calculate_skill_match
)


# ============================================================
# LEARNING RECOMMENDATION DATABASE
# ============================================================

LEARNING_RESOURCES = {

    # --------------------------------------------------------
    # Programming
    # --------------------------------------------------------

    "python": (
        "Strengthen Python fundamentals, OOP, "
        "file handling, APIs and advanced programming."
    ),

    "java": (
        "Improve Java fundamentals, OOP, collections "
        "and backend development."
    ),

    "javascript": (
        "Strengthen JavaScript fundamentals, ES6+, "
        "async programming and browser APIs."
    ),

    "c++": (
        "Practice C++ STL, OOP, algorithms and "
        "competitive programming concepts."
    ),


    # --------------------------------------------------------
    # Data Science
    # --------------------------------------------------------

    "pandas": (
        "Practice advanced Pandas operations including "
        "groupby, merge, pivot tables and data cleaning."
    ),

    "numpy": (
        "Strengthen NumPy arrays, broadcasting, "
        "vectorization and numerical operations."
    ),

    "statistics": (
        "Learn descriptive statistics, probability, "
        "hypothesis testing and statistical inference."
    ),

    "data visualization": (
        "Practice Matplotlib, Seaborn and Plotly "
        "for professional data visualization."
    ),


    # --------------------------------------------------------
    # Machine Learning
    # --------------------------------------------------------

    "machine learning": (
        "Study supervised and unsupervised learning, "
        "model evaluation, feature engineering and "
        "cross-validation."
    ),

    "scikit-learn": (
        "Practice Scikit-learn pipelines, preprocessing, "
        "model selection and evaluation."
    ),

    "deep learning": (
        "Learn neural networks, CNNs, RNNs, "
        "backpropagation and deep learning architectures."
    ),

    "tensorflow": (
        "Practice TensorFlow model building, training, "
        "evaluation and deployment."
    ),

    "pytorch": (
        "Learn PyTorch tensors, neural networks, "
        "training loops and model optimization."
    ),

    "natural language processing": (
        "Study text preprocessing, embeddings, "
        "transformers and NLP model evaluation."
    ),


    # --------------------------------------------------------
    # Databases
    # --------------------------------------------------------

    "sql": (
        "Practice joins, subqueries, CTEs, window functions, "
        "indexes and query optimization."
    ),

    "mysql": (
        "Strengthen MySQL schema design, joins, indexes, "
        "stored procedures and optimization."
    ),

    "postgresql": (
        "Learn PostgreSQL indexing, transactions, "
        "CTEs, window functions and advanced SQL."
    ),

    "mongodb": (
        "Practice MongoDB documents, aggregation pipelines, "
        "indexes and schema design."
    ),


    # --------------------------------------------------------
    # Cloud
    # --------------------------------------------------------

    "aws": (
        "Learn AWS fundamentals including EC2, S3, IAM, "
        "Lambda and basic cloud architecture."
    ),

    "azure": (
        "Learn Azure compute, storage, networking "
        "and identity services."
    ),

    "google cloud": (
        "Study GCP compute, storage, IAM and "
        "machine learning services."
    ),


    # --------------------------------------------------------
    # DevOps
    # --------------------------------------------------------

    "docker": (
        "Learn Docker images, containers, Dockerfiles, "
        "volumes and container networking."
    ),

    "kubernetes": (
        "Study Kubernetes pods, deployments, services "
        "and container orchestration."
    ),

    "git": (
        "Practice Git branching, merging, pull requests "
        "and collaborative development."
    ),

    "linux": (
        "Strengthen Linux commands, permissions, processes "
        "and shell scripting."
    ),


    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    "artificial intelligence": (
        "Study AI fundamentals, search, reasoning, "
        "machine learning and intelligent systems."
    ),

    "large language models": (
        "Learn LLM architecture, embeddings, "
        "fine-tuning and evaluation."
    ),

    "llm": (
        "Learn LLM APIs, prompt engineering, embeddings "
        "and retrieval-based applications."
    ),

    "generative ai": (
        "Build applications using LLMs, embeddings, "
        "RAG and generative AI APIs."
    ),

    "prompt engineering": (
        "Practice structured prompting, few-shot prompting, "
        "tool use and output evaluation."
    ),

    "rag": (
        "Learn document chunking, embeddings, vector databases "
        "and Retrieval-Augmented Generation."
    ),


    # --------------------------------------------------------
    # Big Data
    # --------------------------------------------------------

    "spark": (
        "Learn Apache Spark DataFrames, transformations, "
        "actions and distributed processing."
    ),

    "hadoop": (
        "Study HDFS, MapReduce, YARN and Hadoop architecture."
    ),

    "data engineering": (
        "Learn ETL pipelines, data warehouses, "
        "orchestration and data processing."
    )
}


# ============================================================
# NORMALIZE SKILL NAME
# ============================================================

def normalize_skill(skill):
    """
    Normalize a skill for comparison.
    """

    if not skill:

        return ""

    return (
        str(skill)
        .lower()
        .strip()
    )


# ============================================================
# FIND TARGET JOB
# ============================================================

def determine_target_job(
    job_matches
):
    """
    Determine the highest-ranked job from match results.

    Parameters
    ----------
    job_matches : pandas.DataFrame

    Returns
    -------
    str or None
    """

    if job_matches is None:

        return None


    if getattr(
        job_matches,
        "empty",
        True
    ):

        return None


    possible_columns = [
        "job_title",
        "title",
        "role",
        "job_role"
    ]


    for column in possible_columns:

        if column in job_matches.columns:

            return job_matches.iloc[0][
                column
            ]


    return None


# ============================================================
# GET TARGET JOB SKILLS
# ============================================================

def get_required_skills(
    target_job
):
    """
    Return required skills for a target role.
    """

    if not target_job:

        return []


    return JOB_ROLE_SKILLS.get(
        target_job,
        []
    )


# ============================================================
# FIND SKILL GAPS
# ============================================================

def get_skill_gaps(
    candidate_skills,
    target_job
):
    """
    Find missing skills required for the target role.
    """

    return find_missing_skills(
        candidate_skills,
        target_job
    )


# ============================================================
# CALCULATE SKILL COVERAGE
# ============================================================

def get_skill_coverage(
    candidate_skills,
    target_job
):
    """
    Calculate how much of the target role's
    required skill set the candidate already has.
    """

    return calculate_skill_match(
        candidate_skills,
        target_job
    )


# ============================================================
# GENERATE INDIVIDUAL RECOMMENDATION
# ============================================================

def get_skill_recommendation(
    skill
):
    """
    Generate a learning recommendation for one skill.
    """

    normalized = normalize_skill(
        skill
    )


    # Try exact match

    if normalized in LEARNING_RESOURCES:

        return LEARNING_RESOURCES[
            normalized
        ]


    # Fallback recommendation

    return (
        f"Develop practical experience with "
        f"{format_skill_name(skill)} through "
        f"projects, tutorials and hands-on practice."
    )


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    candidate_skills,
    job_matches
):
    """
    Generate personalized career recommendations.

    Parameters
    ----------
    candidate_skills : list
        Skills detected from the resume.

    job_matches : pandas.DataFrame
        Ranked job matches.

    Returns
    -------
    list
        Recommendation strings.
    """

    recommendations = []


    # --------------------------------------------------------
    # Determine target role
    # --------------------------------------------------------

    target_job = determine_target_job(
        job_matches
    )


    if not target_job:

        return [
            "Add more technical skills and project "
            "experience to improve job matching."
        ]


    # --------------------------------------------------------
    # Calculate skill coverage
    # --------------------------------------------------------

    skill_coverage = get_skill_coverage(
        candidate_skills,
        target_job
    )


    # --------------------------------------------------------
    # Skill coverage recommendation
    # --------------------------------------------------------

    if skill_coverage >= 80:

        recommendations.append(
            f"Your current skills cover approximately "
            f"{skill_coverage:.0f}% of the required skills "
            f"for {target_job}. Focus on advanced projects "
            f"and real-world implementation."
        )

    elif skill_coverage >= 50:

        recommendations.append(
            f"You currently cover approximately "
            f"{skill_coverage:.0f}% of the required skills "
            f"for {target_job}. Focus on closing the "
            f"remaining skill gaps."
        )

    else:

        recommendations.append(
            f"Your current skill coverage is approximately "
            f"{skill_coverage:.0f}% for {target_job}. "
            f"Build foundational knowledge and projects "
            f"in the missing areas."
        )


    # --------------------------------------------------------
    # Find missing skills
    # --------------------------------------------------------

    missing_skills = get_skill_gaps(
        candidate_skills,
        target_job
    )


    # --------------------------------------------------------
    # Generate recommendations for missing skills
    # --------------------------------------------------------

    for skill in missing_skills[:5]:

        recommendation = get_skill_recommendation(
            skill
        )


        recommendations.append(
            f"Learn {skill}: {recommendation}"
        )


    # --------------------------------------------------------
    # Project recommendation
    # --------------------------------------------------------

    if missing_skills:

        recommendations.append(
            f"Build a practical {target_job} project "
            f"that demonstrates {', '.join(missing_skills[:3])}."
        )

    else:

        recommendations.append(
            f"Build an advanced portfolio project "
            f"focused on {target_job} to demonstrate "
            f"your existing skills."
        )


    # --------------------------------------------------------
    # Resume recommendation
    # --------------------------------------------------------

    recommendations.append(
        "Quantify project achievements in your resume "
        "using measurable results such as accuracy, "
        "performance improvements, dataset size or "
        "processing time."
    )


    return recommendations


# ============================================================
# GENERATE SKILL GAP REPORT
# ============================================================

def generate_skill_gap_report(
    candidate_skills,
    target_job
):
    """
    Generate a structured skill gap report.
    """

    required_skills = get_required_skills(
        target_job
    )


    missing_skills = get_skill_gaps(
        candidate_skills,
        target_job
    )


    candidate_normalized = {
        normalize_skill(skill)
        for skill in candidate_skills
    }


    matched_skills = []


    for required in required_skills:

        if normalize_skill(
            required
        ) in candidate_normalized:

            matched_skills.append(
                format_skill_name(
                    required
                )
            )


    return {
        "target_job": target_job,
        "required_skills": [
            format_skill_name(skill)
            for skill in required_skills
        ],
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "coverage": get_skill_coverage(
            candidate_skills,
            target_job
        )
    }


# ============================================================
# CAREER SUMMARY
# ============================================================

def generate_career_summary(
    candidate_skills,
    job_matches
):
    """
    Generate a short summary for the dashboard.
    """

    target_job = determine_target_job(
        job_matches
    )


    if not target_job:

        return {
            "target_role": "Not determined",
            "skill_coverage": 0,
            "missing_skills": [],
            "summary": (
                "More resume information is required "
                "for a reliable job recommendation."
            )
        }


    coverage = get_skill_coverage(
        candidate_skills,
        target_job
    )


    missing = get_skill_gaps(
        candidate_skills,
        target_job
    )


    summary = (
        f"The candidate currently matches "
        f"{coverage:.0f}% of the core skills identified "
        f"for the {target_job} role."
    )


    if missing:

        summary += (
            f" Key development areas include "
            f"{', '.join(missing[:3])}."
        )


    else:

        summary += (
            " The candidate has the core skills "
            "required for this role."
        )


    return {
        "target_role": target_job,
        "skill_coverage": coverage,
        "missing_skills": missing,
        "summary": summary
    }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "RECOMMENDER MODULE"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Sample candidate
    # --------------------------------------------------------

    candidate_skills = [
        "Python",
        "Pandas",
        "NumPy",
        "SQL",
        "Machine Learning",
        "Scikit-learn"
    ]


    target_job = "Data Scientist"


    print(
        f"\nTarget Role: {target_job}"
    )


    # --------------------------------------------------------
    # Skill gap report
    # --------------------------------------------------------

    report = generate_skill_gap_report(
        candidate_skills,
        target_job
    )


    print(
        "\nSkill Coverage:"
    )

    print(
        f"{report['coverage']}%"
    )


    print(
        "\nMatched Skills:"
    )

    for skill in report[
        "matched_skills"
    ]:

        print(
            f"  ✓ {skill}"
        )


    print(
        "\nMissing Skills:"
    )

    for skill in report[
        "missing_skills"
    ]:

        print(
            f"  ✗ {skill}"
        )


    # --------------------------------------------------------
    # Mock job matches
    # --------------------------------------------------------

    import pandas as pd


    mock_matches = pd.DataFrame(
        {
            "job_title": [
                "Data Scientist",
                "Data Analyst",
                "ML Engineer"
            ],

            "match_score": [
                91.5,
                82.3,
                78.6
            ]
        }
    )


    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendations = generate_recommendations(
        candidate_skills,
        mock_matches
    )


    print(
        "\nRecommendations:"
    )


    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{index}. {recommendation}"
        )


    # --------------------------------------------------------
    # Career summary
    # --------------------------------------------------------

    summary = generate_career_summary(
        candidate_skills,
        mock_matches
    )


    print(
        "\nCareer Summary:"
    )

    print(
        summary["summary"]
    )


    print(
        "\nRecommender ready ✅"
    )

    print("=" * 60)