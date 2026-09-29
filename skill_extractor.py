"""
skill_extractor.py

Purpose:
    Extract technical/professional skills from resume text
    and identify missing skills for a target job role.

Approach:
    Curated skill dictionary + text matching.

This module does NOT:
    - Read PDF/DOCX files
    - Train ML models
    - Calculate job similarity
    - Generate the UI
"""

import re


# ============================================================
# SKILL DATABASE
# ============================================================

SKILL_DATABASE = {

    # --------------------------------------------------------
    # Programming Languages
    # --------------------------------------------------------

    "programming": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "c",
        "go",
        "golang",
        "rust",
        "php",
        "ruby",
        "kotlin",
        "swift"
    ],


    # --------------------------------------------------------
    # Data Science
    # --------------------------------------------------------

    "data_science": [
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "jupyter",
        "data analysis",
        "data analytics",
        "data visualization",
        "statistics",
        "statistical analysis"
    ],


    # --------------------------------------------------------
    # Machine Learning
    # --------------------------------------------------------

    "machine_learning": [
        "machine learning",
        "deep learning",
        "scikit-learn",
        "sklearn",
        "tensorflow",
        "pytorch",
        "keras",
        "xgboost",
        "lightgbm",
        "random forest",
        "logistic regression",
        "linear regression",
        "decision tree",
        "neural network",
        "natural language processing",
        "nlp",
        "computer vision",
        "reinforcement learning"
    ],


    # --------------------------------------------------------
    # Databases
    # --------------------------------------------------------

    "databases": [
        "sql",
        "mysql",
        "postgresql",
        "postgres",
        "mongodb",
        "sqlite",
        "oracle",
        "redis",
        "firebase",
        "supabase",
        "database management",
        "dbms"
    ],


    # --------------------------------------------------------
    # Web Development
    # --------------------------------------------------------

    "web_development": [
        "html",
        "css",
        "javascript",
        "react",
        "react.js",
        "next.js",
        "node.js",
        "node",
        "express",
        "express.js",
        "tailwind",
        "tailwind css",
        "rest api",
        "restful api",
        "api development"
    ],


    # --------------------------------------------------------
    # Cloud
    # --------------------------------------------------------

    "cloud": [
        "aws",
        "amazon web services",
        "azure",
        "microsoft azure",
        "google cloud",
        "gcp",
        "cloud computing",
        "cloud deployment",
        "ec2",
        "s3",
        "lambda"
    ],


    # --------------------------------------------------------
    # DevOps
    # --------------------------------------------------------

    "devops": [
        "docker",
        "kubernetes",
        "jenkins",
        "github actions",
        "ci/cd",
        "cicd",
        "terraform",
        "ansible",
        "linux",
        "git",
        "github",
        "gitlab"
    ],


    # --------------------------------------------------------
    # Big Data
    # --------------------------------------------------------

    "big_data": [
        "hadoop",
        "spark",
        "apache spark",
        "hdfs",
        "hive",
        "kafka",
        "big data",
        "data engineering",
        "etl"
    ],


    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    "artificial_intelligence": [
        "artificial intelligence",
        "generative ai",
        "genai",
        "large language models",
        "llm",
        "prompt engineering",
        "transformers",
        "bert",
        "gpt",
        "rag",
        "retrieval augmented generation",
        "computer vision",
        "speech recognition"
    ],


    # --------------------------------------------------------
    # Soft/Professional Skills
    # --------------------------------------------------------

    "professional": [
        "communication",
        "leadership",
        "problem solving",
        "problem-solving",
        "teamwork",
        "time management",
        "project management",
        "critical thinking",
        "analytical thinking"
    ]
}


# ============================================================
# JOB ROLE SKILL REQUIREMENTS
# ============================================================

JOB_ROLE_SKILLS = {

    "Data Scientist": [
        "python",
        "pandas",
        "numpy",
        "sql",
        "machine learning",
        "statistics",
        "scikit-learn",
        "data visualization"
    ],

    "Machine Learning Engineer": [
        "python",
        "machine learning",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "sql",
        "docker",
        "git"
    ],

    "Data Analyst": [
        "python",
        "sql",
        "pandas",
        "excel",
        "statistics",
        "data visualization",
        "power bi",
        "tableau"
    ],

    "Python Developer": [
        "python",
        "git",
        "sql",
        "rest api",
        "django",
        "flask",
        "fastapi"
    ],

    "Full Stack Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "node.js",
        "sql",
        "git",
        "rest api"
    ],

    "AI Engineer": [
        "python",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "nlp",
        "llm",
        "git"
    ],

    "Data Engineer": [
        "python",
        "sql",
        "etl",
        "data engineering",
        "spark",
        "hadoop",
        "aws",
        "docker"
    ],

    "DevOps Engineer": [
        "linux",
        "docker",
        "kubernetes",
        "jenkins",
        "aws",
        "terraform",
        "git",
        "ci/cd"
    ]
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_skill_text(text):
    """
    Normalize text before skill extraction.
    """

    if not text:

        return ""

    text = str(text).lower()

    # Normalize common variations

    replacements = {

        "react.js": "react",

        "node.js": "node.js",

        "scikit learn": "scikit-learn",

        "scikit learn library": "scikit-learn",

        "powerbi": "power bi",

        "postgre sql": "postgresql",

        "postgres sql": "postgresql",

        "machine-learning": "machine learning",

        "deep-learning": "deep learning",

        "natural language processing":
            "natural language processing",

        "artificial-intelligence":
            "artificial intelligence"
    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    return text


# ============================================================
# CHECK WHETHER SKILL EXISTS
# ============================================================

def skill_exists(text, skill):
    """
    Check whether a skill exists in the text.

    Uses word-boundary matching for short skills
    to reduce false positives.
    """

    if not text or not skill:

        return False


    skill = skill.lower().strip()


    # Escape special regex characters

    escaped_skill = re.escape(
        skill
    )


    # Use boundaries for normal skills

    pattern = (
        r"(?<![a-zA-Z0-9])"
        + escaped_skill
        + r"(?![a-zA-Z0-9])"
    )


    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )
    )


# ============================================================
# EXTRACT SKILLS
# ============================================================

def extract_skills(text):
    """
    Extract all recognized skills from resume text.

    Returns
    -------
    list
        Unique detected skills.
    """

    if not text:

        return []


    text = normalize_skill_text(
        text
    )


    detected_skills = []


    for category, skills in SKILL_DATABASE.items():

        for skill in skills:

            if skill_exists(
                text,
                skill
            ):

                # Store original readable format

                formatted_skill = format_skill_name(
                    skill
                )

                if formatted_skill not in detected_skills:

                    detected_skills.append(
                        formatted_skill
                    )


    return sorted(
        detected_skills
    )


# ============================================================
# FORMAT SKILL NAME
# ============================================================

def format_skill_name(skill):
    """
    Convert internal skill names into readable names.
    """

    formatting = {

        "python": "Python",

        "java": "Java",

        "javascript": "JavaScript",

        "typescript": "TypeScript",

        "c++": "C++",

        "c#": "C#",

        "pandas": "Pandas",

        "numpy": "NumPy",

        "scipy": "SciPy",

        "matplotlib": "Matplotlib",

        "seaborn": "Seaborn",

        "plotly": "Plotly",

        "scikit-learn": "Scikit-learn",

        "sklearn": "Scikit-learn",

        "tensorflow": "TensorFlow",

        "pytorch": "PyTorch",

        "keras": "Keras",

        "machine learning": "Machine Learning",

        "deep learning": "Deep Learning",

        "natural language processing":
            "Natural Language Processing",

        "nlp": "NLP",

        "computer vision": "Computer Vision",

        "sql": "SQL",

        "mysql": "MySQL",

        "postgresql": "PostgreSQL",

        "postgres": "PostgreSQL",

        "mongodb": "MongoDB",

        "sqlite": "SQLite",

        "firebase": "Firebase",

        "supabase": "Supabase",

        "html": "HTML",

        "css": "CSS",

        "react": "React",

        "node.js": "Node.js",

        "next.js": "Next.js",

        "tailwind": "Tailwind CSS",

        "rest api": "REST API",

        "aws": "AWS",

        "azure": "Azure",

        "google cloud": "Google Cloud",

        "gcp": "GCP",

        "docker": "Docker",

        "kubernetes": "Kubernetes",

        "jenkins": "Jenkins",

        "terraform": "Terraform",

        "linux": "Linux",

        "git": "Git",

        "github": "GitHub",

        "hadoop": "Hadoop",

        "spark": "Spark",

        "apache spark": "Apache Spark",

        "kafka": "Kafka",

        "big data": "Big Data",

        "data engineering": "Data Engineering",

        "etl": "ETL",

        "artificial intelligence":
            "Artificial Intelligence",

        "generative ai":
            "Generative AI",

        "large language models":
            "Large Language Models",

        "llm":
            "LLM",

        "prompt engineering":
            "Prompt Engineering",

        "transformers":
            "Transformers",

        "rag":
            "RAG",

        "retrieval augmented generation":
            "Retrieval-Augmented Generation"
    }


    return formatting.get(
        skill.lower(),
        skill.title()
    )


# ============================================================
# GET SKILLS BY CATEGORY
# ============================================================

def get_skills_by_category(text):
    """
    Return detected skills grouped by category.

    Example:

    {
        "programming": ["Python", "Java"],
        "data_science": ["Pandas", "NumPy"],
        "machine_learning": ["Machine Learning"]
    }
    """

    if not text:

        return {}


    text = normalize_skill_text(
        text
    )


    results = {}


    for category, skills in SKILL_DATABASE.items():

        detected = []


        for skill in skills:

            if skill_exists(
                text,
                skill
            ):

                detected.append(
                    format_skill_name(skill)
                )


        if detected:

            results[category] = sorted(
                set(detected)
            )


    return results


# ============================================================
# FIND MISSING SKILLS
# ============================================================

def find_missing_skills(
    candidate_skills,
    target_job
):
    """
    Find skills required by a target job
    that are missing from the candidate.

    Parameters
    ----------
    candidate_skills : list
        Skills detected in resume.

    target_job : str
        Target job role.

    Returns
    -------
    list
        Missing skills.
    """

    if not target_job:

        return []


    required_skills = JOB_ROLE_SKILLS.get(
        target_job,
        []
    )


    candidate_normalized = {
        skill.lower()
        for skill in candidate_skills
    }


    missing = []


    for required_skill in required_skills:

        if required_skill.lower() not in candidate_normalized:

            missing.append(
                format_skill_name(
                    required_skill
                )
            )


    return missing


# ============================================================
# SKILL MATCH SCORE
# ============================================================

def calculate_skill_match(
    candidate_skills,
    target_job
):
    """
    Calculate percentage of required job skills
    already present in the candidate resume.
    """

    if not target_job:

        return 0.0


    required_skills = JOB_ROLE_SKILLS.get(
        target_job,
        []
    )


    if not required_skills:

        return 0.0


    candidate_normalized = {
        skill.lower()
        for skill in candidate_skills
    }


    matched = sum(
        1
        for skill in required_skills
        if skill.lower() in candidate_normalized
    )


    score = (
        matched /
        len(required_skills)
    ) * 100


    return round(
        score,
        2
    )


# ============================================================
# GET SKILL CATEGORIES SUMMARY
# ============================================================

def get_skill_summary(text):
    """
    Generate a simple summary of detected skills.
    """

    categorized = get_skills_by_category(
        text
    )


    summary = {}


    for category, skills in categorized.items():

        summary[category] = {
            "count": len(skills),
            "skills": skills
        }


    return summary


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "SKILL EXTRACTOR MODULE"
    )

    print("=" * 60)


    sample_resume = """
    I am a Data Science student with strong experience
    in Python, Pandas, NumPy, SQL and Machine Learning.

    I have worked with Scikit-learn, TensorFlow,
    Matplotlib and Power BI.

    I also have experience with Git, GitHub and Docker.
    """


    print("\nSample Resume:")

    print(sample_resume)


    # --------------------------------------------------------
    # Extract skills
    # --------------------------------------------------------

    skills = extract_skills(
        sample_resume
    )


    print("\nDetected Skills:")

    for skill in skills:

        print(
            f"  ✓ {skill}"
        )


    # --------------------------------------------------------
    # Categorized skills
    # --------------------------------------------------------

    print(
        "\nSkills By Category:"
    )


    categorized = get_skills_by_category(
        sample_resume
    )


    for category, category_skills in categorized.items():

        print(
            f"\n{category}:"
        )

        for skill in category_skills:

            print(
                f"  • {skill}"
            )


    # --------------------------------------------------------
    # Data Scientist analysis
    # --------------------------------------------------------

    target_role = "Data Scientist"


    missing = find_missing_skills(
        skills,
        target_role
    )


    score = calculate_skill_match(
        skills,
        target_role
    )


    print(
        f"\nTarget Role: {target_role}"
    )


    print(
        f"Skill Match: {score}%"
    )


    print(
        "\nMissing Skills:"
    )


    for skill in missing:

        print(
            f"  ✗ {skill}"
        )


    print("\nSkill extractor ready ✅")

    print("=" * 60)