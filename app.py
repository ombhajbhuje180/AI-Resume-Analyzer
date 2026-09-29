import os
import pandas as pd
import streamlit as st
import plotly.express as px

from resume_parser import extract_resume_text
from text_processor import clean_text
from skill_extractor import extract_skills, find_missing_skills
from job_matcher import match_jobs
from recommender import generate_recommendations


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.9;
    }

    .metric-card {
        padding: 20px;
        border-radius: 15px;
        background: white;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .skill {
        display: inline-block;
        padding: 7px 12px;
        margin: 4px;
        border-radius: 20px;
        background-color: #e0e7ff;
        color: #3730a3;
        font-weight: 600;
    }

    .missing-skill {
        display: inline-block;
        padding: 7px 12px;
        margin: 4px;
        border-radius: 20px;
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 600;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "skills" not in st.session_state:
    st.session_state.skills = []

if "job_matches" not in st.session_state:
    st.session_state.job_matches = pd.DataFrame()

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🤖 AI Resume Analyzer</h1>

        <p>
        Intelligent Resume Analysis • Job Role Prediction •
        Skill Gap Detection • Job Matching
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Project Controls")

    st.write(
        """
        Upload a candidate resume and let the AI system
        analyze skills, identify suitable job roles and
        generate career recommendations.
        """
    )

    st.divider()

    st.subheader("System Pipeline")

    st.write("📄 Resume Extraction")
    st.write("🧹 Text Processing")
    st.write("🧠 Skill Extraction")
    st.write("🤖 ML Job Prediction")
    st.write("🎯 Job Matching")
    st.write("💡 Recommendation")

    st.divider()

    st.caption("AI Resume Analyzer v1.0")


# ============================================================
# RESUME UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📄 Upload Resume</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload candidate resume",
    type=["pdf", "docx"],
    help="Supported formats: PDF and DOCX"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file is not None:

    file_extension = uploaded_file.name.split(".")[-1].lower()

    st.info(
        f"📎 Selected file: **{uploaded_file.name}**"
    )

    analyze_button = st.button(
        "🚀 Analyze Resume",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        try:

            # ------------------------------------------------
            # STEP 1 — RESUME EXTRACTION
            # ------------------------------------------------

            with st.spinner("📄 Extracting resume text..."):

                resume_text = extract_resume_text(
                    uploaded_file,
                    file_extension
                )

            if not resume_text or not resume_text.strip():

                st.error(
                    "Unable to extract text from this resume."
                )

                st.stop()


            # ------------------------------------------------
            # STEP 2 — TEXT PROCESSING
            # ------------------------------------------------

            with st.spinner("🧹 Processing resume text..."):

                processed_text = clean_text(resume_text)


            # ------------------------------------------------
            # STEP 3 — SKILL EXTRACTION
            # ------------------------------------------------

            with st.spinner("🧠 Detecting candidate skills..."):

                skills = extract_skills(processed_text)


            # ------------------------------------------------
            # STEP 4 — JOB MATCHING
            # ------------------------------------------------

            with st.spinner("🎯 Matching candidate with jobs..."):

                job_matches = match_jobs(processed_text)


            # ------------------------------------------------
            # STEP 5 — RECOMMENDATIONS
            # ------------------------------------------------

            with st.spinner("💡 Generating recommendations..."):

                recommendations = generate_recommendations(
                    skills,
                    job_matches
                )


            # ------------------------------------------------
            # SAVE RESULTS
            # ------------------------------------------------

            st.session_state.resume_text = resume_text

            st.session_state.skills = skills

            st.session_state.job_matches = job_matches

            st.session_state.recommendations = recommendations

            st.session_state.analysis_complete = True

            st.success(
                "✅ Resume analysis completed successfully!"
            )


        except Exception as error:

            st.error(
                f"❌ Analysis failed: {error}"
            )

            st.exception(error)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_complete:

    resume_text = st.session_state.resume_text

    skills = st.session_state.skills

    job_matches = st.session_state.job_matches

    recommendations = st.session_state.recommendations


    # ========================================================
    # DASHBOARD METRICS
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Resume Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)


    # Number of skills

    with col1:

        st.metric(
            "Skills Detected",
            len(skills)
        )


    # Number of jobs

    with col2:

        if isinstance(job_matches, pd.DataFrame):

            st.metric(
                "Jobs Analyzed",
                len(job_matches)
            )

        else:

            st.metric(
                "Jobs Analyzed",
                0
            )


    # Best match

    best_match = 0

    if isinstance(job_matches, pd.DataFrame):

        if not job_matches.empty:

            possible_columns = [
                "match_score",
                "score",
                "similarity",
                "match_percentage"
            ]

            for column in possible_columns:

                if column in job_matches.columns:

                    best_match = float(
                        job_matches[column].max()
                    )

                    break


    with col3:

        st.metric(
            "Best Match",
            f"{best_match:.1f}%"
        )


    # Resume length

    with col4:

        word_count = len(
            resume_text.split()
        )

        st.metric(
            "Resume Words",
            word_count
        )


    st.divider()


    # ========================================================
    # SKILLS SECTION
    # ========================================================

    st.markdown(
        '<div class="section-title">🧠 Detected Skills</div>',
        unsafe_allow_html=True
    )

    if skills:

        skill_html = ""

        for skill in skills:

            skill_html += (
                f'<span class="skill">{skill}</span>'
            )

        st.markdown(
            skill_html,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "No known skills were detected in the resume."
        )


    # ========================================================
    # JOB MATCHING SECTION
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Recommended Job Roles</div>',
        unsafe_allow_html=True
    )


    if isinstance(job_matches, pd.DataFrame):

        if not job_matches.empty:

            display_df = job_matches.copy()


            # ----------------------------------------------
            # FIND SCORE COLUMN
            # ----------------------------------------------

            score_column = None

            for column in [
                "match_score",
                "score",
                "similarity",
                "match_percentage"
            ]:

                if column in display_df.columns:

                    score_column = column
                    break


            # ----------------------------------------------
            # NORMALIZE SCORE
            # ----------------------------------------------

            if score_column:

                display_df["Match Score"] = (
                    display_df[score_column]
                    .astype(float)
                )

                # Convert 0-1 similarity to percentage

                if display_df["Match Score"].max() <= 1:

                    display_df["Match Score"] *= 100


                display_df["Match Score"] = (
                    display_df["Match Score"]
                    .round(1)
                )


            # ----------------------------------------------
            # DISPLAY TABLE
            # ----------------------------------------------

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


            # ----------------------------------------------
            # CHART
            # ----------------------------------------------

            if score_column:

                chart_column = "Match Score"

                title_column = None

                for column in [
                    "job_title",
                    "title",
                    "role",
                    "job_role"
                ]:

                    if column in display_df.columns:

                        title_column = column
                        break


                if title_column:

                    chart_df = display_df[
                        [title_column, chart_column]
                    ].copy()

                    chart_df = chart_df.head(10)


                    fig = px.bar(
                        chart_df,
                        x=chart_column,
                        y=title_column,
                        orientation="h",
                        title="Top Job Matches",
                        labels={
                            chart_column: "Match Score (%)",
                            title_column: "Job Role"
                        }
                    )

                    fig.update_layout(
                        yaxis={
                            "categoryorder": "total ascending"
                        }
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

        else:

            st.warning(
                "No suitable jobs were found."
            )

    else:

        st.warning(
            "Job matching returned an unexpected format."
        )


    # ========================================================
    # SKILL GAP ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">⚠️ Skill Gap Analysis</div>',
        unsafe_allow_html=True
    )


    # Determine target job

    target_job = None

    if isinstance(job_matches, pd.DataFrame):

        if not job_matches.empty:

            for column in [
                "job_title",
                "title",
                "role",
                "job_role"
            ]:

                if column in job_matches.columns:

                    target_job = job_matches.iloc[0][column]

                    break


    if target_job:

        st.write(
            f"Target Role: **{target_job}**"
        )


    try:

        if target_job:

            missing_skills = find_missing_skills(
                skills,
                target_job
            )

        else:

            missing_skills = []

    except Exception:

        missing_skills = []


    if missing_skills:

        missing_html = ""

        for skill in missing_skills:

            missing_html += (
                f'<span class="missing-skill">'
                f'{skill}'
                f'</span>'
            )

        st.markdown(
            missing_html,
            unsafe_allow_html=True
        )

    else:

        st.success(
            "🎉 No major skill gaps detected."
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">💡 AI Career Recommendations</div>',
        unsafe_allow_html=True
    )


    if recommendations:

        for index, recommendation in enumerate(
            recommendations,
            start=1
        ):

            st.write(
                f"**{index}.** {recommendation}"
            )

    else:

        st.info(
            "No additional recommendations available."
        )


    # ========================================================
    # RESUME PREVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">📄 Extracted Resume Text</div>',
        unsafe_allow_html=True
    )


    with st.expander(
        "View extracted resume content"
    ):

        st.text_area(
            "Resume Text",
            resume_text,
            height=350,
            disabled=True
        )


    # ========================================================
    # SYSTEM PIPELINE
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">⚙️ AI Processing Pipeline</div>',
        unsafe_allow_html=True
    )


    pipeline_cols = st.columns(6)

    pipeline_steps = [
        ("📄", "Resume\nInput"),
        ("🧹", "Text\nProcessing"),
        ("🧠", "Skill\nExtraction"),
        ("🤖", "ML\nPrediction"),
        ("🎯", "Job\nMatching"),
        ("💡", "Recommendation")
    ]


    for column, (icon, label) in zip(
        pipeline_cols,
        pipeline_steps
    ):

        with column:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div style="font-size:30px;">
                        {icon}
                    </div>

                    <div style="font-weight:600;">
                        {label.replace(chr(10), "<br>")}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# INITIAL STATE
# ============================================================

else:

    st.markdown(
        """
        ### 🚀 How It Works

        1. **Upload** a candidate resume
        2. **Extract** resume information
        3. **Detect** technical skills
        4. **Predict** suitable job roles using ML
        5. **Calculate** job compatibility
        6. **Identify** missing skills
        7. **Generate** career recommendations
        """,
    )

    st.info(
        "👆 Upload a PDF or DOCX resume above to start the analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Resume Analyzer & Job Matcher • "
    "Machine Learning + NLP + Recommendation System"
)