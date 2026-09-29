# AI Resume Analyzer & Job Matcher

An AI-powered web application that analyzes resumes, extracts relevant skills and information, matches candidates with suitable job opportunities, and provides job recommendations using Natural Language Processing (NLP) and Machine Learning.

## 🚀 Features

* 📄 Resume parsing and text extraction
* 🧠 Skill extraction from resumes
* 🔍 Resume-to-job matching
* 📊 Resume analysis
* 🎯 Job recommendations
* 🤖 Machine Learning-based classification
* 📝 TF-IDF-based text processing
* 🌐 Interactive Streamlit web interface
* 📁 Job and skill datasets in CSV format

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Natural Language Processing (NLP)**
* **TF-IDF**
* **Machine Learning**
* **CSV**

## 📂 Project Structure

```text
AI-Resume-Analyzer/
│
├── app.py
├── train_model.py
├── resume_parser.py
├── text_processor.py
├── skill_extractor.py
├── job_matcher.py
├── recommender.py
├── utils.py
│
├── data/
│   ├── jobs.csv
│   └── skills.csv
│
├── models/
│   ├── classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔄 How It Works

```text
Resume Upload
      ↓
Resume Parsing
      ↓
Text Processing
      ↓
Skill Extraction
      ↓
Feature Extraction
      ↓
Machine Learning Analysis
      ↓
Job Matching
      ↓
Job Recommendations
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ombhajbhuje180/AI-Resume-Analyzer.git
```

### 2. Navigate to the project

```bash
cd AI-Resume-Analyzer
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🧠 Machine Learning

The project uses Natural Language Processing techniques to transform resume and job-description text into numerical features.

### TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) is used to represent important words and terms from textual data.

### Classification

A trained machine learning classifier is used as part of the resume analysis pipeline.

The trained model and TF-IDF vectorizer are stored in:

```text
models/
├── classifier.pkl
└── tfidf_vectorizer.pkl
```

## 📊 Data

The project contains datasets for jobs and skills:

```text
data/
├── jobs.csv
└── skills.csv
```

These datasets are used by the application for job matching, skill extraction, and recommendations.

## 🎯 Use Cases

* Resume analysis
* Job recommendation systems
* Candidate-job matching
* Skill gap identification
* Career assistance applications
* Recruitment and HR technology

## 🔮 Future Improvements

* Improve resume parsing accuracy
* Add support for more resume formats
* Expand the job dataset
* Add advanced semantic similarity using transformer models
* Add personalized career recommendations
* Add skill-gap analysis
* Deploy the application as a production web service
* Add user authentication and resume history

## 👨‍💻 Project

**AI Resume Analyzer & Job Matcher**

Built using Python, Machine Learning, NLP, and Streamlit.

## 📜 License

This project is intended for educational and portfolio purposes.
