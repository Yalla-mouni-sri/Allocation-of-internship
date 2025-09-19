Internship Matcher 🚀
📌 Overview

Finding the right internship is challenging for students, while organizations struggle to filter suitable candidates. This project aims to bridge that gap by building an AI-powered internship/job recommendation and application system.

🎯 Why? To reduce the manual effort in searching and applying for jobs.

🤔 What? A backend system that recommends relevant jobs to candidates and allows organizations to track applicants.

⚡ How? By using Machine Learning (ML) techniques to compute similarity between candidate skills and job requirements, and providing an interactive flow for both candidates and organizations.

🔍 Features

Candidate Flow:

Upload details (name, email, skills, preferences).

Get job recommendations ranked by relevance.

Apply to jobs with unique application IDs.

Organization Flow:

View jobs and number of applicants.

See detailed applicant info for each JobID.

Manage applications in a centralized CSV file.

🧠 Machine Learning Techniques Used

Text Vectorization (TF-IDF / embeddings)

Candidate skills and job descriptions are converted into numerical vectors.

Cosine Similarity

Computes how closely candidate skills match the job requirements.

Ranking Algorithm

Jobs are sorted in descending order of similarity score, ensuring candidates see the most relevant opportunities first.

🏗️ Tech Stack

Backend: Python (pandas, numpy)

ML/NLP: Scikit-learn (TF-IDF, cosine similarity)

Data Storage: CSV-based job & application database

Other: datetime, os for system handling

⚙️ Installation & Usage
# Clone repo
git clone https://github.com/your-username/internship-matcher.git
cd internship-matcher

# Install dependencies
pip install -r requirements.txt

# Run
python internship_matcher.py

📂 Project Structure
internship-matcher/
│── internship_matcher.py    # Main script (backend logic)
│── jobs.csv                 # Job postings
│── applications.csv         # Application records
│── README.md                # Project documentation
│── requirements.txt         # Python dependencies

🚀 Future Enhancements

Replace CSV with a database (MySQL/PostgreSQL).

Add frontend for candidates & organizations.

Deploy as a web app with Flask/Django.

Use deep learning embeddings (e.g., BERT) for better skill matching.
