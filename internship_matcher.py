"""
Internship matchmaking prototype
Requirements:
  - python 3.8+
  - pip install sentence-transformers pandas
Run:
  python internship_matcher.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone


# Embedding model
try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    raise Exception("Please install sentence-transformers: pip install sentence-transformers pandas") from e

# -------------------------
# Config / Tunable params
# -------------------------
JOBS_CSV = "jobs.csv"
APPLICATIONS_CSV = "applications.csv"
MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast SBERT model
LOCATION_BONUS = 10.0  # add absolute percentage points
SECTOR_BONUS = 5.0     # add absolute percentage points
THRESHOLD = 20.0       # minimum percent to show a job
TOP_N = 20             # max to show (after threshold)

# -------------------------
# Sample jobs (created if jobs.csv missing)
# -------------------------
SAMPLE_JOBS = [
    {"job_id": 1, "organization":"TechCorp", "role":"Data Analyst", "skills_required":"Python, SQL, Excel, Data Analysis", "location":"Hyderabad", "sector":"IT"},
    {"job_id": 2, "organization":"HealthPlus", "role":"ML Engineer", "skills_required":"Python, TensorFlow, Machine Learning, Statistics", "location":"Bangalore", "sector":"Healthcare"},
    {"job_id": 3, "organization":"FinServe", "role":"Backend Developer", "skills_required":"Java, Spring, SQL, REST APIs", "location":"Chennai", "sector":"Finance"},
    {"job_id": 4, "organization":"AgriTech", "role":"Field Officer", "skills_required":"Agriculture Basics, Communication, Surveying", "location":"Vizag", "sector":"Agriculture"},
    {"job_id": 5, "organization":"InnoSoft", "role":"Full Stack Developer", "skills_required":"JavaScript, React, Node.js, SQL", "location":"Hyderabad", "sector":"IT"},
    {"job_id": 6, "organization":"GreenPower", "role":"Energy Analyst", "skills_required":"Energy Modeling, Python, MATLAB, Data Analysis", "location":"Pune", "sector":"Energy"},
    {"job_id": 7, "organization":"TeachWell", "role":"Educational Content Writer", "skills_required":"Content Writing, Curriculum, Communication", "location":"Kolkata", "sector":"Education"},
    {"job_id": 8, "organization":"ShopEase", "role":"Retail Analyst", "skills_required":"Excel, SQL, Retail Analytics, PowerBI", "location":"Mumbai", "sector":"Retail"},
    {"job_id": 9, "organization":"AutoMakers", "role":"Quality Engineer", "skills_required":"Manufacturing Processes, Quality Assurance, AutoCAD", "location":"Ahmedabad", "sector":"Manufacturing"},
    {"job_id": 10, "organization":"CivicWorks", "role":"Urban Planner Intern", "skills_required":"GIS, Urban Planning, Data Analysis", "location":"Lucknow", "sector":"Government"}
]

# -------------------------
# Helpers
# -------------------------
def ensure_jobs_csv():
    if not os.path.exists(JOBS_CSV):
        print(f"'{JOBS_CSV}' not found — creating sample {JOBS_CSV} with 10 rows.")
        df = pd.DataFrame(SAMPLE_JOBS)
        df.to_csv(JOBS_CSV, index=False)
    else:
        # quick check: required columns present?
        df = pd.read_csv(JOBS_CSV)
        required = {"job_id", "organization", "role", "skills_required", "location", "sector"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"{JOBS_CSV} missing required columns. Required: {required}")

def load_jobs():
    df = pd.read_csv(JOBS_CSV, dtype=str)
    # Ensure correct types
    df['job_id'] = df['job_id'].astype(int)
    df['organization'] = df['organization'].fillna("").astype(str)
    df['role'] = df['role'].fillna("").astype(str)
    df['skills_required'] = df['skills_required'].fillna("").astype(str)
    df['location'] = df['location'].fillna("").astype(str)
    df['sector'] = df['sector'].fillna("").astype(str)
    return df

def init_applications_csv():
    if not os.path.exists(APPLICATIONS_CSV):
        df = pd.DataFrame(columns=[
            "application_id", "timestamp", "candidate_name", "candidate_email",
            "candidate_skills", "preferred_location", "preferred_sector",
            "job_id", "job_title", "organization"
        ])
        df.to_csv(APPLICATIONS_CSV, index=False)

def append_application(record: dict):
    df = pd.read_csv(APPLICATIONS_CSV, dtype=str)   
    # df = df.append(record, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

    df.to_csv(APPLICATIONS_CSV, index=False)

def already_applied(candidate_email, job_id):
    if not os.path.exists(APPLICATIONS_CSV):
        return False
    df = pd.read_csv(APPLICATIONS_CSV, dtype=str)
    if candidate_email:
        s = df[(df['candidate_email'].str.lower() == candidate_email.lower()) & (df['job_id'].astype(int) == int(job_id))]
        return not s.empty
    else:
        # fallback: check by name + job
        s = df[(df['candidate_name'].str.lower() == candidate_name.lower()) & (df['job_id'].astype(int) == int(job_id))]
        return not s.empty

# -------------------------
# Main matching logic
# -------------------------
def build_job_embeddings(jobs_df, model):
    # Combine role + skills + sector to form job text
    job_texts = (jobs_df['role'].str.strip() + " | " + jobs_df['skills_required'].str.strip() + " | " + jobs_df['sector'].str.strip()).tolist()
    embeddings = model.encode(job_texts, convert_to_tensor=True)
    return embeddings

def score_jobs_for_candidate(candidate_skills_text, preferred_location, preferred_sector, jobs_df, job_embeddings, model):
    # candidate text: use skills + sector (sector helps semantic similarity)
    candidate_text = candidate_skills_text.strip()
    if preferred_sector:
        candidate_text = candidate_text + " | " + preferred_sector.strip()
    # embed candidate
    candidate_embedding = model.encode(candidate_text, convert_to_tensor=True)
    sims = util.cos_sim(candidate_embedding, job_embeddings).cpu().numpy().flatten()  # values in [-1,1] but usually [0,1]
    # convert to 0-100 scale
    scores = np.clip(sims, 0, 1) * 100.0
    # add bonus points for exact location/sector matches (case-insensitive contains)
    scored_list = []
    for i, row in jobs_df.iterrows():
        score = float(scores[i])
        # location bonus: if user typed 'any' or left blank -> don't boost
        if preferred_location and preferred_location.lower() != "any":
            if preferred_location.lower() in row['location'].lower():
                score += LOCATION_BONUS
        if preferred_sector and preferred_sector.lower() != "any":
            if preferred_sector.lower() in row['sector'].lower():
                score += SECTOR_BONUS
        score = min(score, 100.0)
        scored_list.append(score)
    return np.array(scored_list)

# -------------------------
# Candidate interaction flow
# -------------------------
def candidate_flow(jobs_df, job_embeddings, model):
    print("\n--- Candidate Profile Input ---")
    candidate_name = input("Enter your name: ").strip()
    candidate_email = input("Enter your email (optional, used to prevent duplicate applications): ").strip()
    candidate_skills = input("Enter your skills (comma-separated): ").strip()
    preferred_location = input("Preferred location (city or 'any'): ").strip()
    preferred_sector = input("Preferred sector (or 'any'): ").strip()

    if not candidate_skills:
        print("You must enter at least one skill. Exiting candidate flow.")
        return

    # create candidate text (simple join)
    candidate_skills_text = ", ".join([s.strip() for s in candidate_skills.split(",") if s.strip()])

    # compute scores
    scores = score_jobs_for_candidate(candidate_skills_text, preferred_location, preferred_sector, jobs_df, job_embeddings, model)

    jobs_df = jobs_df.copy()
    jobs_df['score'] = scores

    # filter threshold
    recommended = jobs_df[jobs_df['score'] >= THRESHOLD].sort_values(by='score', ascending=False).head(TOP_N)
    if recommended.empty:
        # if nothing passes threshold, show top 5 to the candidate with a warning
        print("\nNo strong matches (>= {:.0f}%). Showing top {} possible matches instead:".format(THRESHOLD, min(5, len(jobs_df))))
        recommended = jobs_df.sort_values(by='score', ascending=False).head(5)

    # Display recommendations
    print("\n--- Recommended Jobs (high -> low) ---")
    for _, job in recommended.iterrows():
        print(f"JobID: {job['job_id']} | Role: {job['role']} | Org: {job['organization']} | Location: {job['location']} | Sector: {job['sector']} | Score: {job['score']:.2f}%")

    # Build set of allowed job IDs to apply (only from this recommended list)
    allowed_ids = set(recommended['job_id'].astype(int).tolist())

    # Application loop
    print("\nYou may apply to any of the above recommended jobs. Enter JobID to apply, or type 'back' to finish.")
    while True:
        choice = input("Apply to JobID (or 'back'): ").strip()
        if choice.lower() == "back":
            print("Exiting application loop.")
            break
        if not choice.isdigit():
            print("Please enter a numeric JobID or 'back'.")
            continue
        jobid = int(choice)
        if jobid not in allowed_ids:
            print("You can only apply to the recommended JobIDs shown above. Try again.")
            continue
        # check duplicates
        # We will check by email if provided; otherwise by name+job
        existing = False
        df_apps = pd.read_csv(APPLICATIONS_CSV) if os.path.exists(APPLICATIONS_CSV) else pd.DataFrame()
        if not df_apps.empty:
            if candidate_email:
                existing = not df_apps[(df_apps['candidate_email'].str.lower() == candidate_email.lower()) & (df_apps['job_id'].astype(int) == jobid)].empty
            else:
                existing = not df_apps[(df_apps['candidate_name'].str.lower() == candidate_name.lower()) & (df_apps['job_id'].astype(int) == jobid)].empty

        if existing:
            print(f"You already applied to job {jobid}.")
            continue

        # record application
        job_row = jobs_df[jobs_df['job_id'] == jobid].iloc[0]
        
        application_record = {
    "application_id": f"app_{int(datetime.now(timezone.utc).timestamp())}_{jobid}",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "candidate_name": candidate_name,
    "candidate_email": candidate_email,
    "candidate_skills": candidate_skills_text,
    "preferred_location": preferred_location,
    "preferred_sector": preferred_sector,
    "job_id": str(jobid),
}
        append_application(application_record)
        print(f"Application submitted for JobID {jobid} ({job_row['role']} at {job_row['organization']}).")

    print("\nCandidate flow complete. Thank you.")

# -------------------------
# Organization view
# -------------------------
APPLICATIONS_CSV = "applications.csv"

def organization_flow(jobs_df):
    print("\n--- Organization view ---")
    print("Options:")
    print("  1. List all jobs and number of applicants")
    print("  2. Show applicants for a specific JobID")
    print("  3. Back to main menu")

    choice = input("Choose option (1/2/3): ").strip()

    if choice == "1":
        if not os.path.exists(APPLICATIONS_CSV):
            print("No applications yet.")
            return

        df_apps = pd.read_csv(APPLICATIONS_CSV, dtype=str)
        counts = df_apps.groupby('job_id').size().reset_index(name='app_count')

        # Ensure consistent datatypes for merge
        jobs_df['job_id'] = jobs_df['job_id'].astype(str)
        counts['job_id'] = counts['job_id'].astype(str)

        merged = pd.merge(
            jobs_df, counts, on='job_id', how='left'
        ).fillna({'app_count': 0})

        for _, row in merged.iterrows():
            print(f"JobID {row['job_id']}: {row['role']} at {row['organization']} "
                  f"({row['location']} - {row['sector']}) -> {int(row['app_count'])} applicants")

    elif choice == "2":
        jid = input("Enter JobID to view applicants: ").strip()

        if not os.path.exists(APPLICATIONS_CSV):
            print("No applications yet.")
            return

        df_apps = pd.read_csv(APPLICATIONS_CSV, dtype=str)

        df_filtered = df_apps[df_apps['job_id'] == jid]

        if df_filtered.empty:
            print(f"No applicants for JobID {jid}.")
            return

        print(f"\nApplicants for JobID {jid}:")
        for _, r in df_filtered.iterrows():
            print(f"- {r['candidate_name']} | email: {r.get('candidate_email','')} "
                  f"| skills: {r.get('candidate_skills','')} "
                  f"| applied at {r.get('timestamp')}")

    else:
        print("Returning to main menu.")
        return

# -------------------------
# Main program
# -------------------------
def main():
    ensure_jobs_csv()
    init_applications_csv()
    jobs_df = load_jobs()

    print("Loading embedding model (this may download the model on first run)...")
    model = SentenceTransformer(MODEL_NAME)
    job_embeddings = build_job_embeddings(jobs_df, model)
    print("Model loaded and job embeddings ready.")

    # main menu loop
    while True:
        print("\n--- Main Menu ---")
        print("1. Candidate: get job recommendations & apply")
        print("2. Organization: view applicants")
        print("3. Exit")
        choice = input("Choose option (1/2/3): ").strip()
        if choice == "1":
            candidate_flow(jobs_df, job_embeddings, model)
        elif choice == "2":
            organization_flow(jobs_df)
        elif choice == "3":
            print("Exiting. Bye.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
