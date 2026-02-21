from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return list(set(words))

def compute_skill_gap(resume_text, job_descriptions):

    resume_keywords = extract_keywords(resume_text)

    job_keywords = []
    for job in job_descriptions:
        desc = job.get("job_description", "")
        job_keywords += extract_keywords(desc)

    job_keywords = list(set(job_keywords))

    matched = list(set(resume_keywords) & set(job_keywords))
    missing = list(set(job_keywords) - set(resume_keywords))

    match_score = (len(matched) / len(job_keywords) * 100) if job_keywords else 0

    return round(match_score,2), missing[:15]
