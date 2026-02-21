from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_resume_match(resume_text, role, job_descriptions):

    # Combine role + sample job descriptions
    combined_job_text = role

    for job in job_descriptions[:5]:
        desc = job.get("job_description", "")
        combined_job_text += " " + desc

    resume_embedding = model.encode(resume_text)
    job_embedding = model.encode(combined_job_text)

    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    return round(similarity * 100, 2)
