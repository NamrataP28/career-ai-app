import re

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return list(set(words))

def compute_skill_gap(resume_text, job_list):

    resume_kw = extract_keywords(resume_text)

    job_kw = []
    for job in job_list:
        desc = job.get("job_description", "")
        job_kw += extract_keywords(desc)

    job_kw = list(set(job_kw))

    matched = list(set(resume_kw) & set(job_kw))
    missing = list(set(job_kw) - set(resume_kw))

    score = len(matched) / len(job_kw) * 100 if job_kw else 0

    return round(score, 2), missing[:20]
