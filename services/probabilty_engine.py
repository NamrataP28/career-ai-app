def interview_probability(resume_score, tightness, skill_score):

    probability = (
        0.40 * resume_score +
        0.35 * tightness +
        0.25 * skill_score
    )

    return round(min(probability, 95),2)
