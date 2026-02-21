def calculate_opportunity(resume_score, demand, supply, salary_growth):

    tightness = (demand / supply) * 100 if supply > 0 else 0

    opportunity = (
        0.35 * resume_score +
        0.30 * tightness +
        0.20 * salary_growth +
        0.15 * 70  # country attractiveness baseline
    )

    return round(opportunity, 2), round(tightness, 2)
