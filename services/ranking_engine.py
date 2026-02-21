def calculate_opportunity(
    resume_score,
    demand,
    supply,
    salary_growth,
    country_index
):

    # -------- 1. Market Tightness (Normalized) --------
    raw_tightness = demand / supply if supply > 0 else 0

    # Cap extreme values
    tightness = min(raw_tightness * 100, 100)

    # -------- 2. Salary Growth Normalization --------
    salary_growth_norm = min(max(salary_growth, 0), 100)

    # -------- 3. Weighted Opportunity --------
    opportunity = (
        0.35 * resume_score +
        0.30 * tightness +
        0.20 * salary_growth_norm +
        0.15 * country_index
    )

    return round(opportunity, 2), round(tightness, 2)
