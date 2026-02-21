import numpy as np

def percentile_rank(user_score):
    distribution = np.random.normal(60, 15, 1000)
    percentile = sum(distribution < user_score) / len(distribution) * 100
    return round(percentile, 2)
