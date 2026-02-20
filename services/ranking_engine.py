from sklearn.metrics.pairwise import cosine_similarity

class RankingEngine:

    def compute_score(self, resume_embedding, job_embedding,
                      demand, salary_norm, visa_score):

        similarity = cosine_similarity(
            [resume_embedding],
            [job_embedding]
        )[0][0]

        score = (
            0.5 * similarity +
            0.2 * demand +
            0.2 * salary_norm +
            0.1 * visa_score
        )

        return similarity, score
