def extract_salary(jobs):
    salaries = []

    for job in jobs:
        min_sal = job.get("job_salary_min")
        max_sal = job.get("job_salary_max")

        if min_sal and max_sal:
            salaries.append((min_sal + max_sal) / 2)

    if not salaries:
        return 0

    return sum(salaries) / len(salaries)
