def estimate_supply(role, country):
    base_supply = {
        "Data Analyst": 10000,
        "Software Engineer": 20000,
        "Product Manager": 8000,
        "Financial Analyst": 9000
    }

    country_factor = {
        "USA": 1.4,
        "India": 1.8,
        "Germany": 1.1,
        "UK": 1.2,
        "Singapore": 0.8
    }

    base = base_supply.get(role, 7000)
    multiplier = country_factor.get(country, 1.0)

    return int(base * multiplier)
