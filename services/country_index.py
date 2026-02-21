def country_attractiveness(country):

    index = {
        "USA": 90,
        "Germany": 85,
        "UK": 88,
        "Singapore": 80,
        "Canada": 82,
        "Australia": 83,
        "India": 70
    }

    return index.get(country, 75)
