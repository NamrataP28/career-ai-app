class VisaModel:

    def visa_score(self, country):
        visa_friendly = {
            "Canada": 0.8,
            "UK": 0.6,
            "USA": 0.5,
            "India": 0.9
        }
        return visa_friendly.get(country, 0.5)
