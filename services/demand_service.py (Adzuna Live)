import requests
import streamlit as st

def fetch_live_demand(role, country):

    headers = {
        "X-RapidAPI-Key": st.secrets.get("RAPIDAPI_KEY", ""),
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    url = "https://jsearch.p.rapidapi.com/search"

    params = {
        "query": role,
        "page": "1",
        "num_pages": "1",
        "country": country.lower()
    }

    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            return 0, [], []

        data = response.json()
        jobs = data.get("data", [])

        companies = []
        for job in jobs:
            company = job.get("employer_name")
            if company:
                companies.append(company)

        return len(jobs), jobs, list(set(companies))[:10]

    except:
        return 0, [], []
