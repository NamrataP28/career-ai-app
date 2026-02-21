from openai import OpenAI
import streamlit as st


class GPTService:

    def __init__(self):
        # Use Streamlit secrets instead of dotenv
        api_key = st.secrets.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in Streamlit secrets")

        self.client = OpenAI(api_key=api_key)

    def generate_roadmap(self, resume_text, role):

        prompt = f"""
You are a senior career strategist.

Candidate Resume:
{resume_text}

Target Role:
{role}

Provide:

1. Skill Gap Analysis (clear bullets)
2. Recommended Certifications (role-aligned)
3. 2–3 Practical Portfolio Project Ideas
4. 90-Day Structured Roadmap (weekly breakdown)

Be realistic, practical, and market-relevant.
Avoid generic advice.
Keep it structured and professional.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert AI Career Strategist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"⚠️ GPT Error: {str(e)}"
