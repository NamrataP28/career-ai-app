from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class GPTService:

    def generate_roadmap(self, resume_text, role):

        prompt = f"""
        Resume:
        {resume_text}

        Target Role:
        {role}

        Provide:
        - Skill gaps
        - Certifications
        - Project ideas
        - 90-day roadmap
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
