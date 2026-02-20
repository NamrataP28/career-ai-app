import pdfplumber
import re

class ResumeParser:

    def extract_text(self, file):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def extract_experience(self, text):
        match = re.search(r'(\d+)\+?\s*years', text.lower())
        return int(match.group(1)) if match else 0
