from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import os

from vector_search import get_results
from utils import EmptyResumeException, NoSkillsException

load_dotenv()

EXTRACT_SKILLS_PROMPT = """
You are a resume skills extractor.

Your task: extract technical skills from the resume that have one or more experiences or projects that use them.

Resume:
{resume_text}

Rules:
- If the skill is listed but not used in any experiences or projects, do not include it in the output.
- If there are no skills, if the resume is empty, or if the text is not a resume, output an empty string.

Output format:
- List of skills, one per line
For example:
Python
React
Git

Provide no other text in your output.
Provide your output as plain text, with no extra formatting.
"""

RECOMMEND_SKILLS_PROMPT = """
You are a technical skills recommender.

Your task: recommend 1-3 new technical skills to the user that work in synergy with skills they already have based on the related sections found in job postings.

Current skills:
{skills}

Job postings:
{job_postings}

Instructions:
1. Look at the job postings and identify 1-3 technical skills that usually go together with the current skills.
2. Provide some explanation and evidence for your recommendation.
3. DO NOT include skills that are not technical, like "communication", "teamwork", "leadership", etc.
4. DO NOT include skills that are already in the current skills list.
"""

def main():
    print(recommend_skills("../uploads/Resume.pdf"))

# read resume -> extract skills (llm call) -> find relevant chunks (llm call for multi-query) -> generate response (llm call)

# if resume is empty -> exception raised
# if no skills -> exception raised
# if no results -> exception raised
# all good -> normal response
def recommend_skills(resume_path) -> str:
    resume_text = read_resume(resume_path).strip()
    skills = extract_skills(resume_text)
    results = get_results(skills, [0.2, 0.8])
    return invoke_llm(skills, results)

def read_resume(resume_path):
    reader = PdfReader(resume_path)
    resume_text = "".join(page.extract_text() or "" for page in reader.pages).strip()
    if not resume_text:
        raise EmptyResumeException
    return resume_text

def extract_skills(resume_text):
    llm = ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL"), temperature=0)
    prompt = EXTRACT_SKILLS_PROMPT.format(resume_text=resume_text)
    skills = llm.invoke(prompt).content.strip()
    if not skills:
        raise NoSkillsException
    print(f"Skills: {skills}")
    return skills

def invoke_llm(skills, results):
    llm = ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL"), temperature=0.3)
    job_postings = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(RECOMMEND_SKILLS_PROMPT)
    prompt = prompt_template.format(skills=skills, job_postings=job_postings)
    return llm.invoke(prompt).content

if __name__ == "__main__":
    main()
