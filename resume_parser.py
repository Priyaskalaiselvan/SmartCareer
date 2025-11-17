# resume_parser.py -- Lightweight version (No spaCy)

import re
from pdfminer.high_level import extract_text

# Simple technical & soft skill dictionaries
TECH_SKILLS = [
    "python","java","c++","c","sql","mysql","mongodb","postgresql",
    "html","css","javascript","node","react","angular","vue",
    "aws","azure","gcp","docker","kubernetes","linux","git",
    "machine learning","deep learning","nlp","data analysis",
    "power bi","tableau","excel","pandas","numpy","django","flask"
]

SOFT_SKILLS = [
    "communication","leadership","teamwork","problem solving",
    "creativity","time management","adaptability","critical thinking",
    "presentation","collaboration","organization"
]

CERT_KEYWORDS = [
    "certified","certificate","certification","aws","azure","google cloud",
    "oracle","red hat","ccna","pmp","scrum","machine learning"
]

EDU_KEYWORDS = [
    "bachelor","master","phd","diploma","b.tech","bsc","msc","be","degree","engineering"
]

def extract_text_from_pdf(file_bytes):
    try:
        return extract_text(file_bytes)
    except:
        return ""

def extract_keywords(text, keyword_list):
    found = []
    text_lower = text.lower()
    for k in keyword_list:
        if k in text_lower:
            found.append(k)
    return list(set(found))

def extract_certifications(text):
    lines = text.lower().split("\n")
    certs = []
    for line in lines:
        if any(k in line for k in CERT_KEYWORDS):
            certs.append(line.strip())
    return certs[:10]

def extract_experience(text):
    match = re.search(r'(\d+)\s*\+?\s*years', text.lower())
    if match:
        return match.group(1)
    return "Not specified"

def parse_resume(file_bytes):
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        return {}

    tech = extract_keywords(text, TECH_SKILLS)
    soft = extract_keywords(text, SOFT_SKILLS)
    edu = extract_keywords(text, EDU_KEYWORDS)
    certs = extract_certifications(text)
    exp = extract_experience(text)

    return {
        "skills": list(set(tech + soft)),
        "tech_skills": tech,
        "soft_skills": soft,
        "education": edu,
        "certifications": certs,
        "experience_years": exp,
        "raw_text": text[:2000]
    }
