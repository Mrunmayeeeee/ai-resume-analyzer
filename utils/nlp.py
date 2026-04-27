"""
utils/nlp.py — Standalone NLP helpers for ResumeIQ
====================================================
Importable module for use outside Flask (e.g. CLI, notebooks, tests).
"""

import re
import string
from collections import Counter

import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data
for pkg in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words('english'))

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    nlp = None
    print("⚠ spaCy model not found. Run: python -m spacy download en_core_web_sm")


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str) -> list[str]:
    return [t for t in word_tokenize(clean(text))
            if t not in STOP_WORDS and len(t) > 2]


def tfidf_sim(a: str, b: str) -> float:
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        mat = vec.fit_transform([a, b])
        return round(float(cosine_similarity(mat[0:1], mat[1:2])[0][0]) * 100, 1)
    except Exception:
        return 0.0


def spacy_entities(text: str) -> list[str]:
    if not nlp:
        return []
    doc = nlp(text[:10000])
    return [e.text.lower() for e in doc.ents if e.label_ in ('ORG', 'PRODUCT', 'WORK_OF_ART')]


def keyword_gap(resume: str, jd: str) -> dict:
    """Return found/missing keywords + match percentage."""
    jd_tokens  = set(tokenize(jd))
    res_tokens = set(tokenize(resume))
    jd_ents    = set(spacy_entities(jd))
    res_ents   = set(spacy_entities(resume))

    freq  = Counter(tokenize(jd))
    top   = {w for w, _ in freq.most_common(40)} | jd_ents

    found   = sorted((top & res_tokens) | (jd_ents & res_ents))[:12]
    missing = sorted((top - res_tokens) | (jd_ents - res_ents))[:10]
    pct     = int(len(found) / max(len(found) + len(missing), 1) * 100)

    return {'found': found, 'missing': missing, 'match_pct': pct}


def format_score(resume: str) -> int:
    score, lower = 50, resume.lower()
    sections = ['experience', 'education', 'skills', 'summary',
                'projects', 'certifications', 'achievements', 'objective']
    score += sum(1 for s in sections if s in lower) * 5
    score += min((resume.count('•') + resume.count('-')) * 1, 10)
    if len(resume) > 300:  score += 5
    if len(resume) > 3000: score -= 10
    if re.search(r'[\w.+-]+@[\w-]+\.\w+', resume): score += 5
    if re.search(r'\+?\d[\d\s\-().]{7,}', resume):  score += 5
    return min(max(score, 0), 100)


def ats_score(sim: float, kw_pct: int, fmt: int) -> int:
    return int(sim * 0.4 + kw_pct * 0.4 + fmt * 0.2)


def grade(score: int) -> str:
    return {True: 'Excellent'}.get(score >= 80,
           {True: 'Good'}.get(score >= 65,
           {True: 'Fair'}.get(score >= 45, 'Poor')))


def full_analysis(resume: str, jd: str) -> dict:
    """Run full NLP pipeline and return structured results."""
    sim  = tfidf_sim(resume, jd)
    kw   = keyword_gap(resume, jd)
    fmt  = format_score(resume)
    ats  = ats_score(sim, kw['match_pct'], fmt)

    return {
        'ats_score':           ats,
        'grade':               grade(ats),
        'similarity_pct':      sim,
        'keyword_match_pct':   kw['match_pct'],
        'format_score':        fmt,
        'keywords_found':      kw['found'],
        'keywords_missing':    kw['missing'],
    }
