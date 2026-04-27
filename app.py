"""
ResumeIQ — Python Flask Backend
================================
NLP-powered resume analysis using spaCy, NLTK, scikit-learn.
This backend handles keyword extraction, TF-IDF similarity,
and optionally calls the Claude API for richer AI feedback.

Run:
    pip install -r requirements.txt
    python app.py
"""

import os
import json
import re
import string
from collections import Counter

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# NLP
import nltk
import spacy
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data on first run
for pkg in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load spaCy model (run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    nlp = spacy.load('en_core_web_sm')

# ── Optional Claude AI integration ────────────────────────────────
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
USE_AI = bool(ANTHROPIC_API_KEY)

if USE_AI:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

STOP_WORDS = set(stopwords.words('english'))

# Common tech/soft-skill keywords to watch for
SKILL_PATTERNS = [
    # Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
    'kotlin', 'swift', 'ruby', 'php', 'scala', 'r',
    # Frameworks / Libraries
    'react', 'angular', 'vue', 'node', 'django', 'flask', 'fastapi', 'spring',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
    # Cloud / DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ci/cd',
    'jenkins', 'github actions', 'ansible',
    # Data
    'sql', 'nosql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
    'spark', 'hadoop', 'kafka', 'airflow', 'dbt',
    # Soft skills
    'leadership', 'communication', 'agile', 'scrum', 'collaboration',
    'problem-solving', 'mentoring', 'project management',
]


# ── Helper Functions ───────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, extra whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_tokens(text: str) -> list[str]:
    """Tokenise and remove stopwords."""
    tokens = word_tokenize(clean_text(text))
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def tfidf_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two documents via TF-IDF."""
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf = vec.fit_transform([text_a, text_b])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(float(score) * 100, 1)
    except Exception:
        return 0.0


def extract_text_from_pdf(file_storage) -> str:
    """Extract plain text from an uploaded PDF file."""
    reader = PdfReader(file_storage)
    pages = [page.extract_text() or '' for page in reader.pages]
    return '\n'.join(pages).strip()


def extract_named_skills(text: str) -> list[str]:
    """Use spaCy NER + pattern matching to extract skill mentions."""
    doc = nlp(text[:10000])
    found = set()

    # NER labels that often capture tech terms
    for ent in doc.ents:
        if ent.label_ in ('ORG', 'PRODUCT', 'WORK_OF_ART'):
            found.add(ent.text.lower())

    # Pattern matching for known skills
    lower = text.lower()
    for skill in SKILL_PATTERNS:
        if skill in lower:
            found.add(skill)

    return sorted(found)


def keyword_overlap(resume: str, jd: str) -> dict:
    """Find keywords present/missing in resume vs JD."""
    jd_tokens   = set(extract_tokens(jd))
    res_tokens  = set(extract_tokens(resume))
    jd_skills   = set(extract_named_skills(jd))
    res_skills  = set(extract_named_skills(resume))

    # Top JD keywords by frequency
    jd_freq = Counter(extract_tokens(jd))
    top_jd_kws = {w for w, _ in jd_freq.most_common(40)} | jd_skills

    found   = sorted(top_jd_kws & res_tokens | (jd_skills & res_skills))[:12]
    missing = sorted(top_jd_kws - res_tokens | (jd_skills - res_skills))[:10]

    match_pct = int(len(found) / max(len(found) + len(missing), 1) * 100)
    return {'found': found, 'missing': missing, 'match_pct': match_pct}


def score_format(resume: str) -> int:
    """Heuristic format/structure score."""
    score = 50
    lower = resume.lower()

    sections = ['experience', 'education', 'skills', 'summary', 'objective',
                'projects', 'certifications', 'achievements']
    found_sections = sum(1 for s in sections if s in lower)
    score += found_sections * 5          # up to +40 for sections

    bullet_count = resume.count('•') + resume.count('-') + resume.count('*')
    score += min(bullet_count * 1, 10)  # up to +10 for bullets

    if len(resume) > 300:  score += 5   # not too sparse
    if len(resume) > 3000: score -= 10  # not too long

    has_email = bool(re.search(r'[\w.+-]+@[\w-]+\.\w+', resume))
    has_phone = bool(re.search(r'\+?\d[\d\s\-().]{7,}', resume))
    if has_email: score += 5
    if has_phone: score += 5

    return min(max(score, 0), 100)


def compute_ats_score(similarity: float, kw_match: int, fmt: int) -> int:
    """Weighted composite ATS score."""
    return int(similarity * 0.4 + kw_match * 0.4 + fmt * 0.2)


def grade(score: int) -> str:
    if score >= 80: return 'Excellent'
    if score >= 65: return 'Good'
    if score >= 45: return 'Fair'
    return 'Poor'


def rule_based_suggestions(kw_data: dict, fmt_score: int, similarity: float) -> list[dict]:
    """Generate suggestions without AI."""
    tips = []

    if kw_data['missing']:
        top_missing = ', '.join(kw_data['missing'][:5])
        tips.append({'type': 'warn',
                     'text': f"Add these missing keywords from the JD: {top_missing}. "
                             "Integrate them naturally into your experience bullets."})

    if similarity < 50:
        tips.append({'type': 'warn',
                     'text': "Your resume language is quite different from the JD. "
                             "Mirror key phrases and terminology from the job posting."})

    if fmt_score < 60:
        tips.append({'type': 'tip',
                     'text': "Improve structure: include clearly labelled sections "
                             "(Summary, Experience, Skills, Education) for better ATS parsing."})

    tips.append({'type': 'tip',
                 'text': "Quantify your achievements. Replace vague statements with "
                         "metrics: 'Increased sales by 30%' outperforms 'Improved sales'."})

    tips.append({'type': 'ok',
                 'text': "Use standard section headings. ATS systems struggle with "
                         "creative headings like 'My Journey' — stick to 'Experience'."})

    tips.append({'type': 'tip',
                 'text': "Tailor your resume summary/objective for each application. "
                         "Reference the specific role and company name where possible."})

    return tips[:7]


def ai_analysis(resume: str, jd: str, role: str, base_data: dict) -> dict:
    """Call Claude for richer analysis (requires ANTHROPIC_API_KEY)."""
    prompt = f"""You are an expert ATS analyst and resume coach.

Analyze the resume against this job description for: {role}

RESUME (excerpt):
{resume[:3000]}

JOB DESCRIPTION (excerpt):
{jd[:2000]}

Return ONLY valid JSON with this structure:
{{
  "title": "<4-6 word verdict>",
  "verdict": "<2-sentence summary>",
  "strengths": ["<3-5 strengths>"],
  "gaps": ["<3-5 gaps>"],
  "suggestions": [
    {{"type": "warn|ok|tip", "text": "<actionable suggestion>"}}
  ]
}}"""

    msg = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=1000,
        messages=[{'role': 'user', 'content': prompt}]
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r'```json|```', '', raw).strip()
    ai = json.loads(raw)

    # Merge AI output with rule-based scores
    base_data.update({
        'title':       ai.get('title',       'AI Analysis Complete'),
        'verdict':     ai.get('verdict',     ''),
        'strengths':   ai.get('strengths',   []),
        'gaps':        ai.get('gaps',        []),
        'suggestions': ai.get('suggestions', []),
    })
    return base_data


# ── Routes ─────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main HTML interface."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Template Error: {str(e)}", 500

@app.errorhandler(404)
def page_not_found(e):
    return f"Flask 404: {request.url}", 404


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    POST /analyze
    Accepts JSON or multipart form data with optional PDF uploads.
    """
    resume = ''
    jd = ''
    role = ''

    if request.content_type and request.content_type.startswith('multipart/form-data'):
        resume_file = request.files.get('resume_file')
        jd_file = request.files.get('jd_file')

        if resume_file and resume_file.filename.lower().endswith('.pdf'):
            resume = extract_text_from_pdf(resume_file)
        elif resume_file:
            resume = resume_file.read().decode('utf-8', errors='ignore').strip()
        else:
            resume = request.form.get('resume', '').strip()

        if jd_file and jd_file.filename.lower().endswith('.pdf'):
            jd = extract_text_from_pdf(jd_file)
        elif jd_file:
            jd = jd_file.read().decode('utf-8', errors='ignore').strip()
        else:
            jd = request.form.get('jd', '').strip()

        role = request.form.get('role', '').strip()
    else:
        data = request.get_json(force=True) or {}
        resume = data.get('resume', '').strip()
        jd = data.get('jd', '').strip()
        role = data.get('role', '').strip()

    if not resume or not jd or not role:
        return jsonify({'error': 'resume, jd, and role are required'}), 400

    # ── Core NLP pipeline ─────────────────────────────────────────
    similarity  = tfidf_similarity(resume, jd)
    kw_data     = keyword_overlap(resume, jd)
    fmt_score   = score_format(resume)
    ats         = compute_ats_score(similarity, kw_data['match_pct'], fmt_score)

    result = {
        'ats_score':           ats,
        'keyword_match_pct':   kw_data['match_pct'],
        'skills_coverage_pct': int(similarity),
        'format_score':        fmt_score,
        'grade':               grade(ats),
        'title':               'NLP Analysis Complete',
        'verdict':             f'Your resume achieved a {ats}/100 ATS score for the {role} role.',
        'keywords_found':      kw_data['found'],
        'keywords_missing':    kw_data['missing'],
        'strengths':           [f"Strong keyword presence: {', '.join(kw_data['found'][:3])}"] if kw_data['found'] else [],
        'gaps':                [f"Missing keywords: {', '.join(kw_data['missing'][:3])}"] if kw_data['missing'] else [],
        'suggestions':         rule_based_suggestions(kw_data, fmt_score, similarity),
    }

    # ── Enrich with Claude AI if key is set ───────────────────────
    if USE_AI:
        try:
            result = ai_analysis(resume, jd, role, result)
        except Exception as e:
            result['ai_error'] = str(e)

    return jsonify(result)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'ai_enabled': USE_AI})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'true').lower() == 'true'
    print(f"🎯 ResumeIQ running on http://localhost:{port}")
    print(f"   AI Mode: {'✓ Claude enabled' if USE_AI else '✗ NLP only (set ANTHROPIC_API_KEY to enable)'}")
    app.run(host='0.0.0.0', port=port, debug=debug)
