# 🎯 ResumeIQ — AI-Powered Resume Analyzer

A full-stack AI resume analysis tool that gives job seekers ATS scores, keyword gap analysis, and actionable improvement suggestions — powered by Claude AI + NLP.

---

## 📁 Project Structure

```
resumeiq/
├── index.html          ← Standalone frontend (works in browser, no server needed)
├── app.py              ← Flask backend with NLP pipeline
├── cli.py              ← Command-line interface
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment variable template
├── utils/
│   └── nlp.py          ← Reusable NLP helpers (TF-IDF, spaCy, NLTK)
└── README.md
```

---

## 🚀 Quick Start

### Option A — Browser Only (Zero Setup)
Just open `index.html` in your browser. Enter your Anthropic API key in the UI and start analyzing immediately. No server, no install.

### Option B — Python Backend (Full NLP Pipeline)

**1. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**2. Set your API key** *(optional — enables Claude AI mode)*
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

**3. Run the server**
```bash
python app.py
# → http://localhost:5000
```

### Option C — CLI
```bash
# NLP only
python cli.py --resume my_resume.txt --jd job_post.txt --role "Data Scientist"

# With Claude AI (set ANTHROPIC_API_KEY first)
python cli.py --resume my_resume.txt --jd job_post.txt --role "SWE" --ai

# Raw JSON output
python cli.py --resume resume.txt --jd jd.txt --role "PM" --json
```

---

## 🧠 AI / NLP Concepts Used

| Concept | Implementation | File |
|---|---|---|
| **TF-IDF Similarity** | Resume vs JD cosine similarity | `utils/nlp.py` |
| **Keyword Extraction** | Top-N JD tokens + Counter | `utils/nlp.py` |
| **NER (Named Entity Recognition)** | spaCy `en_core_web_sm` | `utils/nlp.py` |
| **Tokenization** | NLTK `word_tokenize` | `utils/nlp.py` |
| **Stopword Removal** | NLTK stopwords corpus | `utils/nlp.py` |
| **LLM Analysis** | Anthropic Claude (structured JSON) | `app.py`, `cli.py` |
| **ATS Scoring** | Weighted composite formula | `utils/nlp.py` |
| **Format Heuristics** | Section/bullet/contact detection | `utils/nlp.py` |

---

## 📊 Scoring Formula

```
ATS Score = (TF-IDF Similarity × 0.4)
          + (Keyword Match %  × 0.4)
          + (Format Score     × 0.2)
```

| Grade | Score |
|---|---|
| 🟢 Excellent | 80–100 |
| 🔵 Good | 65–79 |
| 🟡 Fair | 45–64 |
| 🔴 Poor | 0–44 |

---

## 🔌 API Reference

### `POST /analyze`
**Request body:**
```json
{
  "resume": "Full resume text...",
  "jd":     "Job description text...",
  "role":   "Senior Frontend Engineer"
}
```

**Response:**
```json
{
  "ats_score": 72,
  "keyword_match_pct": 68,
  "skills_coverage_pct": 74,
  "format_score": 80,
  "grade": "Good",
  "title": "Strong but Needs Keywords",
  "verdict": "Your resume is well-structured but missing several key technical terms from the JD.",
  "keywords_found": ["react", "typescript", "aws", "node"],
  "keywords_missing": ["graphql", "docker", "terraform"],
  "strengths": ["Strong React background", "Quantified achievements"],
  "gaps": ["No GraphQL experience mentioned", "Missing cloud certifications"],
  "suggestions": [
    { "type": "warn", "text": "Add GraphQL to your Skills section if you have experience." },
    { "type": "tip",  "text": "Quantify your React projects with user counts or performance metrics." }
  ]
}
```

### `GET /health`
Returns `{ "status": "ok", "ai_enabled": true|false }`

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Vanilla HTML/CSS/JS, Google Fonts (Syne + Space Mono) |
| **Backend** | Python 3.11+, Flask, Flask-CORS |
| **NLP** | spaCy (NER), NLTK (tokenization), scikit-learn (TF-IDF) |
| **AI** | Anthropic Claude `claude-sonnet-4-20250514` |
| **Similarity** | Cosine Similarity via scikit-learn |

---

## 🛠 Extending the Project

- **PDF support**: Add `PyMuPDF` or `pdfplumber` to extract text from uploaded PDFs
- **Streamlit UI**: Replace `index.html` with `streamlit_app.py` using `st.text_area`
- **History**: Add SQLite to store past analyses with `flask-sqlalchemy`
- **Job board scraping**: Use `requests` + `BeautifulSoup` to auto-fetch JDs from LinkedIn/Indeed

---

## 📄 License
MIT — free to use, modify, and distribute.
