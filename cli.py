#!/usr/bin/env python3
"""
cli.py — Run ResumeIQ analysis from the command line
=====================================================
Usage:
    python cli.py --resume path/to/resume.txt --jd path/to/jd.txt --role "Data Scientist"
    python cli.py --resume resume.txt --jd jd.txt --role "SWE" --ai
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.nlp import full_analysis, tfidf_sim, keyword_gap, format_score, ats_score, grade


def print_report(r: dict, role: str):
    score = r['ats_score']
    bar   = '█' * (score // 5) + '░' * (20 - score // 5)
    grade_color = {
        'Excellent': '\033[92m', 'Good': '\033[94m',
        'Fair': '\033[93m', 'Poor': '\033[91m'
    }.get(r['grade'], '')
    reset = '\033[0m'

    print(f"\n{'═'*54}")
    print(f"  🎯  ResumeIQ Analysis — {role}")
    print(f"{'═'*54}")
    print(f"\n  ATS Score   [{bar}] {grade_color}{score}/100  {r['grade']}{reset}")
    print(f"  Keyword Match  : {r['keyword_match_pct']}%")
    print(f"  Similarity     : {r['similarity_pct']}%")
    print(f"  Format Score   : {r['format_score']}/100")

    print(f"\n  ✓ Keywords Found ({len(r['keywords_found'])}):")
    for k in r['keywords_found']:
        print(f"      \033[92m+ {k}\033[0m")

    print(f"\n  ✗ Keywords Missing ({len(r['keywords_missing'])}):")
    for k in r['keywords_missing']:
        print(f"      \033[91m- {k}\033[0m")

    print(f"\n{'═'*54}\n")


def main():
    parser = argparse.ArgumentParser(description='ResumeIQ CLI Analyzer')
    parser.add_argument('--resume', required=True, help='Path to resume .txt file')
    parser.add_argument('--jd',     required=True, help='Path to job description .txt file')
    parser.add_argument('--role',   required=True, help='Target role title')
    parser.add_argument('--json',   action='store_true', help='Output raw JSON')
    parser.add_argument('--ai',     action='store_true', help='Use Claude AI (needs ANTHROPIC_API_KEY)')
    args = parser.parse_args()

    try:
        resume = open(args.resume, encoding='utf-8').read()
        jd     = open(args.jd,     encoding='utf-8').read()
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}"); sys.exit(1)

    result = full_analysis(resume, jd)

    # Optionally enrich with Claude
    if args.ai:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("⚠  ANTHROPIC_API_KEY not set. Running NLP-only mode.")
        else:
            import anthropic, re
            client = anthropic.Anthropic(api_key=api_key)
            prompt = f"""Analyze this resume for the role: {args.role}

RESUME: {resume[:2500]}

JOB DESCRIPTION: {jd[:1500]}

Return JSON: {{"title":"...","verdict":"...","strengths":[...],"gaps":[...],"suggestions":[{{"type":"warn|ok|tip","text":"..."}}]}}"""
            msg = client.messages.create(model='claude-sonnet-4-20250514', max_tokens=900,
                                          messages=[{'role':'user','content':prompt}])
            raw = re.sub(r'```json|```','', msg.content[0].text).strip()
            try:
                ai = json.loads(raw)
                result.update(ai)
            except Exception:
                pass

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result, args.role)


if __name__ == '__main__':
    main()
