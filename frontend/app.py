import time
from datetime import datetime
from io import BytesIO
import re

import pandas as pd
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

st.set_page_config(
    page_title="ORION AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BACKEND_URL = "https://orion-backend-s0e6.onrender.com"

st.markdown("""
<style>
:root {
    --bg-1: #070b17;
    --bg-2: #0d1222;
    --bg-3: #12192d;
    --panel: rgba(255,255,255,0.045);
    --panel-strong: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.08);
    --text: #f3f6ff;
    --muted: #9aa4b2;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(124,58,237,0.16), transparent 24%),
        radial-gradient(circle at top right, rgba(59,130,246,0.10), transparent 22%),
        linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
    color: var(--text);
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

html, body, [class*="css"] {
    color: var(--text);
    font-family: "Georgia", "Times New Roman", serif;
}

p, li, div {
    line-height: 1.55;
}

.hero-wrap {
    text-align: center;
    margin-bottom: 1.35rem;
    animation: fadeUp 0.8s ease-out;
}

.badge {
    display: inline-block;
    padding: 0.42rem 0.85rem;
    border-radius: 999px;
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.09);
    color: #d8dcef;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(10px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}

.hero-title {
    font-size: 3.45rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin-bottom: 0.5rem;
    line-height: 0.98;
    text-shadow: 0 14px 34px rgba(0,0,0,0.3);
}

.hero-subtitle {
    max-width: 700px;
    margin: 0 auto;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.65;
    letter-spacing: 0.01em;
}

.search-shell {
    max-width: 860px;
    margin: 1.45rem auto 0.9rem auto;
    padding: 1.1rem 1.1rem 0.95rem 1.1rem;
    border-radius: 26px;
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow:
        0 16px 40px rgba(0,0,0,0.22),
        inset 0 0 0 1px rgba(255,255,255,0.02);
    backdrop-filter: blur(20px);
    animation: fadeUp 0.9s ease-out;
}

.search-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #c8b7ff;
    font-weight: 700;
    margin-bottom: 0.28rem;
}

.search-caption {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.5;
}

.section-divider {
    position: relative;
    margin: 1.35rem 0 1rem 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.5), rgba(96,165,250,0.4), transparent);
}

.section-divider::after {
    content: "";
    position: absolute;
    left: 50%;
    top: -3px;
    width: 120px;
    height: 7px;
    transform: translateX(-50%);
    background: radial-gradient(circle, rgba(255,255,255,0.28), transparent 68%);
}

.stTextInput {
    margin-bottom: 0.15rem;
}

.stTextInput > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.stTextInput > div > div > div {
    background: rgba(17, 24, 39, 0.96) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(8px);
}

.stTextInput > div > div > div {
    background: rgba(10, 15, 28, 0.98) !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 20px !important;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.03),
                inset 0 1px 1px rgba(255,255,255,0.02),
                inset 0 1px 2px rgba(0,0,0,0.28) !important;
}

.stTextInput > div > div > div:focus-within {
    border: 1px solid rgba(96, 165, 250, 0.4) !important;
    box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.1),
                inset 0 1px 2px rgba(0,0,0,0.3) !important;
}

.stTextInput input {
    font-size: 1rem !important;
    padding-top: 0.2rem !important;
    padding-bottom: 0.2rem !important;
}

.stTextInput input::placeholder {
    color: rgba(255,255,255,0.45) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.45) !important;
    opacity: 1 !important;
}

.stButton {
    display: flex;
    justify-content: center;
    margin-top: 0.45rem;
    margin-bottom: 0.4rem;
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #4f46e5, #7c3aed, #0ea5e9) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    padding: 0.78rem 1.35rem !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 14px 28px rgba(79,70,229,0.24), inset 0 1px 0 rgba(255,255,255,0.14) !important;
    transition: all 0.18s ease !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 30px rgba(79,70,229,0.28), inset 0 1px 0 rgba(255,255,255,0.16) !important;
}

.stButton > button:active, .stDownloadButton > button:active {
    transform: translateY(0);
}

.card {
    background: linear-gradient(180deg, var(--panel-strong), var(--panel));
    border: 1px solid rgba(255,255,255,0.085);
    border-radius: 22px;
    padding: 1.05rem 1.05rem 0.95rem 1.05rem;
    box-shadow: 0 14px 34px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.03);
    backdrop-filter: blur(16px);
    margin-top: 0.8rem;
    animation: fadeUp 0.55s ease-out;
}

.section-kicker {
    color: #c8b7ff;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.35rem;
}

.section-title {
    font-size: 1.18rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
}

.section-copy {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.58;
    margin-top: -0.15rem;
    margin-bottom: 0.2rem;
}

.pipeline {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 0.9rem;
}

.pipe-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.025));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 0.85rem;
    min-height: 118px;
}

.pipe-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    margin-bottom: 0.55rem;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    box-shadow: 0 0 12px rgba(124,58,237,0.42);
}

.pipe-title {
    font-size: 0.94rem;
    font-weight: 700;
    color: var(--text);
}

.pipe-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.28rem;
}

.status-box {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.028));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.95rem 0.95rem;
    margin-top: 0.75rem;
    position: relative;
    overflow: hidden;
}

.status-title {
    color: #c8b7ff;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.35rem;
}

.status-main {
    font-size: 1rem;
    font-weight: 700;
}

.status-sub {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 0.2rem;
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 9px;
    margin-top: 0.75rem;
}

.status-step {
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.028);
    border-radius: 14px;
    padding: 0.68rem 0.72rem;
}

.status-step-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #c4b5fd;
    margin-bottom: 0.25rem;
}

.status-step-text {
    font-size: 0.88rem;
    color: #e8ecff;
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 0.8rem;
}

.metric-box {
    background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.9rem 1rem;
}

.metric-label {
    color: var(--muted);
    font-size: 0.76rem;
}

.metric-value {
    color: var(--text);
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 0.12rem;
}

.report-body {
    color: #eef2ff;
    font-size: 0.98rem;
    line-height: 1.75;
}

.markdown-panel {
    margin-top: 0.35rem;
}

.markdown-panel + .markdown-panel {
    margin-top: 0.45rem;
}

div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: #f4f7ff;
    letter-spacing: -0.02em;
    margin-top: 0.2rem;
    margin-bottom: 0.65rem;
}

div[data-testid="stMarkdownContainer"] p {
    color: #dbe3f0;
    line-height: 1.75;
}

div[data-testid="stMarkdownContainer"] ul,
div[data-testid="stMarkdownContainer"] ol {
    color: #dbe3f0;
    padding-left: 1.25rem;
}

div[data-testid="stMarkdownContainer"] li {
    margin-bottom: 0.38rem;
}

div[data-testid="stMarkdownContainer"] strong {
    color: #ffffff;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.075);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.018));
    margin-top: 0.45rem;
    overflow: hidden;
    box-shadow: 0 10px 24px rgba(0,0,0,0.14);
}

div[data-testid="stExpander"] details summary {
    padding: 0.12rem 0.28rem;
}

div[data-testid="stExpander"] summary p {
    font-size: 0.96rem !important;
    font-weight: 650 !important;
}

.summary-shell {
    margin-top: 0.55rem;
}

.summary-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}

.answering-line {
    color: #e7ecff;
    font-size: 0.96rem;
    font-weight: 600;
    margin-bottom: 0.7rem;
}

.answering-line span {
    color: #a5b4fc;
}

.summary-grid {
    display: grid;
    grid-template-columns: 1.6fr 0.7fr 0.7fr;
    gap: 12px;
}

.summary-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.025));
    border: 1px solid rgba(255,255,255,0.085);
    border-radius: 18px;
    padding: 0.9rem 0.95rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

.summary-kicker {
    color: #c8b7ff;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.35rem;
}

.summary-list {
    margin: 0;
    padding-left: 1rem;
    color: #dbe3f0;
}

.summary-list li {
    margin-bottom: 0.28rem;
}

.count-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 116px;
}

.count-value {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1;
    color: #f8fbff;
    margin-bottom: 0.25rem;
}

.count-label {
    color: var(--muted);
    font-size: 0.8rem;
}

.top-answer-shell {
    margin-top: 0.55rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 1rem 1rem 0.9rem 1rem;
    box-shadow: 0 16px 36px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.04);
}

.top-answer-kicker {
    color: #c8b7ff;
    font-size: 0.73rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.35rem;
}

.top-answer-title {
    font-size: 1.28rem;
    font-weight: 750;
    margin-bottom: 0.35rem;
}

.report-section {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.07);
}

.report-section:first-child {
    border-top: none;
    padding-top: 0;
    margin-top: 0;
}

.report-heading {
    font-size: 1.06rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 0.55rem;
}

.report-paragraph {
    color: #dbe3f0;
    margin-bottom: 0.65rem;
}

.report-list {
    margin: 0.15rem 0 0.75rem 1.1rem;
    color: #dbe3f0;
}

.report-list li {
    margin-bottom: 0.38rem;
}

.source-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 0.9rem;
    margin-bottom: 0.8rem;
}

.source-title {
    font-weight: 700;
    color: #eef2ff;
    margin-bottom: 0.2rem;
}

.source-url {
    color: #93c5fd;
    font-size: 0.8rem;
    word-break: break-all;
}

.source-snippet {
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-top: 0.35rem;
}

.shimmer {
    position: relative;
    overflow: hidden;
    border-radius: 12px;
    background: rgba(255,255,255,0.05);
    height: 10px;
    margin-top: 0.75rem;
    border: 1px solid rgba(255,255,255,0.06);
}

.shimmer::before {
    content: "";
    position: absolute;
    top: 0;
    left: -35%;
    width: 35%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.24), transparent);
    animation: shimmerMove 1.4s infinite;
}

.footer {
    text-align: center;
    color: var(--muted);
    margin-top: 2.3rem;
    font-size: 0.9rem;
    opacity: 0.95;
}

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #5b5cf0, #8b5cf6) !important;
}

@keyframes shimmerMove {
    0% { left: -35%; }
    100% { left: 100%; }
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 900px) {
    .pipeline, .metric-row, .status-grid, .summary-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 2.7rem;
    }
}
</style>
""", unsafe_allow_html=True)

def safe_json(response):
    try:
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def backend_error_message(response, data, parse_error=None):
    status_code = getattr(response, "status_code", "unknown")
    if isinstance(data, dict):
        error_text = data.get("error") or data.get("message")
        details_text = data.get("details")
        if error_text and details_text:
            return f"Backend HTTP {status_code}: {error_text} ({details_text})"
        if error_text:
            return f"Backend HTTP {status_code}: {error_text}"

    if parse_error:
        return f"Backend HTTP {status_code}: invalid JSON response. {parse_error}"

    return f"Backend HTTP {status_code}: unexpected backend response."

def extract_sources(research_payload):
    if not isinstance(research_payload, dict):
        return []
    results = research_payload.get("results", [])
    cleaned = []
    for item in results[:5]:
        cleaned.append({
            "title": item.get("title", "Untitled source"),
            "url": item.get("url", ""),
            "content": item.get("content", "")
        })
    return cleaned


def build_sources_markdown(sources, source_type="web"):
    if not sources:
        return "- No sources available."

    lines = []
    for source in sources:
        title = (source.get("title") or "Untitled source").strip()
        url = (source.get("url") or "").strip()
        if url:
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)


def normalize_markdown(text):
    if not text:
        return ""

    normalized_lines = []
    for raw_line in str(text).replace("\r\n", "\n").split("\n"):
        line = raw_line.rstrip()
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        stripped = re.sub(r"^(#{1,6})(\S)", r"\1 \2", stripped)
        stripped = re.sub(r"^(\d+\.)(\*\*)", r"\1 \2", stripped)
        stripped = re.sub(r"^(\d+\.)(\S)", r"\1 \2", stripped)
        stripped = re.sub(r"^([*-])(\S)", r"\1 \2", stripped)
        stripped = re.sub(r"^(\d+)\.(\d+\s+\*\*)", r"\1. \2", stripped)
        stripped = re.sub(r"\*\*(\S)", r"**\1", stripped)
        stripped = re.sub(r"(\S)\*\*", r"\1**", stripped)

        normalized_lines.append(indent + stripped)

    normalized = "\n".join(normalized_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()

def build_metrics(plan_text, final_text, sources):
    plan_len = len(plan_text.split()) if plan_text else 0
    report_len = len(final_text.split()) if final_text else 0
    return {
        "Sources": len(sources),
        "Plan Words": plan_len,
        "Report Words": report_len
    }


def extract_quick_insights(findings_text, final_text, limit=5):
    cleaned_findings = normalize_markdown(findings_text)
    bullets = []
    for line in cleaned_findings.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            insight = stripped[2:].strip()
            insight = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", insight)
            insight = insight.replace("**", "").replace("`", "")
            bullets.append(insight)
        if len(bullets) >= limit:
            break

    if bullets:
        return bullets[:limit]

    fallback = []
    cleaned_final = normalize_markdown(final_text)
    for paragraph in cleaned_final.split("\n\n"):
        snippet = " ".join(paragraph.split()).strip()
        if not snippet or snippet.startswith("#"):
            continue
        fallback.append(snippet[:140] + "..." if len(snippet) > 140 else snippet)
        if len(fallback) >= limit:
            break
    return fallback


def select_query_focused_insights(query, findings_text, final_text, limit=5):
    query_terms = [term for term in re.findall(r"[a-zA-Z0-9]+", (query or "").lower()) if len(term) > 2]
    candidates = extract_quick_insights(findings_text, final_text, limit=8)

    if not candidates:
        return []

    scored = []
    for item in candidates:
        lowered = item.lower()
        score = sum(1 for term in query_terms if term in lowered)
        scored.append((score, len(item), item))

    scored.sort(key=lambda row: (-row[0], row[1]))
    selected = [item for _, _, item in scored[:limit]]
    return selected[:limit]


def extract_section(markdown_text, section_name):
    if not markdown_text:
        return ""

    lines = markdown_text.splitlines()
    collected = []
    inside = False
    target = section_name.strip().lower()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip().lower()
            if inside and heading != target:
                break
            inside = heading == target
            continue
        if inside:
            collected.append(line)

    return "\n".join(collected).strip()


def extract_direct_answer_bullets(final_text, limit=5):
    direct_answer = normalize_markdown(
        extract_section(final_text, "Direct Answer") or extract_section(final_text, "Top Recommendations")
    )
    bullets = []
    for line in direct_answer.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            item = stripped[2:].strip()
            item = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", item)
            item = item.replace("**", "").replace("`", "")
            bullets.append(item)
        if len(bullets) >= limit:
            break
    return bullets[:limit]


def has_user_facing_content(text):
    cleaned = normalize_markdown(text or "")
    if not cleaned:
        return False
    blocked_phrases = [
        "the answer should",
        "the response should",
        "the clearest response",
        "depends on the retrieved evidence",
        "depends on the evidence",
        "if reporting is mixed",
        "this fallback",
        "aligned to the query",
        "best supported by",
    ]
    lowered = cleaned.lower()
    return not any(phrase in lowered for phrase in blocked_phrases)


def has_user_facing_list(items):
    if not items:
        return False
    return any(has_user_facing_content(item) for item in items if isinstance(item, str))


def fallback_answer_payload(query_type="web", user_query=""):
    if query_type == "resume":
        return {
            "primary_title": "Direct Answer",
            "recommendations": [
                f"For {user_query or 'this resume question'}, the best-fit roles are usually the ones most clearly supported by your projects, tools, and measurable results.",
            ],
            "reasons_title": "Why",
            "reasons": [
                "Role fit is strongest where your achievements are easiest to prove.",
                "Employers usually respond better to focused positioning than broad self-description.",
                "Clear outcomes and ownership matter more than long generic skill lists.",
            ],
            "insights_title": "Key Insights",
            "insights": "- Strong projects usually matter more than broad claims.\n- A narrower role target usually creates a stronger resume.",
            "improvement_title": "",
            "improvement_tips": [],
            "extra_sections": [
                {
                    "title": "Conclusion",
                    "items": [
                        "A focused role direction will usually produce a better resume strategy than aiming everywhere at once.",
                    ],
                },
            ],
        }

    if query_type == "study":
        return {
            "primary_title": "Direct Answer",
            "recommendations": [
                f"For {user_query or 'this study question'}, the clearest explanation starts with the core idea and then moves to the most important supporting points.",
            ],
            "reasons_title": "Why",
            "reasons": [
                "Students usually remember concepts better when the explanation is simple first.",
                "Repeated themes and likely question angles matter most for revision.",
                "Clear examples make recall easier than abstract summaries.",
            ],
            "insights_title": "Key Insights",
            "insights": "- Begin with the definition and scope.\n- Then move to examples, likely questions, and revision points.",
            "improvement_title": "",
            "improvement_tips": [],
            "extra_sections": [
                {
                    "title": "Conclusion",
                    "items": [
                        "A strong study answer makes the topic easier to recall and explain under pressure.",
                    ],
                },
            ],
        }

    if query_type == "interview":
        return {
            "primary_title": "Direct Answer",
            "recommendations": [
                f"For {user_query or 'this interview question'}, the strongest direction is usually a concrete example that shows what you did, why you did it, and what result it produced.",
            ],
            "reasons_title": "Why",
            "reasons": [
                "Interviewers usually trust specific examples more than abstract claims.",
                "Decision-making and trade-offs make answers feel stronger and more credible.",
                "Clear outcomes help the answer sound real and memorable.",
            ],
            "insights_title": "Key Insights",
            "insights": "- Specific examples matter more than theory.\n- Outcomes and trade-offs usually drive the strongest follow-up questions.",
            "improvement_title": "",
            "improvement_tips": [],
            "extra_sections": [
                {
                    "title": "Conclusion",
                    "items": [
                        "A direct example-based answer will usually perform better than a broad textbook-style one.",
                    ],
                }
            ],
        }

    if query_type == "web":
        return {
            "primary_title": "Direct Answer",
            "recommendations": [
                f"For {user_query or 'the current question'}, the most likely outcome is uncertain, but the evidence points more toward a contested or unclear result than a simple decisive one.",
            ],
            "reasons_title": "Why",
            "reasons": [
                "Current reporting does not support a simple one-sided conclusion.",
                "Political, strategic, and external pressures usually shape the outcome together.",
                "When certainty is limited, the clearest answer is the likeliest conclusion stated plainly.",
            ],
            "insights_title": "Key Insights",
            "insights": "- A clear winner is often hard to predict in complex geopolitical conflicts.\n- External alliances and escalation risks usually matter as much as raw capability.",
            "improvement_title": "",
            "improvement_tips": [],
            "extra_sections": [
                {
                    "title": "Conclusion",
                    "items": [
                        "The likeliest outcome is prolonged uncertainty or confrontation rather than a clean decisive result.",
                    ],
                }
            ],
        }

    return {
        "primary_title": "Direct Answer",
        "recommendations": [
            f"For {user_query or 'the current question'}, the most likely conclusion is uncertain but still points more toward a contested or incomplete outcome than a simple decisive one.",
        ],
        "reasons_title": "Why",
        "reasons": [
            "The available information is limited.",
            "The answer stays cautious while still providing a real conclusion.",
            "A direct answer is more useful than generic commentary.",
        ],
        "insights_title": "Key Insights",
        "insights": "- Limited evidence usually means more uncertainty.\n- A plain-language conclusion is still better than meta commentary.",
        "improvement_title": "",
        "improvement_tips": [],
        "extra_sections": [
            {
                "title": "Conclusion",
                "items": [
                    "The strongest fallback is still a direct answer, even when confidence is limited.",
                ],
            }
        ],
    }


def normalize_answer_payload(answer_payload, query_type="general", user_query=""):
    fallback = fallback_answer_payload(query_type=query_type or "web", user_query=user_query)

    if not isinstance(answer_payload, dict):
        return fallback

    recommendations = [
        item.strip() for item in answer_payload.get("recommendations", [])
        if isinstance(item, str) and item.strip()
    ][:5]
    reasons = [
        item.strip() for item in answer_payload.get("reasons", [])
        if isinstance(item, str) and item.strip()
    ][:5]
    insights = answer_payload.get("insights", "")
    insights = insights.strip() if isinstance(insights, str) else ""
    improvement_tips = [
        item.strip() for item in answer_payload.get("improvement_tips", [])
        if isinstance(item, str) and item.strip()
    ][:5]

    if not recommendations:
        recommendations = fallback["recommendations"]
    if not reasons:
        reasons = fallback["reasons"]
    if not insights:
        insights = fallback["insights"]

    extra_sections = []
    raw_extra_sections = answer_payload.get("extra_sections", fallback.get("extra_sections", []))
    if isinstance(raw_extra_sections, list):
        for section in raw_extra_sections[:6]:
            if not isinstance(section, dict):
                continue
            title = section.get("title", "")
            title = title.strip() if isinstance(title, str) else ""
            items = [
                item.strip() for item in section.get("items", [])
                if isinstance(item, str) and item.strip()
            ][:6]
            if title and items:
                extra_sections.append({"title": title, "items": items})

    return {
        "primary_title": answer_payload.get("primary_title", fallback["primary_title"]),
        "recommendations": recommendations,
        "reasons_title": answer_payload.get("reasons_title", fallback["reasons_title"]),
        "reasons": reasons,
        "insights_title": answer_payload.get("insights_title", fallback["insights_title"]),
        "insights": insights,
        "improvement_title": answer_payload.get("improvement_title", fallback["improvement_title"]),
        "improvement_tips": improvement_tips or fallback.get("improvement_tips", []),
        "extra_sections": extra_sections,
    }

def build_chart():
    chart_data = pd.DataFrame({
        "Quality Dimension": ["Clarity", "Depth", "Usability", "Presentation"],
        "Score": [9, 9, 8, 9]
    })
    st.bar_chart(chart_data.set_index("Quality Dimension"), use_container_width=True)


def render_markdown_panel(kicker, title, text, description=""):
    cleaned_text = normalize_markdown(text) or "_No content available._"
    container = st.container()
    with container:
        if kicker:
            st.caption(kicker.upper())
        st.markdown(f"### {title}")
        if description:
            st.caption(description)
        st.markdown(cleaned_text)

def _clean_pdf_text(value):
    return (value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _strip_markdown_for_pdf(value):
    text = value or ""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 - \2", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return text


def generate_pdf(query_title, report_text, web_sources, mode_label="Web Research"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        topMargin=42,
        bottomMargin=42,
        leftMargin=44,
        rightMargin=44,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    meta_style = ParagraphStyle(
        "PdfMeta",
        parent=styles["BodyText"],
        fontSize=9,
        textColor=colors.HexColor("#5b6475"),
        leading=12,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "PdfSection",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#172033"),
        spaceBefore=8,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "PdfBody",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=15,
        spaceAfter=7,
    )
    bullet_style = ParagraphStyle(
        "PdfBullet",
        parent=body_style,
        leftIndent=14,
        bulletIndent=2,
        spaceAfter=5,
    )
    story = []
    safe_query = _clean_pdf_text(query_title or "Research Brief")
    safe_date = _clean_pdf_text(datetime.now().strftime("%B %d, %Y"))

    story.append(Paragraph(f"ORION AI - { _clean_pdf_text(mode_label) }", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Query:</b> {safe_query}", meta_style))
    story.append(Paragraph(f"<b>Date:</b> {safe_date}", meta_style))
    story.append(Spacer(1, 10))

    skip_embedded_sources = False
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        plain_line = _strip_markdown_for_pdf(line)
        safe_line = _clean_pdf_text(plain_line)
        if line.startswith("## "):
            section_name = plain_line[3:].strip().lower()
            if section_name in {"sources", "web sources"}:
                skip_embedded_sources = True
                continue
            story.append(Paragraph(safe_line[3:], section_style))
            continue
        if line.startswith("# "):
            story.append(Paragraph(safe_line[2:], section_style))
            continue
        if skip_embedded_sources:
            continue
        if line.startswith(("- ", "* ")):
            story.append(Paragraph(safe_line[2:], bullet_style, bulletText="•"))
            continue
        if ":" in line and len(line) < 90 and not line.endswith("."):
            story.append(Paragraph(f"<b>{safe_line}</b>", section_style))
            continue

        story.append(Paragraph(safe_line, body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Web Sources", section_style))
    if web_sources:
        for index, source in enumerate(web_sources, start=1):
            title = _clean_pdf_text(source.get("title", f"Source {index}"))
            url = _clean_pdf_text(source.get("url", ""))
            content = _clean_pdf_text(source.get("content", ""))
            story.append(Paragraph(f"<b>{index}. {title}</b>", body_style))
            if url:
                story.append(Paragraph(url, meta_style))
            if content:
                snippet = content[:260] + "..." if len(content) > 260 else content
                story.append(Paragraph(snippet, body_style))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No web sources available.", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

st.markdown("""
<div class="hero-wrap">
    <div class="badge">Autonomous Research System for Insight Generation</div>
    <div class="hero-title">🚀 ORION AI</div>
    <div class="hero-subtitle">
        Premium AI research engine for structured planning, live web-backed analysis, and polished report generation.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="search-shell">
    <div class="search-label">Research Brief</div>
    <div class="search-caption">Frame your topic clearly to generate a more focused plan, stronger sources, and a sharper final report.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

query = st.text_input(
    "",
    placeholder="Describe what you want to research...",
    label_visibility="collapsed"
)

if st.button("Run Research", use_container_width=False):
    if not query.strip():
        st.warning("Please enter a research topic.")
    else:
        st.markdown("""
        <div class="card">
            <div class="section-kicker">Workflow</div>
            <div class="section-title">Agent Workflow</div>
            <div class="section-copy">The run is organized as a premium research pipeline, turning a brief into a structured plan, evidence set, and final narrative.</div>
            <div class="pipeline">
                <div class="pipe-card">
                    <div class="pipe-dot"></div>
                    <div class="pipe-title">Planner</div>
                    <div class="pipe-sub">Breaks the topic into a clear strategy.</div>
                </div>
                <div class="pipe-card">
                    <div class="pipe-dot"></div>
                    <div class="pipe-title">Researcher</div>
                    <div class="pipe-sub">Finds relevant web-backed insights.</div>
                </div>
                <div class="pipe-card">
                    <div class="pipe-dot"></div>
                    <div class="pipe-title">Writer</div>
                    <div class="pipe-sub">Builds a polished final report.</div>
                </div>
                <div class="pipe-card">
                    <div class="pipe-dot"></div>
                    <div class="pipe-title">Exporter</div>
                    <div class="pipe-sub">Packages the result into PDF.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        status_box = st.empty()

        status_box.markdown("""
        <div class="status-box">
            <div class="status-title">Live Status</div>
            <div class="status-main">⚡ AI agents are working...</div>
            <div class="status-sub">Planning, searching, and composing a structured research brief.</div>
            <div class="status-grid">
                <div class="status-step">
                    <div class="status-step-label">Planner</div>
                    <div class="status-step-text">Building the research outline</div>
                </div>
                <div class="status-step">
                    <div class="status-step-label">Researcher</div>
                    <div class="status-step-text">Collecting live supporting evidence</div>
                </div>
                <div class="status-step">
                    <div class="status-step-label">Writer</div>
                    <div class="status-step-text">Assembling the final polished report</div>
                </div>
            </div>
            <div class="shimmer"></div>
        </div>
        """, unsafe_allow_html=True)

        progress = st.progress(0)

        try:
            progress.progress(12)
            time.sleep(0.15)

            response = requests.post(
                f"{BACKEND_URL}/research",
                params={"query": query},
                timeout=180
            )

            progress.progress(62)

            data, parse_error = safe_json(response)

            if data is None:
                st.error(backend_error_message(response, None, parse_error))
                try:
                    st.code(response.text[:1500])
                except Exception:
                    pass
                st.stop()

            if response.status_code != 200:
                st.error(backend_error_message(response, data))
                try:
                    st.code(response.text[:1500])
                except Exception:
                    pass
                st.stop()

            if not data.get("success", False):
                st.error(backend_error_message(response, data))
                st.stop()

            plan_text = data.get("plan", "")
            findings_text = data.get("key_findings", data.get("findings", ""))
            final_text = data.get("final_report", data.get("final", ""))
            structured_response = data.get("structured_response", final_text)
            research_payload = data.get("research", {})
            web_sources = data.get("web_sources", []) or data.get("sources", []) or extract_sources(research_payload)
            sources = web_sources
            debug_source_count = data.get("research", {}).get("debug_source_count", len(web_sources))
            metrics = build_metrics(plan_text, final_text, sources)
            query_type = data.get("query_type", "web")
            mode_label = data.get("mode", "Web Research")
            answer_payload = normalize_answer_payload(
                data.get("answer_payload", {}),
                query_type=query_type,
                user_query=query,
            )
            primary_title = answer_payload["primary_title"]
            recommendations = answer_payload["recommendations"]
            reasons_title = answer_payload["reasons_title"]
            reasons = answer_payload["reasons"]
            insights_title = answer_payload["insights_title"]
            personalized_text = answer_payload["insights"]
            improvement_title = answer_payload["improvement_title"]
            improvement_tips = answer_payload["improvement_tips"]
            extra_sections = answer_payload["extra_sections"]
            quick_insights = recommendations[:5]

            progress.progress(100)

            status_box.markdown("""
            <div class="status-box">
                <div class="status-title">Live Status</div>
                <div class="status-main">✅ Research completed successfully</div>
                <div class="status-sub">ORION generated a source-grounded answer with supporting context and export-ready output.</div>
                <div class="status-grid">
                    <div class="status-step">
                        <div class="status-step-label">Plan</div>
                        <div class="status-step-text">Research strategy prepared</div>
                    </div>
                    <div class="status-step">
                        <div class="status-step-label">Evidence</div>
                        <div class="status-step-text">Sources collected and distilled</div>
                    </div>
                    <div class="status-step">
                        <div class="status-step-label">Delivery</div>
                        <div class="status-step-text">Report and PDF are ready</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Total Sources</div>
                    <div class="metric-value">{metrics["Sources"]}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Plan Words</div>
                    <div class="metric-value">{metrics["Plan Words"]}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Report Words</div>
                    <div class="metric-value">{metrics["Report Words"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"DEBUG: Sources fetched = {debug_source_count}")

            web_sources_markdown = build_sources_markdown(web_sources, source_type="web")
            insight_items = "".join(f"<li>{item}</li>" for item in quick_insights)
            st.markdown(f"""
            <div class="summary-shell">
                <div class="answering-line"><span>Answering:</span> {query}</div>
                <div class="answering-line"><span>Mode:</span> {mode_label}</div>
                <div class="summary-title">Quick Insights</div>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-kicker">{primary_title}</div>
                        <ul class="summary-list">
                            {insight_items}
                        </ul>
                    </div>
                    <div class="summary-card count-card">
                        <div class="summary-kicker">Sources</div>
                        <div class="count-value">{len(web_sources)}</div>
                        <div class="count-label">Live references used</div>
                    </div>
                    <div class="summary-card count-card">
                        <div class="summary-kicker">Mode</div>
                        <div class="count-value">{mode_label}</div>
                        <div class="count-label">Active answer style</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            top_recommendations_text = "\n".join(f"- {item}" for item in recommendations)
            st.markdown(f"""
            <div class="top-answer-shell">
                <div class="top-answer-kicker">Primary Output</div>
                <div class="top-answer-title">{primary_title}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(top_recommendations_text)

            if has_user_facing_list(reasons):
                with st.expander(reasons_title, expanded=False):
                    st.markdown("\n".join(f"- {item}" for item in reasons if has_user_facing_content(item)))

            if has_user_facing_content(personalized_text):
                with st.expander(insights_title, expanded=False):
                    st.markdown(normalize_markdown(personalized_text))

            if improvement_tips:
                with st.expander(improvement_title, expanded=False):
                    st.markdown("\n".join(f"- {item}" for item in improvement_tips))

            for section in extra_sections:
                visible_items = [item for item in section["items"] if has_user_facing_content(item)]
                if not visible_items:
                    continue
                with st.expander(section["title"], expanded=False):
                    st.markdown("\n".join(f"- {item}" for item in visible_items))

            with st.expander("🧠 Research Plan", expanded=False):
                st.markdown(normalize_markdown(plan_text) or "_No content available._")

            with st.expander("Key Findings", expanded=False):
                st.markdown(normalize_markdown(findings_text) or "_No content available._")

            combined_sources_text = (
                f"## Sources\n{normalize_markdown(web_sources_markdown) or '_No web sources available._'}"
            )
            with st.expander(f"Sources ({len(web_sources)})", expanded=False):
                st.markdown(combined_sources_text)

            pdf_file = generate_pdf(query, structured_response, web_sources, mode_label=mode_label)
            st.download_button(
                "📄 Download Final Report (PDF)",
                pdf_file,
                file_name="orion_report.pdf",
                mime="application/pdf"
            )

        except requests.exceptions.Timeout:
            st.error("The request took too long. Try a shorter topic or check backend logs.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Check your Render URL.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.markdown("<div class='footer'>✨ Built by Aniruddha Pathak</div>", unsafe_allow_html=True)
