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

.upload-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.55rem;
    margin-bottom: 0.2rem;
}

.upload-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.42rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
    color: #dce4f5;
    font-size: 0.8rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

.upload-chip-size {
    color: #96a0b5;
    font-size: 0.75rem;
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
        return response.json()
    except Exception:
        return None

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
        if source_type == "document":
            filename = (source.get("source_filename") or title).strip()
            chunk_id = source.get("chunk_id", 0)
            lines.append(f"- **{filename}** (chunk {chunk_id})")
        elif url:
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
    direct_answer = normalize_markdown(extract_section(final_text, "Top Recommendations"))
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


def normalize_answer_payload(answer_payload):
    fallback = {
        "recommendations": [
            "Software Developer",
            "Data Analyst",
            "Machine Learning Engineer",
        ],
        "reasons": [
            "These roles align well with common technical and analytical skill paths.",
            "They remain practical, in-demand options across many industries.",
            "They offer strong flexibility for future specialization.",
        ],
        "insights": "These fallback recommendations are shown so the answer stays useful and actionable.",
        "improvement_tips": [],
    }

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

    return {
        "recommendations": recommendations,
        "reasons": reasons,
        "insights": insights,
        "improvement_tips": improvement_tips,
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


def generate_pdf(query_title, report_text, web_sources, document_sources):
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

    story.append(Paragraph("ORION AI Research Report", styles["Title"]))
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
            if section_name in {"sources", "web sources", "document sources"}:
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

    story.append(Spacer(1, 8))
    story.append(Paragraph("Document Sources", section_style))
    if document_sources:
        for index, source in enumerate(document_sources, start=1):
            filename = _clean_pdf_text(source.get("source_filename", source.get("title", f"Document {index}")))
            chunk_id = source.get("chunk_id", 0)
            content = _clean_pdf_text(source.get("content", ""))
            story.append(Paragraph(f"<b>{index}. {filename}</b> (chunk {chunk_id})", body_style))
            if content:
                snippet = content[:260] + "..." if len(content) > 260 else content
                story.append(Paragraph(snippet, body_style))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No document sources available.", body_style))

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

st.markdown("""
<div class="card">
    <div class="section-kicker">Knowledge Base</div>
    <div class="section-title">Index Local Documents</div>
    <div class="section-copy">Upload PDF or TXT files to add document-grounded retrieval alongside live web research.</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    upload_chips = []
    for uploaded_file in uploaded_files:
        size_kb = max(1, int(len(uploaded_file.getvalue()) / 1024))
        upload_chips.append(
            f"""
            <div class="upload-chip">
                <span>{uploaded_file.name}</span>
                <span class="upload-chip-size">{size_kb} KB</span>
            </div>
            """
        )
    st.markdown(
        f'<div class="upload-chip-row">{"".join(upload_chips)}</div>',
        unsafe_allow_html=True,
    )

if st.button("Index Documents", use_container_width=False):
    if not uploaded_files:
        st.warning("Please upload at least one PDF or TXT file to index.")
    else:
        files_payload = []
        for uploaded_file in uploaded_files:
            files_payload.append(
                (
                    "files",
                    (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream"),
                )
            )

        try:
            index_response = requests.post(
                f"{BACKEND_URL}/documents/index",
                files=files_payload,
                timeout=180,
            )
            index_data = safe_json(index_response)

            if index_response.status_code != 200 or not index_data or not index_data.get("success", False):
                st.error(
                    (index_data or {}).get("error", f"Failed to index documents. HTTP {index_response.status_code}")
                )
            else:
                indexed_files = index_data.get("files", [])
                summary_lines = [
                    f"- **{item.get('filename', 'Unknown file')}**: {item.get('chunks_indexed', 0)} chunks indexed"
                    for item in indexed_files
                ]
                render_markdown_panel(
                    "Index Status",
                    "Document Indexing Complete",
                    "\n".join(summary_lines),
                    "These files are now available for document retrieval.",
                )
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend for document indexing.")
        except Exception as e:
            st.error(f"Unexpected indexing error: {e}")

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

            data = safe_json(response)

            if data is None:
                st.error("Backend returned invalid JSON.")
                st.stop()

            if response.status_code != 200:
                st.error(f"Backend HTTP error: {response.status_code}")
                try:
                    st.code(response.text)
                except Exception:
                    pass
                st.stop()

            if not data.get("success", False):
                st.error(f"Error: {data.get('error', 'Unknown backend error')}")
                st.stop()

            plan_text = data.get("plan", "")
            findings_text = data.get("key_findings", data.get("findings", ""))
            final_text = data.get("final_report", data.get("final", ""))
            structured_response = data.get("structured_response", final_text)
            research_payload = data.get("research", {})
            web_sources = data.get("web_sources", []) or extract_sources(research_payload)
            document_sources = data.get("document_sources", [])
            sources = data.get("sources", []) or (web_sources + document_sources)
            metrics = build_metrics(plan_text, final_text, sources)
            answer_payload = normalize_answer_payload(data.get("answer_payload", {}))
            recommendations = answer_payload["recommendations"]
            reasons = answer_payload["reasons"]
            personalized_text = answer_payload["insights"]
            improvement_tips = answer_payload["improvement_tips"]
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

            web_sources_markdown = build_sources_markdown(web_sources, source_type="web")
            document_sources_markdown = build_sources_markdown(document_sources, source_type="document")
            insight_items = "".join(f"<li>{item}</li>" for item in quick_insights)
            st.markdown(f"""
            <div class="summary-shell">
                <div class="answering-line"><span>Answering:</span> {query}</div>
                <div class="summary-title">Quick Insights</div>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-kicker">Direct Answer</div>
                        <ul class="summary-list">
                            {insight_items}
                        </ul>
                    </div>
                    <div class="summary-card count-card">
                        <div class="summary-kicker">Web Sources</div>
                        <div class="count-value">{len(web_sources)}</div>
                        <div class="count-label">Live references used</div>
                    </div>
                    <div class="summary-card count-card">
                        <div class="summary-kicker">Document Sources</div>
                        <div class="count-value">{len(document_sources)}</div>
                        <div class="count-label">Indexed chunks used</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            top_recommendations_text = "\n".join(f"- {item}" for item in recommendations)
            st.markdown("""
            <div class="top-answer-shell">
                <div class="top-answer-kicker">Primary Output</div>
                <div class="top-answer-title">Top Recommendations</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(top_recommendations_text)

            full_report_sections = []
            full_report_sections.append("## Why These Recommendations\n" + "\n".join(f"- {item}" for item in reasons))
            full_report_sections.append("## Personalized Insight\n" + normalize_markdown(personalized_text))
            if improvement_tips:
                full_report_sections.append("## Improvement Tips\n" + "\n".join(f"- {item}" for item in improvement_tips))
            full_report_sections.append("## Key Findings\n" + (normalize_markdown(findings_text) or "_No content available._"))
            full_report_text = "\n\n".join(full_report_sections)

            with st.expander("🧠 Research Plan", expanded=False):
                st.markdown(normalize_markdown(plan_text) or "_No content available._")

            with st.expander("Full Report", expanded=False):
                st.markdown(full_report_text)

            combined_sources_text = (
                f"## Web Sources\n{normalize_markdown(web_sources_markdown) or '_No web sources available._'}\n\n"
                f"## Document Sources\n{normalize_markdown(document_sources_markdown) or '_No document sources available._'}"
            )
            with st.expander(f"Sources ({len(web_sources) + len(document_sources)})", expanded=False):
                st.markdown(combined_sources_text)

            pdf_file = generate_pdf(query, structured_response, web_sources, document_sources)
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
