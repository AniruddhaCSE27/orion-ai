import time
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from reportlab.lib.styles import getSampleStyleSheet
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
    padding-bottom: 2rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

html, body, [class*="css"] {
    color: var(--text);
}

.hero-wrap {
    text-align: center;
    margin-bottom: 1.4rem;
    animation: fadeUp 0.8s ease-out;
}

.badge {
    display: inline-block;
    padding: 0.38rem 0.78rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.07);
    color: #d8dcef;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.9rem;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin-bottom: 0.45rem;
    line-height: 1.0;
}

.hero-subtitle {
    max-width: 760px;
    margin: 0 auto;
    color: var(--muted);
    font-size: 1.03rem;
    line-height: 1.6;
}

.search-shell {
    max-width: 860px;
    margin: 1.7rem auto 0.9rem auto;
    padding: 1rem;
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.025));
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow:
        0 10px 35px rgba(0,0,0,0.25),
        inset 0 0 0 1px rgba(255,255,255,0.02);
    backdrop-filter: blur(18px);
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
    font-size: 0.9rem;
    line-height: 1.5;
}

.stTextInput {
    margin-bottom: 0.4rem;
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
    background: rgba(13, 18, 34, 0.98) !important;
    border: 1px solid rgba(139, 92, 246, 0.28) !important;
    border-radius: 18px !important;
    box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.06),
                inset 0 1px 2px rgba(0,0,0,0.3) !important;
}

.stTextInput > div > div > div:focus-within {
    border: 1px solid rgba(139, 92, 246, 0.55) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.12),
                inset 0 1px 2px rgba(0,0,0,0.3) !important;
}

.stTextInput input::placeholder {
    color: rgba(255,255,255,0.45) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.45) !important;
    opacity: 1 !important;
}

.stButton {
    display: flex;
    justify-content: center;
    margin-top: 0.3rem;
    margin-bottom: 0.4rem;
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #5b5cf0, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.76rem 1.35rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 10px 22px rgba(99,102,241,0.22) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 26px rgba(99,102,241,0.28) !important;
}

.card {
    background: linear-gradient(180deg, var(--panel-strong), var(--panel));
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.05rem 1.05rem 0.9rem 1.05rem;
    box-shadow: 0 10px 28px rgba(0,0,0,0.22);
    backdrop-filter: blur(14px);
    margin-top: 1rem;
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
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.65rem;
}

.pipeline {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 0.7rem;
}

.pipe-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 0.9rem;
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
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1rem;
    margin-top: 1rem;
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

.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 1rem;
}

.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 0.9rem;
}

.metric-label {
    color: var(--muted);
    font-size: 0.76rem;
}

.metric-value {
    color: var(--text);
    font-size: 1.08rem;
    font-weight: 700;
    margin-top: 0.12rem;
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
    margin-top: 2rem;
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

def build_metrics(plan_text, final_text, sources):
    plan_len = len(plan_text.split()) if plan_text else 0
    report_len = len(final_text.split()) if final_text else 0
    return {
        "Sources": len(sources),
        "Plan Words": plan_len,
        "Report Words": report_len
    }

def build_chart():
    chart_data = pd.DataFrame({
        "Quality Dimension": ["Clarity", "Depth", "Usability", "Presentation"],
        "Score": [9, 9, 8, 9]
    })
    st.bar_chart(chart_data.set_index("Quality Dimension"), use_container_width=True)

def generate_pdf(report_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("ORION AI Research Report", styles["Title"]))
    story.append(Spacer(1, 12))

    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph(line, styles["BodyText"]))
            story.append(Spacer(1, 6))

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
            <div class="status-sub">Planning → Searching → Writing final report</div>
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
            final_text = data.get("final", "")
            research_payload = data.get("research", {})
            sources = extract_sources(research_payload)
            metrics = build_metrics(plan_text, final_text, sources)

            progress.progress(100)

            status_box.markdown("""
            <div class="status-box">
                <div class="status-title">Live Status</div>
                <div class="status-main">✅ Research completed successfully</div>
                <div class="status-sub">ORION finished planning, searching, and writing your report.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Sources</div>
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

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="card">
                    <div class="section-kicker">Planner Output</div>
                    <div class="section-title">🧠 Structured Plan</div>
                </div>
                """, unsafe_allow_html=True)
                st.write(plan_text)

            with col2:
                st.markdown("""
                <div class="card">
                    <div class="section-kicker">Final Output</div>
                    <div class="section-title">📊 Polished Report</div>
                </div>
                """, unsafe_allow_html=True)
                st.write(final_text)

            left, right = st.columns([0.95, 1.05], gap="large")

            with left:
                st.markdown("""
                <div class="card">
                    <div class="section-kicker">Quality View</div>
                    <div class="section-title">📈 Insight Snapshot</div>
                </div>
                """, unsafe_allow_html=True)
                build_chart()

            with right:
                st.markdown("""
                <div class="card">
                    <div class="section-kicker">References</div>
                    <div class="section-title">🔗 Top Sources</div>
                </div>
                """, unsafe_allow_html=True)

                if sources:
                    for src in sources:
                        snippet = src["content"][:220] + "..." if src["content"] else "No snippet available."
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">{src["title"]}</div>
                            <div class="source-url">{src["url"]}</div>
                            <div class="source-snippet">{snippet}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No sources available.")

            pdf_file = generate_pdf(final_text)
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