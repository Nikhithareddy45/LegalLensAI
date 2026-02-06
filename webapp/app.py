import io
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.summarization.fusion import FusionSummarizer
from src.query_system.query_engine import QueryEngine
from src.risk_detection.semantic import SemanticRiskDetector
from src.risk_detection.rules import RuleBasedRiskDetector
from src.risk_detection.fusion import RiskFusion

RISKY_KEYWORDS = [
    "liability",
    "indemnif",
    "warrant",
    "disclaim",
    "limit of liability",
    "consequential",
    "incidental",
    "lost profits",
    "terminate for convenience",
    "non-compete",
    "no liability",
    "shall not be liable",
]


@st.cache_resource
def get_summarizer() -> FusionSummarizer:
    return FusionSummarizer()


@st.cache_resource
def get_query_engine() -> QueryEngine:
    return QueryEngine()


@st.cache_resource
def get_risk_fusion() -> RiskFusion:
    semantic = SemanticRiskDetector()
    rules = RuleBasedRiskDetector()
    return RiskFusion(semantic, rules)


def _split_clauses(text: str) -> list[str]:
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    return [p + "." for p in parts]


def _analyze_risks(text: str) -> pd.DataFrame:
    fusion = get_risk_fusion()
    clauses = _split_clauses(text)
    rows = []
    for clause in clauses:
        r = fusion.fuse(clause)
        clause_lower = clause.lower()
        keywords_hit = any(kw in clause_lower for kw in RISKY_KEYWORDS)
        rules_triggered = r["rule_triggered_count"] > 0
        base_sev = int(round(r["overall_semantic_risk"] * 5))
        sev = max(1, base_sev)
        conf = max(0.35, float(r["overall_semantic_risk"]))
        if keywords_hit:
            sev = 4 if ("liability" in clause_lower or "disclaim" in clause_lower) else max(sev, 3)
            conf = max(conf, 0.82)
        elif rules_triggered:
            sev = max(sev, 3)
            conf = max(conf, 0.70)
        rows.append({"clause": clause, "severity": sev, "confidence": conf})
    return pd.DataFrame(rows)

def _style_risk_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def row_color(row):
        sev = row["severity"]
        if sev >= 4:
            bg = "#f8d7da"  # soft red
        elif sev == 3:
            bg = "#fff3cd"  # soft amber
        elif sev == 2:
            bg = "#e2e3e5"  # soft gray
        else:
            bg = "#d4edda"  # soft green
        return [f"background-color: {bg}; color: #000000"] * len(row)
    styler = df.style.apply(row_color, axis=1)
    styler = styler.set_properties(**{"color": "#000000"})
    return styler


def generate_pdf_report(df_risk: pd.DataFrame, summary: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph("LegalLensAI Report", styles["Title"]))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("Contract Summary", styles["Heading2"]))
    elems.append(Paragraph(summary, styles["BodyText"]))
    elems.append(Spacer(1, 12))
    # Summary metrics
    try:
        avg_sev = float(df_risk["severity"].mean())
        elems.append(Paragraph(f"Average severity: {avg_sev:.2f}", styles["BodyText"]))
    except Exception:
        pass
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("Risks by Clause", styles["Heading2"]))
    # Sort by severity desc
    df_sorted = df_risk.sort_values(by=["severity", "confidence"], ascending=[False, False]).reset_index(drop=True)
    data = [["Clause", "Severity", "Confidence"]]
    for _, row in df_sorted.iterrows():
        clause_para = Paragraph(str(row["clause"]), styles["BodyText"])
        data.append([clause_para, int(row["severity"]), f"{row['confidence']:.3f}"])
    table = Table(data, repeatRows=1, colWidths=[320, 80, 80])
    styles_list = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        sev_cell = data[i][1]
        sev_val = int(sev_cell) if isinstance(sev_cell, int) or str(sev_cell).isdigit() else 1
        if sev_val >= 4:
            bg = colors.Color(1, 0.8, 0.8)  # light red
        elif sev_val == 3:
            bg = colors.Color(1, 0.88, 0.7)  # light orange
        elif sev_val == 2:
            bg = colors.Color(1, 0.98, 0.77)  # light yellow
        else:
            bg = colors.Color(0.86, 0.93, 0.78)  # light green
        styles_list.append(("BACKGROUND", (0, i), (-1, i), bg))
    table.setStyle(TableStyle(styles_list))
    elems.append(table)
    elems.append(PageBreak())
    elems.append(Paragraph("Notes", styles["Heading2"]))
    elems.append(Paragraph("This report summarizes detected risks and confidence per clause. Review high severity items first.", styles["BodyText"]))
    doc.build(elems)
    return buf.getvalue()


st.set_page_config(page_title="LegalLensAI", layout="wide")
st.title("LegalLensAI – AI-Powered Contract Summarization, Risk Detection, and Legal Query Assistant")

tab1, tab2, tab3 = st.tabs(["📤 Upload Contract", "🔍 Risk Analysis", "💬 Legal QA"])

with tab1:
    uploaded = st.file_uploader("Upload contract (TXT/PDF)", type=["txt", "pdf"])
    manual_text = st.text_area("Or paste contract text", height=200)
    if uploaded is not None or manual_text:
        if manual_text:
            text = manual_text
        elif uploaded.type == "text/plain":
            text = uploaded.read().decode("utf-8")
        else:
            text = "PDF parsing not implemented yet"
        st.session_state.text = text
        st.success("Contract loaded!")
        with st.spinner("Generating instant summary..."):
            try:
                summarizer = get_summarizer()
                instant_summary = summarizer.summarize(st.session_state.text)
                st.session_state.summary = instant_summary
                st.subheader("Instant Summary")
                st.write(instant_summary)
            except Exception as e:
                st.error(f"Summary generation failed: {e}")
        try:
            clauses = _split_clauses(st.session_state.text)
            st.session_state.clauses = clauses
            engine = get_query_engine()
            st.session_state.clause_embs = engine.retriever._embed(clauses)
        except Exception as e:
            st.warning(f"Precomputing embeddings failed: {e}")

with tab2:
    if st.button("Analyze Risks", type="primary"):
        if "text" not in st.session_state or not st.session_state.text:
            st.error("Please upload a contract first.")
        else:
            with st.spinner("Analyzing risks..."):
                df_risk = _analyze_risks(st.session_state.text)
                summarizer = get_summarizer()
                summary = summarizer.summarize(st.session_state.text)
                st.subheader("Risk Heatmap – 41 Clauses")
                st.dataframe(_style_risk_df(df_risk), width="stretch")
                st.subheader("Contract Summary")
                st.write(summary)
                pdf_bytes = generate_pdf_report(df_risk, summary)
                st.download_button("Download Full Report (PDF)", pdf_bytes, "LegalLensAI_Report.pdf", mime="application/pdf")

with tab3:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])
    prompt = st.chat_input("Ask a question about the contract")
    if prompt:
        if "text" not in st.session_state or not st.session_state.text:
            st.error("Please upload a contract first.")
        else:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.spinner("Retrieving evidence and generating answer..."):
                clauses = st.session_state.get("clauses") or _split_clauses(st.session_state.text)
                # Fast retrieval using precomputed clause embeddings
                try:
                    engine = get_query_engine()
                    query_emb = engine.retriever._embed([prompt])
                    clause_embs = st.session_state.get("clause_embs")
                    if clause_embs is None:
                        clause_embs = engine.retriever._embed(clauses)
                        st.session_state.clause_embs = clause_embs
                    sims = cosine_similarity(query_emb, clause_embs)[0]
                    top_idx = sims.argsort()[-3:][::-1]
                    retrieved = [{"clause": clauses[i], "similarity_score": float(sims[i])} for i in top_idx]
                except Exception:
                    # Fallback to engine retrieval
                    retrieved = engine.process_query(prompt, clauses)
                # Generic question handling: answer with summary
                lower = prompt.lower()
                generic = any(kw in lower for kw in ["what is the contract about", "what is the document related", "overview", "summary"])
                if generic and st.session_state.get("summary"):
                    ans = st.session_state.summary
                    evidence = [{"clause": r["clause"], "score": r["similarity_score"]} for r in retrieved]
                    conf = min(0.95, max(s["similarity_score"] for s in retrieved)) if retrieved else 0.7
                else:
                    ans, evidence, conf = get_query_engine().answer(prompt, clauses, st.session_state.text)
            if not evidence:
                st.session_state.chat.append({"role": "assistant", "content": "No evidence found."})
            else:
                answer_block = f"Answer: {ans}\nConfidence: {conf:.3f}\n\nTop Evidence:\n" + "\n".join(
                    [f"• {e['clause']} (score {e['score']:.3f})" for e in evidence]
                )
                st.session_state.chat.append({"role": "assistant", "content": answer_block})
            st.rerun()
