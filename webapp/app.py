import io
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
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


def _split_clauses(text: str) -> list[str]:
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    return [p + "." for p in parts]


def _analyze_risks(text: str) -> pd.DataFrame:
    semantic = SemanticRiskDetector()
    rules = RuleBasedRiskDetector()
    fusion = RiskFusion(semantic, rules)
    clauses = _split_clauses(text)
    rows = []
    for clause in clauses:
        r = fusion.fuse(clause)
        sev = int(round(r["overall_semantic_risk"] * 5))
        conf = float(r["overall_semantic_risk"])
        rows.append({"clause": clause, "severity": sev, "confidence": conf})
    return pd.DataFrame(rows)


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
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightcyan]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elems.append(table)
    elems.append(PageBreak())
    elems.append(Paragraph("Notes", styles["Heading2"]))
    elems.append(Paragraph("This report summarizes detected risks and confidence per clause. Review high severity items first.", styles["BodyText"]))
    doc.build(elems)
    return buf.getvalue()


st.set_page_config(page_title="LegalLensAI", layout="wide")
st.title("LegalLensAI – Contract Intelligence for Lawyers")

tab1, tab2, tab3 = st.tabs(["📤 Upload Contract", "🔍 Risk & Summary", "💬 Legal QA"])

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

with tab2:
    if st.button("Analyze Risks & Generate Summary", type="primary"):
        if "text" not in st.session_state or not st.session_state.text:
            st.error("Please upload a contract first.")
        else:
            with st.spinner("Running hybrid model..."):
                df_risk = _analyze_risks(st.session_state.text)
                summarizer = FusionSummarizer()
                summary = summarizer.summarize(st.session_state.text)
                st.subheader("Risk Heatmap – 41 Clauses")
                st.dataframe(df_risk.style.background_gradient(subset=["severity"], cmap="RdYlGn_r"))
                st.subheader("Contract Summary")
                st.write(summary)
                pdf_bytes = generate_pdf_report(df_risk, summary)
                st.download_button("Download Full Report (PDF)", pdf_bytes, "LegalLensAI_Report.pdf", mime="application/pdf")

with tab3:
    query = st.text_input("Ask any question about the contract")
    if st.button("Get Answer"):
        if "text" not in st.session_state or not st.session_state.text:
            st.error("Please upload a contract first.")
        else:
            clauses = _split_clauses(st.session_state.text)
            engine = QueryEngine()
            results = engine.process_query(query, clauses)
            if not results:
                st.warning("No evidence found.")
            else:
                top = sorted(results, key=lambda r: r["similarity_score"], reverse=True)[0]
                ans = top["summary"]
                conf = float(top["similarity_score"])
                evidence = [{"clause": r["clause"], "score": r["similarity_score"]} for r in results[:3]]
                st.write("Answer:", ans)
                st.write("Confidence:", f"{conf:.3f}")
                st.json(evidence)
