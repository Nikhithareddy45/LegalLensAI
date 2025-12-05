# api/app.py
import os
from fastapi import HTTPException
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ai.risk_detector import RiskDetector
from pathlib import Path
import uuid
import sys
import io

# Add project root to path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Import AI modules
from ai.preprocess import preprocess_contracts
from ai.build_index import build_index
from ai.retriever import ContractRetriever
from ai.summarizer import ContractSummarizer
from ai.qa_reader import LegalQASystem
from ai.config import TXT_DIR
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
import re

app = FastAPI(title="Contract Intelligence API")

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUM = None
QA = None
RISK = RiskDetector()
def suggest_from_risks(risks):
    templates = {
        "termination": [
            "What is the termination notice period?",
            "Can termination occur without cause?"
        ],
        "liability": [
            "What liability caps or exclusions apply?"
        ],
        "indemnity": [
            "What indemnity obligations are specified?"
        ],
        "renewal": [
            "Are there automatic renewal terms?",
            "How can renewal be prevented?"
        ],
        "confidentiality": [
            "What confidentiality obligations and exceptions are defined?"
        ],
        "payment terms": [
            "What are payment terms and deadlines?",
            "Are late fees or interest specified?"
        ],
        "jurisdiction": [
            "Which jurisdiction applies?",
            "How are disputes resolved?"
        ],
        "governing law": [
            "What is the governing law?"
        ],
        "notice period": [
            "What are notice periods for key actions?"
        ]
    }
    order = {"High": 3, "Medium": 2, "Low": 1}
    risks_sorted = sorted(risks, key=lambda r: -order.get(r.get("weight","Low"),1))
    qs = []
    seen = set()
    for r in risks_sorted:
        k = r.get("type","")
        for q in templates.get(k, []):
            if q not in seen:
                seen.add(q)
                qs.append(q)
        if len(qs) >= 8:
            break
    if not qs:
        qs = [
            "What are the key obligations and risks?",
            "What are payment terms and termination conditions?"
        ]
    return qs
def ensure_loaded():
    global SUM, QA
    if SUM is None:
        SUM = ContractSummarizer()
    if QA is None:
        QA = LegalQASystem()
    return SUM, QA


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt", ".pdf"]:
        return JSONResponse({"error": "Only .txt or .pdf"}, status_code=400)

    filename = f"user_{uuid.uuid4().hex}.txt"
    save_path = Path(TXT_DIR) / filename

    contents = await file.read()

    if ext == ".pdf":
        text = extract_pdf_text(contents)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        try:
            text = contents.decode("utf-8", errors="ignore")
        except Exception:
            text = contents.decode("latin-1", errors="ignore")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(clean_text(text))

    doc_id = filename.replace(".txt", "")
    global SUM
    if SUM is None:
        SUM = ContractSummarizer()
    summary = SUM.summarize_document(doc_id)

    txt_path = Path(TXT_DIR) / f"{doc_id}.txt"
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    risks = RISK.analyze(raw_text)

    queries = suggest_from_risks(risks)

    return {"ok": True, "doc_id": doc_id, "summary": summary, "risks": risks, "queries": queries}


@app.post("/summarize")
def summarize(doc_id: str = Form(...)):
    SUM, _ = ensure_loaded()
    result = SUM.summarize_document(doc_id)
    return {"summary": result}


@app.post("/qa")
def qa(question: str = Form(...), doc_id: str = Form(None)):
    _, QA = ensure_loaded()
    answers = QA.answer(question, doc_id=doc_id)
    return {"answers": answers}
@app.post("/upload_text")
def upload_text(text: str = Form(...)):
    """
    Accept pasted text from the UI, save as txt, run preprocess+index, and return doc_id.
    """
    # create a deterministic doc id so subsequent calls can reference it
    doc_id = f"user_paste_{uuid.uuid4().hex[:8]}"
    save_path = Path(TXT_DIR) / f"{doc_id}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    global SUM
    if SUM is None:
        SUM = ContractSummarizer()
    summary = SUM.summarize_document(doc_id)

    with open(save_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    risks = RISK.analyze(raw_text)

    queries = suggest_from_risks(risks)

    return {"ok": True, "doc_id": doc_id, "summary": summary, "risks": risks, "queries": queries}

@app.post("/risk")
def risk(doc_id: str = Form(...)):
    """
    Run simple risk detector on the document's raw text and return risk list.
    """
    txt_path = Path(TXT_DIR) / f"{doc_id}.txt"
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    items = RISK.analyze(text)  # returns list of {"type","weight","context"}
    # convert weight to a numeric score if you like; keep as str for UI
    return {"doc_id": doc_id, "risks": items}

@app.post("/auto_queries")
def auto_queries(doc_id: str = Form(None)):
    if doc_id:
        txt_path = Path(TXT_DIR) / f"{doc_id}.txt"
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            risks = RISK.analyze(text)
            return {"doc_id": doc_id, "queries": suggest_from_risks(risks)}
    suggestions = [
        "What are the key obligations and risks?",
        "What are payment terms and termination conditions?"
    ]
    return {"doc_id": doc_id, "queries": suggestions}
def clean_text(t: str) -> str:
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()

def extract_pdf_text(contents: bytes) -> str:
    try:
        # First try pdfminer (better layout and text coverage)
        bio = io.BytesIO(contents)
        text = pdfminer_extract_text(bio)
        if text and len(text.strip()) > 50:
            return clean_text(text)
    except Exception:
        pass
    # Fallback to PyPDF2
    try:
        bio = io.BytesIO(contents)
        reader = PdfReader(bio)
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        return clean_text(text)
    except Exception:
        return ""
