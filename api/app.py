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

app = FastAPI(title="Contract Intelligence API")

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RETR = None
SUM = None
QA = None
RISK = RiskDetector()
def ensure_loaded():
    global RETR, SUM, QA
    if RETR is None:
        RETR = ContractRetriever()
    if SUM is None:
        SUM = ContractSummarizer()
    if QA is None:
        QA = LegalQASystem()
    return RETR, SUM, QA


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt"]:
        return JSONResponse({"error": "Only .txt supported now"}, status_code=400)

    filename = f"user_{uuid.uuid4().hex}.txt"
    save_path = Path(TXT_DIR) / filename

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Reprocess everything
    preprocess_contracts()
    build_index()

    global RETR, SUM, QA
    RETR = SUM = QA = None

    return {"ok": True, "doc_id": filename.replace(".txt", "")}


@app.post("/summarize")
def summarize(doc_id: str = Form(...)):
    _, SUM, _ = ensure_loaded()
    result = SUM.summarize_document(doc_id)
    return {"summary": result}


@app.post("/qa")
def qa(question: str = Form(...)):
    _, _, QA = ensure_loaded()
    answers = QA.answer(question)
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

    # run pipeline
    try:
        preprocess_contracts()
        build_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    # reset loaded singletons so our new doc is included
    global RETR, SUM, QA
    RETR = SUM = QA = None

    return {"ok": True, "doc_id": doc_id}

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
    """
    Return a small list of suggested queries. If doc_id exists, you can optionally
    use the retriever for doc-specific suggestions (we return generic suggestions here).
    """
    suggestions = [
        "What is the termination notice period?",
        "What payment terms are specified?",
        "Is there an indemnity clause? What does it require?",
        "Are there any limitation of liability clauses?",
        "What are renewal / automatic renewal terms?"
    ]
    return {"doc_id": doc_id, "queries": suggestions}