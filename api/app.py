# api/app.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
