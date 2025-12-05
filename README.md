# Legal Lens

A single-page app for uploading `.txt`/`.pdf` contracts, generating a concise executive summary, detecting risks with weightage, suggesting follow-up questions based on risks, and answering user questions about the document.

## Features
- Upload by text input or `.txt/.pdf` file
- Immediate 7–15 point executive summary
- Risk detection with weightage (`High`, `Medium`, `Low`) and context snippets
- Suggested questions tailored to detected risks
- Ask Anything: short, direct answers based on the whole document
- Tabbed UI: Summary and Risk Detection, plus a sticky Q&A bar at the bottom

## Tech Stack
- Backend: `FastAPI`, `Uvicorn`
- PDF/Text: `pdfminer.six` (primary), `PyPDF2` (fallback)
- Frontend: `React` (Vite), `Axios`
- Styling: plain CSS (`frontend/src/index.css`)
- Languages: Python (backend), JavaScript (frontend)

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend
- Install dependencies: `python -m pip install -r requirements.txt`
- Run server: `python -m uvicorn api.app:app --reload --port 8000`
- Base URL: `http://localhost:8000`

### Frontend
- Install dependencies: `npm install` in `frontend`
- Run dev server: `npm run dev`
- Open the printed URL (e.g., `http://localhost:5175/`)

## How To Use
- Paste contract text in the left panel and click `Submit Pasted Text`, or upload a `.txt/.pdf` and click `Upload File`.
- After upload, the right panel shows:
  - Summary (first tab) as main bullet points
  - Risk Detection (second tab) with weightage and context
  - Suggested Questions based on detected risks
- Type your question in the sticky `Ask Anything` bar and click `Ask` to get a short answer.

## API Endpoints
- `POST /upload_text` — body: `text`; returns `{doc_id, summary, risks, queries}`
- `POST /upload` — multipart `file` (`.txt/.pdf`); returns `{doc_id, summary, risks, queries}`
- `POST /summarize` — form `doc_id`; returns `{summary}`
- `POST /risk` — form `doc_id`; returns `{risks}`
- `POST /auto_queries` — form `doc_id` (optional); returns `{queries}`
- `POST /qa` — form `question`, `doc_id` (optional); returns `{answers}`

## Architecture
- `api/app.py`
  - Handles uploads, PDF extraction, text cleanup, and orchestrates immediate results
  - Provides endpoints for summary, risk, suggestions, and Q&A
- `ai/summarizer.py`
  - Generates 7–15 bullet points by selecting key sentences using contract keywords and light heuristics
- `ai/risk_detector.py`
  - Rule-based risk detection; returns type, weightage, and a context snippet
- `ai/qa_reader.py`
  - Finds the most relevant sentence(s) across the entire document and returns one concise answer
- `frontend/src/App.jsx`
  - Single page with tabs, upload actions, and a sticky Q&A bar
- `frontend/src/index.css`
  - Light, colorful theme and component styles

## Models and Logic
- Existing (original design references):
  - QA: `deepset/roberta-base-squad2` via `transformers.pipeline`
  - Embeddings: `SentenceTransformer` for semantic scoring
- Current implementation (fast and deterministic):
  - PDF extraction: `pdfminer.six` → `PyPDF2` fallback
  - Summary: heuristic sentence selection (keywords like termination, payment, liability, confidentiality, renewal, governing law, jurisdiction)
  - Q&A: sentence-level scoring based on question terms, single concise answer
  - Suggested questions: derived from detected risks via templates prioritized by weightage

## Data Flow
1. Upload `.txt/.pdf` or pasted text to backend
2. Backend extracts/cleans text and saves as `.txt` under `TXT_DIR`
3. Backend immediately computes summary, risks, and risk-based questions and returns them
4. Frontend displays results in tabs; Q&A uses `doc_id` to answer from the saved text

## Frequently Asked Questions
- Why is my PDF text incomplete?
  - Many PDFs are scanned or have complex layouts. The app tries `pdfminer.six` first and falls back to `PyPDF2`. If extraction is poor, consider uploading a text version.
- Can I change the number of summary points?
  - Yes. In `ai/summarizer.py` you can adjust the cap (currently 15) and the minimum (7).
- Are answers guaranteed to be perfect?
  - Answers are concise and based on the most relevant sentences. For legal decisions, verify against the document.

## Troubleshooting
- Frontend cannot reach backend
  - Ensure backend is running on `http://localhost:8000` and CORS is enabled in `api/app.py`.
- Upload succeeds but results are empty
  - Check the uploaded text under `TXT_DIR`; if the text is only headers or boilerplate, consider providing a clearer source.

## Project Structure
```
MAJOR/
├─ api/
│  └─ app.py
├─ ai/
│  ├─ summarizer.py
│  ├─ risk_detector.py
│  ├─ qa_reader.py
│  └─ config.py
├─ frontend/
│  ├─ src/App.jsx
│  ├─ src/index.css
│  └─ package.json
└─ requirements.txt
```

## Notes
- No dark mode; UI uses a bright, pastel light theme.
- Suggested questions depend on risks detected; clicking a suggestion fills the question bar and triggers Q&A.

