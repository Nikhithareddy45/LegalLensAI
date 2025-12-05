// frontend/src/App.jsx
import { useState } from "react";
import axios from "axios";
const API = "http://localhost:8000";

export default function App() {
  const [pasteText, setPasteText] = useState("");
  const [file, setFile] = useState(null);
  const [docId, setDocId] = useState("");
  const [summary, setSummary] = useState("");
  const [risks, setRisks] = useState([]);
  const [suggested, setSuggested] = useState([]);
  const [question, setQuestion] = useState("");
  const [answers, setAnswers] = useState([]);
  const [loading, setLoading] = useState(false);

  // 1) Upload / paste flows
  const submitPastedText = async () => {
    if (!pasteText.trim()) return alert("Paste some text first");
    setLoading(true);
    try {
      const form = new URLSearchParams();
      form.append("text", pasteText);
      const res = await axios.post(`${API}/upload_text`, form);
      setDocId(res.data.doc_id);
      await loadAfterIndex(res.data.doc_id);
    } catch (err) {
      alert(
        "Upload text failed: " + (err?.response?.data?.detail || err?.message)
      );
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = async () => {
    if (!file) return alert("Choose a file (.txt only)");
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await axios.post(`${API}/upload`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setDocId(res.data.doc_id);
      await loadAfterIndex(res.data.doc_id);
    } catch (err) {
      alert(
        "File upload failed: " + (err?.response?.data?.detail || err?.message)
      );
    } finally {
      setLoading(false);
    }
  };

  // Common: after indexing, fetch summary, risks, and suggested queries
  const loadAfterIndex = async (id) => {
    setLoading(true);
    try {
      // summary
      const sform = new URLSearchParams();
      sform.append("doc_id", id);
      const sres = await axios.post(`${API}/summarize`, sform);
      setSummary(sres.data.summary);

      // risk
      const rform = new URLSearchParams();
      rform.append("doc_id", id);
      const rres = await axios.post(`${API}/risk`, rform);
      setRisks(rres.data.risks || []);

      // suggestions
      const qform = new URLSearchParams();
      qform.append("doc_id", id);
      const qres = await axios.post(`${API}/auto_queries`, qform);
      setSuggested(qres.data.queries || []);
    } catch (err) {
      console.error(err);
      alert("Post-index fetch failed: " + (err?.message || err));
    } finally {
      setLoading(false);
    }
  };

  // 2) Ask free-form question against the indexed contract
  const askQuestion = async () => {
    if (!question.trim()) return;
    if (!docId) return alert("Upload or paste a document first.");
    setLoading(true);
    try {
      const form = new URLSearchParams();
      form.append("question", question);
      const res = await axios.post(`${API}/qa`, form);
      setAnswers(res.data.answers || []);
    } catch (err) {
      alert("QA failed: " + (err?.message || err?.response?.data?.detail));
    } finally {
      setLoading(false);
    }
  };

  // UI helpers
  const riskColor = (w) => {
    if (!w) return "#ddd";
    if (w.toLowerCase() === "high") return "#dc6161ff";
    if (w.toLowerCase() === "medium") return "#fff0b3";
    return "#e8ffe8";
  };

  return (
    <div
      style={{
        fontFamily: "Inter, Arial",
        width: "100rem",
        margin: "0 auto",
      }}
    >
      <div
        style={{
          width: "90%",
          margin: "0 auto",
        }}
      >
        <h1>Contract Intelligence — One Page</h1>

        <section
          style={{
            display: "grid",
            gap: 12,
            gridTemplateColumns: "1fr 1fr",
            marginTop: 18,
          }}
        >
          {/* Left: Paste / Upload */}
          <div
            style={{ padding: 12, border: "1px solid #eee", borderRadius: 8 }}
          >
            <h3>Paste contract text</h3>
            <textarea
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              rows={10}
              style={{ width: "90%", padding: 8, fontFamily: "monospace" }}
              placeholder="Paste contract text here..."
            />
            <div style={{ marginTop: 8 }}>
              <button onClick={submitPastedText} disabled={loading}>
                Submit Pasted Text
              </button>
            </div>

            <hr style={{ margin: "18px 0" }} />

            <h3>Or upload a .txt file</h3>
            <input
              type="file"
              accept=".txt,.pdf"
              onChange={(e) => setFile(e.target.files[0])}
            />
            <div style={{ marginTop: 8 }}>
              <button onClick={uploadFile} disabled={loading}>
                Upload File
              </button>
            </div>

            <div style={{ marginTop: 12, fontSize: 13, color: "#666" }}>
              <div>
                <strong>Document ID:</strong> {docId || "—"}
              </div>
              <div style={{ marginTop: 6 }}>
                {loading ? "Working… (indexing may take a few seconds)" : ""}
              </div>
            </div>
          </div>

          {/* Right: Summary + Risks */}
          <div
            style={{ padding: 12, border: "1px solid #eee", borderRadius: 8 }}
          >
            <h3>Executive Summary</h3>
            <div
              style={{
                minHeight: 140,
                background: "#fafafa",
                padding: 10,
                borderRadius: 6,
              }}
            >
              <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                {summary || "Summary will appear here after upload."}
              </pre>
            </div>

            <h3 style={{ marginTop: 14 }}>Risk Detection</h3>
            {risks.length === 0 ? (
              <div style={{ color: "#666" }}>
                No risks detected yet. Click "Submit" to analyze.
              </div>
            ) : (
              <div style={{ display: "grid", gap: 8 }}>
                {risks.map((r, i) => (
                  <div
                    key={i}
                    style={{
                      padding: 10,
                      borderRadius: 6,
                      background: riskColor(r.weight),
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                      }}
                    >
                      <strong>{r.type}</strong>
                      <span style={{ fontSize: 13 }}>{r.weight}</span>
                    </div>
                    <div style={{ marginTop: 6, color: "#333" }}>
                      {r.context}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <h3 style={{ marginTop: 14 }}>Suggested Questions</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {suggested.length === 0 ? (
                <div style={{ color: "#666" }}>No suggestions yet.</div>
              ) : (
                suggested.map((q, i) => (
                  <button
                    key={i}
                    style={{ textAlign: "left" }}
                    onClick={() => {
                      setQuestion(q);
                      askQuestion();
                    }}
                  >
                    {q}
                  </button>
                ))
              )}
            </div>
          </div>
        </section>

        {/* Bottom: QA */}
        <section
          style={{
            marginTop: 18,
            padding: 12,
            border: "1px solid #eee",
            borderRadius: 8,
          }}
        >
          <h3>Ask Anything</h3>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              style={{ flex: 1, padding: 10 }}
              placeholder="Type your question about the document..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
            <button onClick={askQuestion} disabled={!question || loading}>
              Ask
            </button>
          </div>

          <div style={{ marginTop: 14 }}>
            {answers.length === 0 ? (
              <div style={{ color: "#666" }}>No answers yet.</div>
            ) : (
              answers.map((a, i) => (
                <div
                  key={i}
                  style={{
                    padding: 10,
                    border: "1px solid #ddd",
                    borderRadius: 6,
                    marginBottom: 8,
                  }}
                >
                  <div>
                    <strong>Answer:</strong> {a.answer}
                  </div>
                  <div style={{ fontSize: 13, color: "#666" }}>
                    Score: {a.score}
                  </div>
                  <div
                    style={{ marginTop: 8, background: "#fafafa", padding: 8 }}
                  >
                    {a.context}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
