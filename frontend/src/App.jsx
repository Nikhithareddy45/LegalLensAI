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
  const [tab, setTab] = useState("summary");
  const summaryLines = (summary || "").split("\n").filter((l) => l.trim());

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
    if (!file) return alert("Choose a .txt/.pdf file");
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
      form.append("doc_id", docId);
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
    if (!w) return "#e0e0e0";
    if (w.toLowerCase() === "high") return "#ffe5e5";
    if (w.toLowerCase() === "medium") return "#fff9db";
    return "#eaffea";
  };

  return (
    <div
      style={{
        fontFamily: "Poppins, Segoe UI, Helvetica, Arial, sans-serif",
        width: "95vw",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          margin: "0 auto",
          width: "95%",
        }}
      >
        <h1>Legal Lens</h1>
        <h4>AI-Powered Contract Summarization, Risk Detection, and Legal Query Assistant</h4>

        <div style={{ flex: 1, overflowY: "auto" ,  width: "98%", margin: "0 auto"}}>
        <section
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          {/* Left: Paste / Upload */}
          <div
            style={{ padding: 16, border: "1px solid #e0e0e0", borderRadius: 12, background: "#ffffff", boxShadow: "0 4px 12px rgba(0,0,0,0.06)", display:'flex',flexDirection:'row'}}
          >
            <div style={{width:'70%'}}>
              <h3>Paste contract text</h3>
            <textarea
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              rows={10}
              style={{ width: "95%", padding: 10, fontFamily: "monospace", borderRadius: 8, border: "1px solid #b0bec5" }}
              placeholder="Paste contract text here..."
            />
            <div style={{ marginTop: 8 }}>
              <button onClick={submitPastedText} disabled={loading}>
                Submit Pasted Text
              </button>
            </div>

           </div>
          <div style={{width:'auto', display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:5}}>
            
            <h3>Or upload a .txt/.pdf file</h3>
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
          </div>

          {/* Right: Summary + Risks */}
          <div
            style={{ padding: 12, border: "1px solid #e0e0e0", borderRadius: 12, background: "#ffffff" }}
          >
            <div style={{ display: "flex", gap: 8,marginBottom: 11 }}>
              <button
                onClick={() => setTab("summary")}
                style={{
                  padding: "8px 12px",
                  background: tab === "summary" ? "#e3f2fd" : "#fff",
                  borderBottom: tab === "summary" ? "3px solid #42a5f5" : "1px solid #ddd",
                  cursor: "pointer",
                }}
              >
                Executive Summary
              </button>
              <button
                onClick={() => setTab("risks")}
                style={{
                  padding: "8px 12px",
                  borderBottom: tab === "risks" ? "3px solid #ef5350" : "1px solid #ddd",
                  background: tab === "risks" ? "#ffebee" : "#fff",
                  cursor: "pointer",
                }}
              >
                Risk Detection
              </button>
            </div>

            {tab === "summary" ? (
              <div style={{ width: "100%" }}>
                {summaryLines.length === 0 ? (
                  <div style={{ color: "#666" }}>Summary will appear here after upload.</div>
                ) : (
                  <div style={{ display: "grid", gap: 8 }}>
                    {summaryLines.slice(0, 15).map((line, i) => (
                      <div key={i} style={{ width: "100%", padding: 10, background: "#e3f2fd", borderRadius: 8, borderLeft: "4px solid #42a5f5", color: "#0d47a1" }}>
                        - {line.replace(/^\s*-\s*/, "")}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              risks.length === 0 ? (
                <div style={{ color: "#666" }}>No risks detected yet.</div>
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
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <strong>{r.type}</strong>
                        <span style={{ fontSize: 13 }}>{r.weight}</span>
                      </div>
                      <div style={{ marginTop: 6, color: "#3e4a59" }}>{r.context}</div>
                    </div>
                  ))}
                </div>
              )
            )}

            <h3 style={{ marginTop: 14 }}>Suggested Questions</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {suggested.length === 0 ? (
                <div style={{ color: "#666" }}>No suggestions yet.</div>
              ) : (
                suggested.map((q, i) => (
                  <button
                    key={i}
                    style={{ textAlign: "left", background: "#fff8e1", color: "#8d6e63", border: "1px solid #ffe0b2", borderRadius: 20, padding: "8px 12px" }}
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
        </div>

        {/* Bottom: QA */}
        <section
          style={{
            marginTop: 0,
            padding: 16,
            width: "100%",
            background: "#ffffff",
            position: "sticky",
            bottom: 0,
            boxShadow: "0 -6px 16px rgba(0,0,0,0.06)",
            border:"1px solid #2ba3dfff", borderRadius: 8 
          }}
        >
          <h3>Ask Anything</h3>
          <div style={{ display: "flex", gap: 8 , width:"100%"}}>
            <input
              style={{ flex: 1, padding: 16, borderRadius: 8, border: "1px solid #b0bec5" }}
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
              <div style={{ display: "grid", gap: 8 }}>
                {answers.map((a, i) => (
                  <div key={i} style={{ padding: 10, background: "#e8f5e9", borderRadius: 8, border: "1px solid #c8e6c9", color: "#1b5e20" }}>
                    - {a}
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
