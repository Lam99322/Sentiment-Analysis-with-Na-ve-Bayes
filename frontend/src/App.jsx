import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [classifyResult, setClassifyResult] = useState(null);

  const [file, setFile] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);

  const [url, setUrl] = useState("");
  const [urlResult, setUrlResult] = useState(null);
  const [history, setHistory] = useState([]);

  const API_BASE = "http://localhost:5000";

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API_BASE}/history`);
      setHistory(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleClassify = async () => {
    if (!text.trim()) return;
    const res = await axios.post(`${API_BASE}/classify`, { text });
    setClassifyResult(res.data);
  };

  const handleUploadAndTrain = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    const res = await axios.post(`${API_BASE}/upload`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    setTrainingResult(res.data);
  };

  const handleClassifyUrl = async () => {
    if (!url.trim()) return;
    const res = await axios.post(`${API_BASE}/classify-url`, { url });
    setUrlResult(res.data);
    fetchHistory();
  };

  // ===== GAUGE COMPONENT =====
  const Gauge = ({ positive, negative }) => {
    const total = positive + negative;
    const percent = total > 0 ? Math.round((positive / total) * 100) : 0;
    const radius = 80;
    const circ = 2 * Math.PI * radius;
    const offset = circ - (circ * percent) / 100;

    const getColor = () => {
      if (percent >= 60) return "#4CAF50";
      if (percent >= 40) return "#FFC107";
      return "#F44336";
    };

    return (
      <div style={{ textAlign: "center", marginTop: 20 }}>
        <svg width="220" height="140" viewBox="0 0 220 140">
          <path
            d="M20 120 A100 100 0 0 1 200 120"
            fill="none"
            stroke="#ddd"
            strokeWidth="18"
          />
          <path
            d="M20 120 A100 100 0 0 1 200 120"
            fill="none"
            stroke={getColor()}
            strokeWidth="18"
            strokeDasharray={circ}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 1s" }}
          />
          <text x="110" y="105" textAnchor="middle" fontSize="24" fontWeight="600">
            {percent}%
          </text>
        </svg>
        <p style={{ fontSize: 18, marginTop: 10 }}>
          😊 Positive: <b>{positive}</b> | 😡 Negative: <b>{negative}</b>
        </p>
      </div>
    );
  };

  // ===== STYLES =====
  const container = {
    maxWidth: 800,
    margin: "0 auto",
    padding: "20px",
    fontFamily: "Poppins",
  };

  const section = {
    background: "#f7f7f7",
    padding: "20px",
    borderRadius: "10px",
    marginBottom: "30px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
  };

  const inputStyle = {
    width: "100%",
    padding: "10px",
    fontSize: "16px",
    borderRadius: "6px",
    border: "1px solid #ccc",
    marginTop: "5px",
  };

  const buttonStyle = {
    padding: "10px 20px",
    marginTop: "10px",
    fontSize: "16px",
    border: "none",
    borderRadius: "6px",
    backgroundColor: "#4CAF50",
    color: "#fff",
    cursor: "pointer",
  };

  return (
    <div style={container}>
      <h1 style={{ textAlign: "center", marginBottom: "40px" }}>Sentiment Analysis with Naïve Bayes</h1>

      {/* Text Analysis */}
      <div style={section}>
        <h2>Analyze Text</h2>
        <textarea
          rows="4"
          style={inputStyle}
          placeholder="Type your text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button style={buttonStyle} onClick={handleClassify}>Classify</button>
        {classifyResult && (
          <div style={{ marginTop: "15px", background: "#fff", padding: "10px", borderRadius: "6px" }}>
            <pre>{JSON.stringify(classifyResult, null, 2)}</pre>
          </div>
        )}
      </div>

      {/* Upload & Train */}
      <div style={section}>
        <h2>Upload Dataset</h2>
        <input type="file" style={inputStyle} onChange={(e) => setFile(e.target.files[0])} />
        <button style={buttonStyle} onClick={handleUploadAndTrain}>Upload & Train</button>
        {trainingResult && (
          <div style={{ marginTop: "15px", background: "#fff", padding: "10px", borderRadius: "6px" }}>
            <pre>{JSON.stringify(trainingResult, null, 2)}</pre>
            {trainingResult.positive !== undefined && (
              <Gauge positive={trainingResult.positive} negative={trainingResult.negative} />
            )}
          </div>
        )}
      </div>

      {/* URL Analysis */}
      <div style={section}>
        <h2>Analyze URL</h2>
        <input type="text" style={inputStyle} placeholder="Enter URL..." value={url} onChange={(e) => setUrl(e.target.value)} />
        <button style={buttonStyle} onClick={handleClassifyUrl}>Analyze URL</button>
        {urlResult && (
          <div style={{ marginTop: "15px", background: "#fff", padding: "10px", borderRadius: "6px" }}>
            <p><b>Preview:</b> {urlResult.content_preview}</p>
            <pre>{JSON.stringify(urlResult, null, 2)}</pre>
            <Gauge positive={Math.round(urlResult.score_pos * 100)} negative={Math.round(urlResult.score_neg * 100)} />
          </div>
        )}
      </div>

      {/* History */}
      <div style={section}>
        <h2>History</h2>
        {history.length === 0 ? <p>No history yet</p> : (
          <ul style={{ listStyle: "none", paddingLeft: 0 }}>
            {history.map((h, i) => (
              <li key={i} style={{ marginBottom: "10px" }}>
                <a href={h.url} target="_blank" rel="noopener noreferrer" style={{ color: "#007BFF" }}>{h.url}</a> - {h.result.label} (
                {Math.round(h.result.score_pos * 100)}% positive)
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default App;
