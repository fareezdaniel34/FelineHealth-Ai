import { useEffect, useRef, useState } from "react";
import "./App.css";

// Flask backend address (change in a .env file with VITE_API_URL if needed)
const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:5000";
const MAX_MB = 10;
const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];

const STATUS = {
  disease: { label: "Possible disease", icon: "!", className: "status-critical" },
  healthy: { label: "No disease detected", icon: "✓", className: "status-good" },
  uncertain: { label: "Unclear result", icon: "?", className: "status-warning" },
};

/* ---------------------------------------------------------------- */
function StatusBadge({ type }) {
  const s = STATUS[type];
  return (
    <span className={`badge ${s.className}`}>
      <span className="badge-icon" aria-hidden="true">{s.icon}</span>
      {s.label}
    </span>
  );
}

function ProbabilityBars({ probabilities, topLabel }) {
  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="probs">
      <h3 className="section-title">Probability for each class</h3>
      <ul className="prob-list">
        {rows.map(([name, p]) => {
          const pct = Math.round(p * 1000) / 10;
          return (
            <li key={name} className="prob-row" title={`${name}: ${pct}%`}>
              <span className="prob-name">{name}</span>
              <span className="prob-track">
                <span
                  className={`prob-fill ${name === topLabel ? "is-top" : ""}`}
                  style={{ width: `${Math.max(pct, 0.5)}%` }}
                />
              </span>
              <span className="prob-value">{pct.toFixed(1)}%</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function ResultCard({ result, onReset }) {
  const [showOverlay, setShowOverlay] = useState(true);
  const { prediction, disease_info: info, images, lesion } = result;
  const hasOverlay = Boolean(images.overlay);
  const imgSrc = hasOverlay && showOverlay ? images.overlay : images.original;

  return (
    <section className="card result" aria-live="polite">
      <div className="result-header">
        <StatusBadge type={result.result_type} />
        <span className="muted small">Analysed in {result.processing_ms} ms</span>
      </div>
      <h2 className="result-title">{result.message}</h2>

      <div className="result-grid">
        <figure className="result-figure">
          <img src={imgSrc} alt={hasOverlay && showOverlay
            ? "Uploaded photo with the detected lesion area highlighted in red"
            : "Uploaded photo"} />
          {hasOverlay && (
            <div className="toggle" role="group" aria-label="Image view">
              <button className={showOverlay ? "active" : ""} onClick={() => setShowOverlay(true)}>
                Lesion highlighted
              </button>
              <button className={!showOverlay ? "active" : ""} onClick={() => setShowOverlay(false)}>
                Original
              </button>
            </div>
          )}
          {lesion.shown && (
            <figcaption className="muted small">
              Red area: region the U-Net marked as a possible lesion
              ({lesion.area_pct}% of the photo). Yellow box: area analysed closely.
            </figcaption>
          )}
        </figure>

        <div className="result-side">
          <div className="confidence">
            <span className="muted small">Most likely</span>
            <span className="confidence-label">{prediction.label}</span>
            <span className="confidence-value">{(prediction.confidence * 100).toFixed(1)}%</span>
          </div>
          <ProbabilityBars probabilities={prediction.probabilities} topLabel={prediction.label} />
        </div>
      </div>

      <div className="info">
        <h3 className="section-title">{info.name}</h3>
        <p>{info.description}</p>
        {result.result_type !== "healthy" && (
          <p><strong>Contagious:</strong> {info.contagious}</p>
        )}
        <p><strong>What to do:</strong> {info.advice}</p>
      </div>

      <button className="btn btn-secondary" onClick={onReset}>Check another photo</button>
    </section>
  );
}

/* ---------------------------------------------------------------- */
export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [serverOk, setServerOk] = useState(null);   // null = checking
  const inputRef = useRef(null);

  // check once that the Flask server and models are ready
  useEffect(() => {
    fetch(`${API}/api/health`)
      .then((r) => r.json())
      .then((d) => setServerOk(Boolean(d.models_loaded)))
      .catch(() => setServerOk(false));
  }, []);

  // free the preview image memory when it changes
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function chooseFile(f) {
    setError("");
    setResult(null);
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) {
      setError("Please choose a JPG, PNG or WEBP photo.");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`The photo is too large. Maximum size is ${MAX_MB} MB.`);
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    chooseFile(e.dataTransfer.files?.[0]);
  }

  async function analyse() {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const body = new FormData();
      body.append("image", file);
      const res = await fetch(`${API}/api/predict`, { method: "POST", body });
      const data = await res.json();
      if (!res.ok || data.status !== "ok") {
        throw new Error(data.message || `Server error (${res.status})`);
      }
      setResult(data);
    } catch (err) {
      setError(err.message === "Failed to fetch"
        ? "Cannot reach the server. Is the Flask backend running?"
        : err.message);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="app">
      <header className="header">
        <h1>
          <span className="logo" aria-hidden="true">🐾</span> FelineHealth-AI
        </h1>
        <p className="muted">
          Automated cat skin disease screening: Ringworm, Flea Allergy and Scabies
        </p>
      </header>

      {serverOk === false && (
        <div className="alert" role="alert">
          The analysis server is offline or the models are not loaded. Start it with{" "}
          <code>python app.py</code> in the backend folder.
        </div>
      )}

      <main>
        {!result && (
          <section className="card">
            <div
              className={`dropzone ${dragging ? "is-dragging" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inputRef.current?.click(); }
              }}
              role="button"
              tabIndex={0}
              aria-label="Upload a cat skin photo: click or drag and drop"
            >
              {preview ? (
                <img className="preview" src={preview} alt="Selected photo preview" />
              ) : (
                <>
                  <div className="drop-icon" aria-hidden="true">⬆</div>
                  <p><strong>Drag and drop</strong> a photo of your cat's skin here</p>
                  <p className="muted small">or click to choose a file (JPG, PNG, WEBP, max {MAX_MB} MB)</p>
                </>
              )}
            </div>
            {/* kept OUTSIDE the drop zone so its click does not bubble back into it */}
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED.join(",")}
              onChange={(e) => chooseFile(e.target.files?.[0])}
              hidden
            />

            {error && <p className="error" role="alert">{error}</p>}

            <div className="actions">
              <button className="btn" onClick={() => inputRef.current?.click()} disabled={loading}>
                Choose File
              </button>
              <button
                className="btn btn-primary"
                onClick={analyse}
                disabled={!file || loading || serverOk === false}
              >
                {loading ? <><span className="spinner" aria-hidden="true" /> Analysing…</> : "Analyse"}
              </button>
              {file && !loading && (
                <button className="btn btn-link" onClick={reset}>Remove</button>
              )}
            </div>

            <div className="tips">
              <h3 className="section-title">Tips for a good photo</h3>
              <ul>
                <li>Take a close-up of the affected skin area.</li>
                <li>Use good lighting and keep the camera steady.</li>
                <li>Part the fur so the skin is visible.</li>
              </ul>
            </div>
          </section>
        )}

        {result && <ResultCard result={result} onReset={reset} />}
      </main>

      <footer className="footer muted small">
        FelineHealth-AI is a preliminary screening tool, not a diagnosis. Please consult a
        veterinarian to confirm any skin condition.
      </footer>
    </div>
  );
}