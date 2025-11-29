import React, { useEffect, useMemo, useRef, useState } from "react";

const searchModes = [
  { key: "text", label: "Text Search", icon: "bx bx-text" },
  { key: "text2image", label: "Text to Image", icon: "bx bx-image" },
  { key: "image2image", label: "Image to Image", icon: "bx bx-photo-album" },
  { key: "face", label: "Face Search", icon: "bx bx-face" },
  { key: "object", label: "Object Search", icon: "bx bx-cube" },
];

const typeIcon = {
  pdf: "bxs-file-pdf",
  csv: "bxs-file-doc",
  png: "bxs-image",
  jpg: "bxs-image",
  jpeg: "bxs-image",
  docx: "bxs-file-doc",
  doc: "bxs-file-doc",
  other: "bxs-file-blank",
};

const formatSize = (bytes = 0) => {
  if (!bytes || bytes <= 0) return "-";
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), sizes.length - 1);
  return `${(bytes / 1024 ** i).toFixed(i === 0 ? 0 : 1)}${sizes[i]}`;
};

export default function App() {
  const apiBase = useMemo(() => import.meta.env.VITE_API_BASE || "", []);
  const [files, setFiles] = useState([]);
  const [filesLoading, setFilesLoading] = useState(false);
  const [activeMode, setActiveMode] = useState("text");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [answer, setAnswer] = useState("");
  const [status, setStatus] = useState("");
  const [topK, setTopK] = useState(5);
  const [rerank, setRerank] = useState(true);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [faceFile, setFaceFile] = useState(null);
  const [folderPath, setFolderPath] = useState("");
  const addFileInputRef = useRef(null);
  const imageInputRef = useRef(null);
  const faceInputRef = useRef(null);

  useEffect(() => {
    const load = async () => {
      setFilesLoading(true);
      try {
        const res = await fetch(`${apiBase}/api/files`);
        const data = await res.json();
        setFiles(data.files || []);
      } catch (err) {
        setStatus("Could not load files");
      } finally {
        setFilesLoading(false);
      }
    };
    load();
  }, [apiBase]);

  const runJson = async (path, body) => {
    const res = await fetch(`${apiBase}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  };

  const onSearch = async () => {
    setStatus("");
    setAnswer("");
    if (activeMode === "text" && !query.trim()) return;
    if (activeMode === "text2image" && !query.trim()) {
      setStatus("Please describe the image you want.");
      return;
    }
    if (activeMode === "object" && !query.trim()) {
      setStatus("Enter an object to search.");
      return;
    }
    if (activeMode === "image2image" && !imageFile) {
      setStatus("Select an image to search with.");
      return;
    }
    if (activeMode === "face" && !faceFile) {
      setStatus("Upload a face image to search.");
      return;
    }

    setLoading(true);
    try {
      if (activeMode === "text") {
        const data = await runJson("/api/search/text", { query, top_k: topK, rerank });
        setResults(data.results || []);
        setAnswer(data.answer || "");
        setPreview((data.results || [])[0] || null);
      } else if (activeMode === "text2image") {
        const data = await runJson("/api/search/text-image", { query, top_k: topK });
        setResults(data.results || []);
        setAnswer(data.answer || "");
        setPreview((data.results || [])[0] || null);
      } else if (activeMode === "object") {
        const data = await runJson("/api/search/object", { query, top_k: topK });
        setResults(data.results || []);
        setAnswer(data.answer || "");
        setPreview((data.results || [])[0] || null);
      } else if (activeMode === "image2image") {
        const fd = new FormData();
        fd.append("file", imageFile);
        fd.append("top_k", topK.toString());
        const res = await fetch(`${apiBase}/api/search/image`, { method: "POST", body: fd });
        const data = await res.json();
        setResults(data.results || []);
        setAnswer(data.answer || "");
        setPreview((data.results || [])[0] || null);
      } else if (activeMode === "face") {
        const fd = new FormData();
        fd.append("file", faceFile);
        fd.append("top_k", topK.toString());
        const res = await fetch(`${apiBase}/api/search/face`, { method: "POST", body: fd });
        const data = await res.json();
        setResults(data.results || []);
        setAnswer(data.answer || "");
        setPreview((data.results || [])[0] || null);
      }
    } catch (err) {
      setStatus("Search failed");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const onReindex = async () => {
    setStatus("Reindexing...");
    try {
      const res = await fetch(`${apiBase}/api/reindex`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: folderPath || null }),
      });
      const data = await res.json();
      setStatus(data.status || "Reindex started");
    } catch (err) {
      setStatus("Reindex failed");
    }
  };

  const onDeleteFile = async (path) => {
    try {
      await fetch(`${apiBase}/api/files/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, delete_file: true }),
      });
      // reload files
      const res = await fetch(`${apiBase}/api/files`);
      const data = await res.json();
      setFiles(data.files || []);
    } catch (err) {
      setStatus("Delete failed");
    }
  };

  const onAddFile = async (file) => {
    if (!file) return;
    setStatus("Adding file...");
    try {
      const fd = new FormData();
      fd.append("file", file);
      await fetch(`${apiBase}/api/files/add`, {
        method: "POST",
        body: fd,
      });
      const res = await fetch(`${apiBase}/api/files`);
      const data = await res.json();
      setFiles(data.files || []);
      setStatus("");
    } catch (err) {
      setStatus("Add file failed");
    }
  };

  return (
    <div className="app-shell">
      <div className="bg-glow blue"></div>
      <div className="bg-glow purple"></div>

      <div className="window">
        <header className="topbar">
          <div className="crumb">AI Powered File Explorer</div>
          <div className="top-actions">
            <div className="folder-input">
              <i className="bx bx-folder-open" />
              <input
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                placeholder="Enter folder path to index"
              />
              <button className="ghost-btn" onClick={onReindex}>
                <i className="bx bx-upload" /> Index Folder
              </button>
            </div>
            <button className="ghost-btn" onClick={() => addFileInputRef.current?.click()}>
              <i className="bx bx-plus" /> Add File
            </button>
            <input
              ref={addFileInputRef}
              type="file"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                onAddFile(f);
              }}
            />
          </div>
        </header>

        <div className="layout">
          <aside className="sidebar">
            <div className="sidebar-title">EXPLORER</div>
            {searchModes.map((mode) => (
              <button
                key={mode.key}
                className={`nav-item ${activeMode === mode.key ? "active" : ""}`}
                onClick={() => {
                  setActiveMode(mode.key);
                  setResults([]);
                  setPreview(null);
                  setImageFile(null);
                  setFaceFile(null);
                  setAnswer("");
                }}
              >
                <i className={`${mode.icon}`} />
                {mode.label}
              </button>
            ))}
          </aside>

          <main className="main-panel">
            <div className="panel glass">
              <div className="panel-header">
                <div>
                  <div className="panel-title">File Explorer</div>
                  <div className="panel-sub">Curated by AI indexer</div>
                </div>
                <div className="panel-sub">70%</div>
              </div>
              <div className="table">
                <div className="table-head">
                  <div>Name</div>
                  <div>Type</div>
                  <div>Size</div>
                  <div></div>
                </div>
                <div className="table-body">
                  {files.map((file) => (
                    <div
                      key={file.name + file.path}
                      className="table-row"
                      onClick={() => setPreview(file)}
                    >
                      <div className="row-name">
                        <i className={`bx ${typeIcon[file.type] || typeIcon.other}`} />
                        <span>{file.name}</span>
                      </div>
                      <div className="row-type">{file.type}</div>
                      <div className="row-size">{formatSize(file.size)}</div>
                      <div className="row-actions">
                        <button
                          className="ghost-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteFile(file.path);
                          }}
                        >
                          <i className="bx bx-trash" />
                        </button>
                      </div>
                    </div>
                  ))}
                  {!files.length && !filesLoading && (
                    <div className="table-row muted">No files</div>
                  )}
                </div>
              </div>
            </div>
          </main>

          <section className="right-rail">
            <div className="panel glass">
              <div className="panel-header">
                <div className="panel-title">
                  {searchModes.find((m) => m.key === activeMode)?.label || "Search"}
                </div>
                <div className="panel-sub">30%</div>
              </div>
              {activeMode === "text" && (
                <textarea
                  className="input glass-input"
                  placeholder="Search documents..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), onSearch())}
                />
              )}
              {activeMode === "text2image" && (
                <input
                  className="input glass-input"
                  placeholder="Describe the image you want..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onSearch()}
                />
              )}
              {activeMode === "image2image" && (
                <div
                  className="uploader"
                  onClick={() => imageInputRef.current?.click()}
                >
                  {imageFile ? `Selected: ${imageFile.name}` : "Click to choose an image"}
                  <input
                    ref={imageInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => setImageFile(e.target.files?.[0] || null)}
                  />
                </div>
              )}
              {activeMode === "face" && (
                <div
                  className="uploader"
                  onClick={() => faceInputRef.current?.click()}
                >
                  {faceFile ? `Selected: ${faceFile.name}` : "Upload portrait or group photo"}
                  <input
                    ref={faceInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => setFaceFile(e.target.files?.[0] || null)}
                  />
                </div>
              )}
              {activeMode === "object" && (
                <input
                  className="input glass-input"
                  placeholder="Search for objects (Laptop, Car, Phone...)"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onSearch()}
                />
              )}

              <div className="controls">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={rerank}
                    onChange={() => setRerank((v) => !v)}
                  />
                  LLM rerank
                </label>
                <div className="topk">
                  <span>Top-K</span>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                  />
                </div>
                <button className="primary-btn" onClick={onSearch}>
                  <i className="bx bx-search" /> Search
                </button>
              </div>
              {status && <div className="status">{status}</div>}
            </div>

            <div className="panel glass">
              <div className="panel-header">
                <div className="panel-title">Search Results</div>
                <div className="panel-sub">{results.length} items</div>
              </div>
              {answer && activeMode !== "text2image" && (
                <div className="answer-box">
                  <div className="answer-title">LLM Answer</div>
                  <div className="answer-text">{answer}</div>
                </div>
              )}
              {loading && <div className="panel-sub">Searching...</div>}
              {!loading && !results.length && <div className="panel-sub">No results</div>}
              <div className="results">
                {results.map((res, idx) => {
                  const displayPath = res.path || res.file_path || res.image_path || res.image || "";
                  const displayName =
                    res.name ||
                    res.label ||
                    (displayPath ? displayPath.split(/[/\\]/).pop() : `Result ${idx + 1}`);
                  const similarity = res.similarity ?? res.confidence ?? 0;
                  const showImage =
                    (["text2image", "image2image", "face", "object"].includes(activeMode) ||
                      res.type === "image" ||
                      res.image_path) && displayPath;
                  const summary =
                    res.summary || res.content || res.context || res.label || displayPath || "No preview available";
                  const iconKey = res.type && typeIcon[res.type] ? res.type : "other";

                  return (
                    <div
                      key={`${displayName}-${displayPath}-${idx}`}
                      className="result-card"
                      onClick={() => setPreview(res)}
                    >
                      <div className="result-head">
                        <div className="result-title">
                          <i className={`bx ${typeIcon[iconKey] || typeIcon.other}`} />
                          {displayName}
                        </div>
                        <span className="badge">Similarity {Math.round(similarity * 100)}%</span>
                      </div>
                      <div className="result-path">{displayPath || "No path available"}</div>
                      {showImage ? (
                        <div className="result-image">
                          <img
                            src={`${apiBase}/api/file?path=${encodeURIComponent(displayPath)}`}
                            alt={displayName}
                          />
                        </div>
                      ) : (
                        <div className="result-summary">{summary}</div>
                      )}
                      <div className="tags">
                        {(res.tags || []).map((tag) => (
                          <span className="tag" key={tag}>
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
