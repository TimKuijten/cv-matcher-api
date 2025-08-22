import os, re, numpy as np
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- import the contains-based expander from your synonyms.py ---
from synonyms import expand_with_synonyms

# --------------------------
# CONFIG
# -------------------------
API_KEY = os.getenv("CV_API_KEY", "change-me")  # set in env

# Map short ids to real server-side folders containing .txt CVs
ALLOWED_FOLDERS: Dict[str, str] = {
    # e.g. on Render: /srv/cv-lib mounted persistent disk
    "translated": "/srv/cv-lib/translated",
    "english": "/srv/cv-lib/english",
}

WP_ORIGIN = os.getenv("WP_ORIGIN", "https://your-wordpress-domain.com")

# Allow forcing TF-IDF via env to keep memory low
FORCE_TFIDF = os.getenv("CV_FORCE_TFIDF", "false").lower() in ("1", "true", "yes")

app = FastAPI(title="CV Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[WP_ORIGIN],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------
# Vectorization backend: try embeddings; fall back to TF‑IDF
# Also allow forcing TF-IDF via env.
# -------------------------
USE_EMBEDDINGS = not FORCE_TFIDF
MODEL = None
try:
    if USE_EMBEDDINGS:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
        # Smaller model is lighter on memory; change if you prefer multilingual
        MODEL = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    else:
        raise ImportError("Forced TF-IDF")
except Exception:
    USE_EMBEDDINGS = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1)

# -------------------------
# Auth
# -------------------------
def require_api_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return True

# -------------------------
# IO + math helpers
# -------------------------
def _read_txts(folder: str) -> dict:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    data = {}
    for name in os.listdir(folder):
        if name.lower().endswith(".txt"):
            path = os.path.join(folder, name)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = re.sub(r"\s+", " ", f.read()).strip()
                if txt:
                    data[name] = txt
    return data

# -------------------------
# Schemas
# -------------------------
class ExtraField(BaseModel):
    name: Optional[str] = None
    text: str
    weight: float = 1.0

class MatchRequest(BaseModel):
    job_description: str
    folder_id: str
    top_n: Optional[int] = 5
    include_preview_chars: Optional[int] = 0
    extras: List[ExtraField] = []  # up to 5 expected in UI

class PartScore(BaseModel):
    name: str
    weight: float
    similarity: float

class MatchResponseItem(BaseModel):
    rank: int
    filename: str
    similarity: float
    preview: Optional[str] = None
    parts: List[PartScore]

class MatchResponse(BaseModel):
    method: str
    folder_id: str
    results_count: int
    results: List[MatchResponseItem]

# -------------------------
# Utility endpoints
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "cv-matcher-api", "docs": "/docs"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# -------------------------
# Main endpoints
# -------------------------
@app.get("/folders")
def list_folders():
    """Let the UI discover allowed folders."""
    return [{"id": k, "label": k} for k in ALLOWED_FOLDERS.keys()]

@app.post("/match", response_model=MatchResponse, dependencies=[Depends(require_api_key)])
def match_resumes(req: MatchRequest):
    jd = (req.job_description or "").strip()
    if not jd:
        raise HTTPException(status_code=400, detail="job_description is required.")
    if req.folder_id not in ALLOWED_FOLDERS:
        raise HTTPException(status_code=400, detail="Unknown folder_id.")
    folder = ALLOWED_FOLDERS[req.folder_id]

    resumes = _read_txts(folder)
    if not resumes:
        raise HTTPException(status_code=404, detail="No .txt resumes found in the selected folder.")

    names = list(resumes.keys())
    docs  = list(resumes.values())

    # --- Expand synonyms (CONTAINS-based) for JD and each extra field ---
    # This broadens the query so semantically equivalent phrases match stronger.
    # --- STRICT synonym expansion (exact phrase only) ---
    # With your strict synonyms.py, expansion happens only if the entire phrase matches a key.
    from synonyms import expand_with_synonyms
    jd_expanded = expand_with_synonyms(jd) if jd else jd

    query_parts = [("JD", jd_expanded, 1.0)]
    for f in req.extras[:5]:
        t = (f.text or "").strip()
        if t:
            t_expanded = expand_with_synonyms(t)
            t_expanded = expand_with_synonyms(t)  # exact-phrase expansion only
            query_parts.append((f.name or "Extra Field", t_expanded, float(f.weight or 0.0)))

    # Compute similarities (per-part and combined)
    try:
        if USE_EMBEDDINGS and MODEL is not None:
            # Encode CVs once (for small/medium corpora). If memory is tight, use batching.
            cv_vecs = MODEL.encode(docs, batch_size=16, show_progress_bar=False, normalize_embeddings=True)

            # Encode each query part
            part_vecs = MODEL.encode([p[1] for p in query_parts], normalize_embeddings=True)
            part_sims = _cosine_similarity(part_vecs, cv_vecs)  # (k, n)

            # Weighted combined vector
            weights = np.array([p[2] for p in query_parts], dtype=np.float32).reshape(-1, 1)
            combined_vec = (part_vecs * weights).sum(axis=0)
            norm = np.linalg.norm(combined_vec) or 1.0
            combined_vec = combined_vec / norm

            sims = _cosine_similarity([combined_vec], cv_vecs).flatten()
            method = "embeddings"
        else:
            X_docs = VEC.fit_transform(docs)
            X_q_parts = VEC.transform([p[1] for p in query_parts])
            part_sims = _cosine_similarity(X_q_parts, X_docs)

            weights = np.array([p[2] for p in query_parts], dtype=np.float32).reshape(-1, 1)
            weighted = (X_q_parts.multiply(weights)).sum(axis=0)
            sims = _cosine_similarity(weighted, X_docs).flatten()
            method = "tfidf"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")

    order = np.argsort(-sims)
    top_n = max(1, min(req.top_n or 5, len(names)))
    results: List[MatchResponseItem] = []

    for r in range(top_n):
        idx = int(order[r])
        parts = []
        for j, (nm, _, w) in enumerate(query_parts):
            parts.append(PartScore(name=nm, weight=float(w), similarity=float(round(float(part_sims[j, idx]), 6))))
        preview = None
        if (req.include_preview_chars or 0) > 0:
            preview = docs[idx][: int(req.include_preview_chars)]
        results.append(
            MatchResponseItem(
                rank=r + 1,
                filename=names[idx],
                similarity=float(round(float(sims[idx]), 6)),
                preview=preview,
                parts=parts
            )
        )

    return MatchResponse(
        method=method,
        folder_id=req.folder_id,
        results_count=len(results),
        results=results
    )

# Render/Heroku-style start
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
