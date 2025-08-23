import os, re, numpy as np, unicodedata
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- strict, exact-phrase synonym expansion ---
from synonyms import expand_with_synonyms

# -------------------------
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
# Vectorization backend: try embeddings; fall back to TF-IDF
# Also allow forcing TF-IDF via env.
# -------------------------
USE_EMBEDDINGS = not FORCE_TFIDF
MODEL = None
try:
    if USE_EMBEDDINGS:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        MODEL = SentenceTransformer(model_name)
        print(f"[CV-Matcher] Loaded embeddings model: {model_name}")
    else:
        raise ImportError("Forced TF-IDF")
except Exception as e:
    print(f"[CV-Matcher] Embeddings unavailable ({e}). Falling back to TF-IDF.")
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
# STRICT language presence rule (European languages)
# -------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()

# English + Spanish aliases for European languages (normalized)
EU_LANG_ALIASES = {
    "english": {"english", "ingles"},
    "spanish": {"spanish", "espanol"},
    "portuguese": {"portuguese", "portugues"},
    "french": {"french", "frances"},
    "german": {"german", "aleman"},
    "italian": {"italian", "italiano"},
    "dutch": {"dutch", "holandes", "neerlandes"},
    "russian": {"russian", "ruso"},
    "polish": {"polish", "polaco"},
    "czech": {"czech", "checo"},
    "slovak": {"slovak", "eslovaco"},
    "ukrainian": {"ukrainian", "ucraniano"},
    "romanian": {"romanian", "rumano"},
    "greek": {"greek", "griego"},
    "hungarian": {"hungarian", "hungaro", "magyar"},
    "swedish": {"swedish", "sueco"},
    "norwegian": {"norwegian", "noruego"},
    "danish": {"danish", "danes"},
    "finnish": {"finnish", "finlandes", "suomi"},
    "turkish": {"turkish", "turco"},
    "bulgarian": {"bulgarian", "bulgaro"},
    "serbian": {"serbian", "serbio"},
    "croatian": {"croatian", "croata"},
    "bosnian": {"bosnian", "bosnio"},
    "slovenian": {"slovenian", "esloveno"},
    "lithuanian": {"lithuanian", "lituano"},
    "latvian": {"latvian", "leton"},
    "estonian": {"estonian", "estonio"},
    "albanian": {"albanian", "albanes"},
    "irish": {"irish", "irlandes", "gaelic"},
}

LANG_TOKEN_TO_CANON = {}
for canon, toks in EU_LANG_ALIASES.items():
    for t in toks:
        LANG_TOKEN_TO_CANON[_norm(t)] = canon

def extract_language_tokens(text: str) -> set:
    """Return set of canonical language names mentioned in the text (normalized substring check)."""
    n = _norm(text)
    hits = set()
    for tok, canon in LANG_TOKEN_TO_CANON.items():
        if tok in n:
            hits.add(canon)
    return hits

def is_language_restricted_field(field_text: str) -> set:
    """If the extra field mentions one or more languages, return the set of those languages; else empty set."""
    return extract_language_tokens(field_text)

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
    jd_weight: float = 1.0           # adjustable JD weight
    extras: List[ExtraField] = []    # up to 5 expected in UI

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

@app.get("/mode")
def mode():
    return {
        "use_embeddings": bool(USE_EMBEDDINGS and MODEL is not None),
        "model": os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2") if MODEL else None,
        "method": "embeddings" if (USE_EMBEDDINGS and MODEL) else "tfidf"
    }

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



    # --- STRICT synonym expansion (exact phrase only) ---


    jd_expanded = expand_with_synonyms(jd) if jd else jd
    jd_weight = float(req.jd_weight or 1.0)

    query_parts = [("JD", jd_expanded, jd_weight)]
    for f in req.extras[:5]:
        t = (f.text or "").strip()
        if t:

            t_expanded = expand_with_synonyms(t)  # exact-phrase expansion only
            query_parts.append((f.name or "Extra Field", t_expanded, float(f.weight or 0.0)))

    # Compute similarities (per-part and combined reference vector)
    try:
        if USE_EMBEDDINGS and MODEL is not None:

            cv_vecs = MODEL.encode(docs, batch_size=16, show_progress_bar=False, normalize_embeddings=True)


            part_vecs = MODEL.encode([p[1] for p in query_parts], normalize_embeddings=True)
            part_sims = _cosine_similarity(part_vecs, cv_vecs)  # (k, n)

            weights_vec = np.array([p[2] for p in query_parts], dtype=np.float32).reshape(-1, 1)
            combined_vec = (part_vecs * weights_vec).sum(axis=0)

            norm = np.linalg.norm(combined_vec) or 1.0
            combined_vec = combined_vec / norm

            sims = _cosine_similarity([combined_vec], cv_vecs).flatten()
            method = "embeddings"
        else:
            X_docs = VEC.fit_transform(docs)
            X_q_parts = VEC.transform([p[1] for p in query_parts])
            part_sims = _cosine_similarity(X_q_parts, X_docs)

            weights_vec = np.array([p[2] for p in query_parts], dtype=np.float32).reshape(-1, 1)
            weighted = (X_q_parts.multiply(weights_vec)).sum(axis=0)
            sims = _cosine_similarity(weighted, X_docs).flatten()
            method = "tfidf"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")

    # -------- Strict language rule on Extras + rule-aware final score --------
    lang_requirements_per_part = []
    for j, (nm, qtext, w) in enumerate(query_parts):
        if j == 0:
            lang_requirements_per_part.append(set())  # JD not restricted; change if you want JD strict too
        else:
            lang_requirements_per_part.append(is_language_restricted_field(qtext))  # set() or set of langs

    weights = np.array([p[2] for p in query_parts], dtype=np.float32)
    w_sum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0

    adj_scores = np.zeros(len(names), dtype=np.float32)

    for i in range(len(names)):
        acc = 0.0
        present_langs = extract_language_tokens(docs[i])
        for j in range(len(query_parts)):
            sim_ji = float(part_sims[j, i])
            req_langs = lang_requirements_per_part[j]
            if req_langs and present_langs.isdisjoint(req_langs):
                sim_ji = 0.0  # hard zero if language not literally present
            acc += weights[j] * sim_ji
        adj_scores[i] = acc / w_sum

    sims = adj_scores
    method = method + "+rules"

    order = np.argsort(-sims)
    top_n = max(1, min(req.top_n or 5, len(names)))
    results: List[MatchResponseItem] = []

    for r in range(top_n):
        idx = int(order[r])
        parts = []
        present_langs = extract_language_tokens(docs[idx])
        for j, (nm, qtext, w) in enumerate(query_parts):
            sim_val = float(part_sims[j, idx])
            req_langs = lang_requirements_per_part[j]
            if req_langs and present_langs.isdisjoint(req_langs):
                sim_val = 0.0
            parts.append(PartScore(name=nm, weight=float(w), similarity=float(round(sim_val, 6))))
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
