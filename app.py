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
@@ -35,7 +35,7 @@
)

# -------------------------
# Vectorization backend: try embeddings; fall back to TF-IDF
# Also allow forcing TF-IDF via env.
# -------------------------
USE_EMBEDDINGS = not FORCE_TFIDF
@@ -55,6 +55,7 @@
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1)

# -------------------------
# Auth
# -------------------------
@@ -79,6 +80,66 @@ def _read_txts(folder: str) -> dict:
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
@@ -92,8 +153,8 @@ class MatchRequest(BaseModel):
    folder_id: str
    top_n: Optional[int] = 5
    include_preview_chars: Optional[int] = 0
    jd_weight: float = 1.0           # adjustable JD weight
    extras: List[ExtraField] = []    # up to 5 expected in UI

class PartScore(BaseModel):
    name: str
@@ -124,6 +185,14 @@ def root():
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
@@ -148,60 +217,83 @@ def match_resumes(req: MatchRequest):
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
@@ -221,14 +313,6 @@ def match_resumes(req: MatchRequest):
        results_count=len(results),
        results=results
    )









# Render/Heroku-style start
if __name__ == "__main__":
