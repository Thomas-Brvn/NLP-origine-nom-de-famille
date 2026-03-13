"""
Moteur de recherche : N-grammes TF-IDF + normalisation + fallback Levenshtein.
Features : extraction de variantes, famille linguistique, liens "voir aussi".
"""

import json
import unicodedata
import re
from pathlib import Path

import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BASE = Path(__file__).parent.parent

# ─── Chargement ──────────────────────────────────────────────────────────────

DATA = BASE / "data"

with open(DATA / "names.json") as f:
    _names_data = json.load(f)
with open(DATA / "origins.json") as f:
    _origins_noms = json.load(f)
with open(DATA / "prenoms.json") as f:
    _prenoms_data = json.load(f)
with open(DATA / "origins_prenoms_fr.json") as f:
    _origins_prenoms = json.load(f)
with open(DATA / "prenoms_insee.json") as f:
    _insee_prenoms: dict[str, int] = json.load(f)
with open(DATA / "noms_insee.json") as f:
    _insee_noms: dict[str, int] = json.load(f)

# Rangs prénoms
_prenoms_sorted = sorted(_insee_prenoms.items(), key=lambda x: x[1], reverse=True)
_prenoms_rank: dict[str, int] = {n: i + 1 for i, (n, _) in enumerate(_prenoms_sorted)}

# Rangs noms (exclusion de la catégorie fourre-tout)
_noms_sorted = [(n, c) for n, c in sorted(_insee_noms.items(), key=lambda x: x[1], reverse=True) if n != "AUTRES NOMS"]
_noms_rank: dict[str, int] = {n: i + 1 for i, (n, _) in enumerate(_noms_sorted)}


def _get_insee_prenom(name: str) -> dict | None:
    key = name.upper()
    count = _insee_prenoms.get(key)
    if count is None:
        return None
    return {"count": count, "rank": _prenoms_rank[key], "total": len(_prenoms_rank)}


def _get_insee_nom(name: str) -> dict | None:
    key = name.upper()
    count = _insee_noms.get(key)
    if count is None:
        return None
    rank = _noms_rank.get(key)
    if rank is None:
        return None
    return {"count": count, "rank": rank, "total": len(_noms_rank)}

_all_names_set = {n["name"] for n in _names_data}

# ─── Normalisation ───────────────────────────────────────────────────────────

PARTICULES = {"le", "la", "les", "de", "du", "des", "d", "l"}

def normalize(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[-''\s]+", " ", name).strip()
    parts = name.split()
    while parts and parts[0] in PARTICULES:
        parts = parts[1:]
    return "".join(parts)

# ─── Extraction de variantes ─────────────────────────────────────────────────

_VARIANT_PATTERNS = [
    r'[Vv]ariantes?\s*[:;]\s*([^.(]+)',
    r'[Ff]ormes? voisines?\s*[:;]\s*([^.(]+)',
    r'[Ff]ormes? proches?\s*[:;]\s*([^.(]+)',
    r'[Aa]utres? formes?\s*[:;]\s*([^.(]+)',
]

def extract_variants(text: str) -> list[str]:
    """Extrait les variantes orthographiques mentionnées dans un texte d'origine."""
    variants = []
    for pat in _VARIANT_PATTERNS:
        for m in re.finditer(pat, text):
            raw = m.group(1)
            # Supprimer codes de départements ex: (25, 38)
            raw = re.sub(r'\(\d[\d,\s]*\)', '', raw)
            # Supprimer parenthèses restantes avec leur contenu
            raw = re.sub(r'\([^)]*\)', '', raw)
            # Séparer sur virgule ou point-virgule
            parts = re.split(r'[,;]', raw)
            for part in parts:
                part = part.strip().strip('.,;)')
                # Garder seulement si ça ressemble à un nom (lettres, tirets, espaces)
                if part and re.match(r'^[A-ZÀ-Ÿa-zà-ÿ][A-ZÀ-Ÿa-zà-ÿ\s-]{1,30}$', part):
                    variants.append(part)
    # Dédoublonner en préservant l'ordre
    seen = set()
    result = []
    for v in variants:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            result.append(v)
    return result[:8]

# ─── Famille linguistique ────────────────────────────────────────────────────

_LANG_RULES: list[tuple[str, str]] = [
    # Ordre important : du plus spécifique au plus général
    (r'\bbreton\b|\ben breton\b|\bkelt\b', "Breton"),
    (r'\bbasque\b|\bvascon\b', "Basque"),
    (r'\boccitan\b|\bprovençal\b|\blangue d.oc\b', "Occitan"),
    (r'\bgaulois\b|\bgallo\b', "Gaulois"),
    (r'\bgermanique\b|\bfrancique\b|\bgoth\b|\ballemand\b|\bvieux haut.allemand\b', "Germanique"),
    (r'\barabe\b|\bharabe\b', "Arabe"),
    (r'\bhébreu\b|\bbibliqu\b|\bantique hébreu\b', "Hébreu"),
    (r'\bgrec\b|\bgrec ancien\b|\blatino.grec\b', "Grec"),
    (r'\blatin\b|\bdu latin\b|\blatine\b', "Latin"),
    (r'\bespagnol\b|\bcastillan\b|\bcatalan\b', "Espagnol"),
    (r'\bitalien\b|\blombard\b|\bsicilien\b', "Italien"),
    (r'\bflamand\b|\bnéerlandais\b', "Flamand/Néerlandais"),
    (r'\balsacien\b|\balémanique\b', "Alsacien"),
    (r'\bslave\b|\bpolonais\b|\btchèque\b', "Slave"),
]

def detect_language_family(text: str) -> str | None:
    """Détecte la famille linguistique d'origine à partir du texte."""
    text_lower = text.lower()
    for pattern, label in _LANG_RULES:
        if re.search(pattern, text_lower):
            return label
    return None

# ─── Liens "voir aussi" ──────────────────────────────────────────────────────

_SEE_ALSO_PATTERN = re.compile(
    r'\bvoir\s+(?!ce nom|les noms?|aussi\b)([A-ZÀ-Ÿ][a-zà-ÿ-]{2,}(?:\s[A-ZÀ-Ÿ][a-zà-ÿ-]+)?)\b'
)

def extract_see_also(text: str) -> list[str]:
    """Extrait les noms mentionnés dans 'voir NomPropre', filtrés sur la base."""
    results = []
    seen = set()
    for m in _SEE_ALSO_PATTERN.finditer(text):
        name = m.group(1).strip()
        name_lower = name.lower()
        if name_lower not in seen and name_lower in _all_names_set:
            seen.add(name_lower)
            results.append(name)
    return results[:5]

# ─── Enrichissement d'un résultat ────────────────────────────────────────────

def _enrich(origin_text: str) -> dict:
    """Ajoute variantes, famille linguistique et liens voir-aussi au texte brut."""
    return {
        "variants": extract_variants(origin_text),
        "language_family": detect_language_family(origin_text),
        "see_also": extract_see_also(origin_text),
    }

# ─── Index noms de famille ───────────────────────────────────────────────────

_norm_to_entry: dict[str, dict] = {}
for entry in _names_data:
    nk = normalize(entry["name"])
    if nk not in _norm_to_entry:
        _norm_to_entry[nk] = entry

_norm_keys = list(_norm_to_entry.keys())

print(f"[search] Indexation TF-IDF noms ({len(_norm_keys)})...", end=" ", flush=True)
_vec_noms = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
_mat_noms = _vec_noms.fit_transform(_norm_keys)
print("OK")

# ─── Index prénoms ───────────────────────────────────────────────────────────

_norm_to_prenom: dict[str, dict] = {}
for entry in _prenoms_data:
    nk = normalize(entry["name"])
    if nk not in _norm_to_prenom:
        _norm_to_prenom[nk] = entry

_prenom_keys = list(_norm_to_prenom.keys())

print(f"[search] Indexation TF-IDF prénoms ({len(_prenom_keys)})...", end=" ", flush=True)
_vec_prenoms = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
_mat_prenoms = _vec_prenoms.fit_transform(_prenom_keys)
print("OK")

# ─── Fonctions de récupération ───────────────────────────────────────────────

def _get_origin_text(origin_ids: list[str]) -> str:
    texts = [_origins_noms[oid] for oid in origin_ids if oid in _origins_noms]
    return "\n\n".join(texts) if texts else ""

# Détecte "Forme X de/d'[Name]", "Variante de/d'[Name]", etc.
_CROSSREF_RE = re.compile(
    r"(?:Forme|Variante|Diminutif|Féminin(?:e)?|Version|Masculin(?:e)?)\b[^d'\n]{0,40}d['']"
    r"([A-ZÀ-Ÿ][a-zA-ZÀ-ÿ-]{1,30})",
    re.IGNORECASE,
)

def _resolve_prenom_origin(origin_ids: list[str], _depth: int = 0) -> str:
    """Retourne le texte d'origine, en résolvant les renvois simples (max 1 niveau)."""
    texts = [_origins_prenoms[oid] for oid in origin_ids if oid in _origins_prenoms]
    text = "\n\n".join(texts) if texts else ""

    # Si le texte est court et ressemble à un renvoi, on suit la référence
    if _depth == 0 and len(text) < 200:
        m = _CROSSREF_RE.search(text)
        if m:
            ref_name = m.group(1)
            ref_norm = normalize(ref_name)
            if ref_norm in _norm_to_prenom:
                ref_entry = _norm_to_prenom[ref_norm]
                ref_text = _resolve_prenom_origin(ref_entry.get("origins", []), _depth=1)
                if ref_text and ref_text != text:
                    # Garder le texte original court + ajouter la définition complète
                    return text + "\n\n" + ref_text
    return text

def _get_prenom_text(origin_ids: list[str]) -> str:
    return _resolve_prenom_origin(origin_ids)

def _tfidf_search(query_norm: str, vectorizer, matrix, keys, k: int = 5) -> list[tuple[str, float]]:
    qvec = vectorizer.transform([query_norm])
    sims = cosine_similarity(qvec, matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:k]
    return [(keys[i], float(sims[i])) for i in top_idx if sims[i] > 0]

# ─── API publique ─────────────────────────────────────────────────────────────

def search_nom(query: str, k: int = 5) -> list[dict]:
    q = normalize(query)

    # 1. Exact
    if q in _norm_to_entry:
        entry = _norm_to_entry[q]
        origin = _get_origin_text(entry.get("origins", []))
        return [{
            "name": entry["name"],
            "score": 1.0,
            "match_type": "exact",
            "origin": origin,
            "insee": _get_insee_nom(entry["name"]),
            **_enrich(origin),
        }]

    # 2. TF-IDF (top-k avec scores cosinus)
    tfidf_candidates = _tfidf_search(q, _vec_noms, _mat_noms, _norm_keys, k)

    # 3. Levenshtein (top-k triés par distance)
    lev_sorted = sorted(_norm_keys, key=lambda n: Levenshtein.distance(q, n))
    lev_candidates = lev_sorted[:k]

    # 4. Fusion par rang : chaque méthode contribue (k - rang) points
    scores: dict[str, float] = {}
    for rank, (name, _) in enumerate(tfidf_candidates):
        scores[name] = scores.get(name, 0.0) + (k - rank)
    for rank, name in enumerate(lev_candidates):
        scores[name] = scores.get(name, 0.0) + (k - rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for norm_name, combined_score in ranked:
        entry = _norm_to_entry[norm_name]
        origin = _get_origin_text(entry.get("origins", []))
        results.append({
            "name": entry["name"],
            "score": round(combined_score / (2 * k), 3),  # normalisé [0, 1]
            "match_type": "approximate",
            "origin": origin,
            "insee": _get_insee_nom(entry["name"]),
            **_enrich(origin),
        })
    return results

def search_prenom(query: str, k: int = 3) -> list[dict]:
    q = normalize(query)

    if q in _norm_to_prenom:
        entry = _norm_to_prenom[q]
        origin = _get_prenom_text(entry.get("origins", []))
        return [{
            "name": entry["name"],
            "score": 1.0,
            "match_type": "exact",
            "origin": origin,
            "language_family": detect_language_family(origin),
            "insee": _get_insee_prenom(entry["name"]),
        }]

    candidates = _tfidf_search(q, _vec_prenoms, _mat_prenoms, _prenom_keys, k)
    results = []
    for norm_name, score in candidates:
        entry = _norm_to_prenom[norm_name]
        origin = _get_prenom_text(entry.get("origins", []))
        results.append({
            "name": entry["name"],
            "score": round(score, 3),
            "match_type": "approximate",
            "origin": origin,
            "language_family": detect_language_family(origin),
            "insee": _get_insee_prenom(entry["name"]),
        })
    return results
