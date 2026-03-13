"""
Benchmark des méthodes de correspondance de noms de famille.
Compare : Levenshtein, Double Metaphone, N-grammes TF-IDF, Hybride.
"""

import json
import unicodedata
import re
import time
from collections import defaultdict

import Levenshtein
from metaphone import doublemetaphone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─── Chargement des données ──────────────────────────────────────────────────

from pathlib import Path
BASE = Path(__file__).parent.parent / "data"

with open(BASE / "names.json") as f:
    names_data = json.load(f)
with open(BASE / "origins.json") as f:
    origins = json.load(f)

# Liste des noms dans la base (normalisés)
ALL_NAMES = [n["name"] for n in names_data]
print(f"Base : {len(ALL_NAMES)} noms chargés")


# ─── Normalisation ───────────────────────────────────────────────────────────

PARTICULES = {"le", "la", "les", "de", "du", "des", "d", "l"}

def normalize(name: str) -> str:
    """Minuscule, sans accents, sans particules initiales, sans espaces/tirets."""
    name = name.lower().strip()
    # Supprimer les accents
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Retirer tirets et apostrophes
    name = re.sub(r"[-'']", " ", name)
    # Retirer la particule initiale (Le Febre → febre)
    parts = name.split()
    if len(parts) > 1 and parts[0] in PARTICULES:
        parts = parts[1:]
    return "".join(parts)  # coller les mots (lefebre, notlefebvre)


# ─── Jeu de test ────────────────────────────────────────────────────────────
# Format : (nom_saisi, nom_cible_dans_base)
# On teste des variantes orthographiques, phonétiques, et des particules.

TEST_CASES = [
    # Fautes de frappe légères
    ("dupont",       "dupont"),
    ("dupond",       "dupont"),
    ("dupon",        "dupont"),
    ("martin",       "martin"),
    ("martinn",      "martin"),
    # Variantes orthographiques classiques
    ("potier",       "potier"),
    ("pottier",      "potier"),
    ("lefebvre",     "lefebvre"),
    ("lefebre",      "lefebvre"),
    ("le febvre",    "lefebvre"),
    ("bernard",      "bernard"),
    ("bernart",      "bernard"),
    ("thomas",       "thomas"),
    ("tomas",        "thomas"),
    # Phonétique
    ("gauthier",     "gauthier"),
    ("gothier",      "gauthier"),
    ("chevalier",    "chevalier"),
    ("chevalié",     "chevalier"),
    ("charpentier",  "charpentier"),
    ("charpantier",  "charpentier"),
    # Noms avec agglutination (Le X → X, Du X → X)
    ("boyer",        "boyer"),
    ("du boyer",     "boyer"),
    ("girard",       "girard"),
    ("le girard",    "girard"),
    ("moulin",       "moulin"),
    ("du moulin",    "moulin"),
    ("perrin",       "perrin"),
    ("le perrin",    "perrin"),
    # Noms bretons/régionaux
    ("riou",         "riou"),
    ("rioux",        "riou"),
    ("morel",        "morel"),
    ("morelle",      "morel"),
    # Noms courants / transcription
    ("garcia",       "garcia"),
    ("garsia",       "garcia"),
    ("moreau",       "moreau"),
    ("morot",        "moreau"),
    ("picard",       "picard"),
    ("picart",       "picard"),
    ("richard",      "richard"),
    ("richar",       "richard"),
    ("rousseau",     "rousseau"),
    ("rouseau",      "rousseau"),
]

print(f"Jeu de test : {len(TEST_CASES)} cas\n")


# ─── Index normalisé ─────────────────────────────────────────────────────────

NORM_TO_ORIGINAL = {}
for n in ALL_NAMES:
    nk = normalize(n)
    if nk not in NORM_TO_ORIGINAL:
        NORM_TO_ORIGINAL[nk] = n

NORM_NAMES = list(NORM_TO_ORIGINAL.keys())


# ─── Méthode 1 : Lookup exact ────────────────────────────────────────────────

def search_exact(query: str, k: int = 5):
    q = normalize(query)
    if q in NORM_TO_ORIGINAL:
        return [NORM_TO_ORIGINAL[q]]
    return []


# ─── Méthode 2 : Levenshtein ─────────────────────────────────────────────────

def search_levenshtein(query: str, k: int = 5):
    q = normalize(query)
    scores = [(Levenshtein.distance(q, n), n) for n in NORM_NAMES]
    scores.sort()
    return [NORM_TO_ORIGINAL[n] for _, n in scores[:k]]


# ─── Méthode 3 : Double Metaphone ────────────────────────────────────────────

# Pré-calcul des clés phonétiques
METAPHONE_INDEX = defaultdict(list)
for n in NORM_NAMES:
    keys = doublemetaphone(n)
    for k in keys:
        if k:
            METAPHONE_INDEX[k].append(n)

def search_metaphone(query: str, k: int = 5):
    q = normalize(query)
    keys = doublemetaphone(q)
    candidates = set()
    for key in keys:
        if key:
            candidates.update(METAPHONE_INDEX.get(key, []))
    if not candidates:
        return []
    # Si plusieurs candidats, trier par Levenshtein
    scored = sorted(candidates, key=lambda n: Levenshtein.distance(q, n))
    return [NORM_TO_ORIGINAL[n] for n in scored[:k]]


# ─── Méthode 4 : N-grammes TF-IDF ────────────────────────────────────────────

print("Construction de l'index TF-IDF (bigrammes)...", end=" ", flush=True)
t0 = time.time()
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
tfidf_matrix = vectorizer.fit_transform(NORM_NAMES)
print(f"OK ({time.time()-t0:.1f}s)")

def search_ngrams(query: str, k: int = 5):
    q = normalize(query)
    qvec = vectorizer.transform([q])
    sims = cosine_similarity(qvec, tfidf_matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:k]
    return [NORM_TO_ORIGINAL[NORM_NAMES[i]] for i in top_idx if sims[i] > 0]


# ─── Méthode 5 : Hybride ─────────────────────────────────────────────────────

def search_hybrid(query: str, k: int = 5):
    # 1. Exact
    exact = search_exact(query)
    if exact:
        return exact
    # 2. Phonétique
    phon = search_metaphone(query, k)
    # 3. N-grammes
    ngram = search_ngrams(query, k)
    # Fusion : prioriser les candidats présents dans les deux listes
    seen = {}
    for rank, name in enumerate(phon):
        seen[name] = seen.get(name, 0) + (k - rank)
    for rank, name in enumerate(ngram):
        seen[name] = seen.get(name, 0) + (k - rank)
    sorted_candidates = sorted(seen, key=seen.get, reverse=True)
    return sorted_candidates[:k]


# ─── Évaluation ──────────────────────────────────────────────────────────────

def evaluate(method_fn, method_name: str, k: int = 5):
    hit1 = 0   # cible en position 1
    hit3 = 0   # cible dans top 3
    hitk = 0   # cible dans top k
    not_found = []
    times = []

    for query, target in TEST_CASES:
        t0 = time.time()
        results = method_fn(query, k)
        times.append(time.time() - t0)

        norm_target = normalize(target)
        norm_results = [normalize(r) for r in results]

        if norm_results and norm_results[0] == norm_target:
            hit1 += 1
        if norm_target in norm_results[:3]:
            hit3 += 1
        if norm_target in norm_results:
            hitk += 1
        else:
            not_found.append((query, target, results[:3]))

    n = len(TEST_CASES)
    print(f"\n{'='*55}")
    print(f"  {method_name}")
    print(f"{'='*55}")
    print(f"  Précision@1  : {hit1}/{n} = {hit1/n*100:.1f}%")
    print(f"  Précision@3  : {hit3}/{n} = {hit3/n*100:.1f}%")
    print(f"  Précision@{k}  : {hitk}/{n} = {hitk/n*100:.1f}%")
    print(f"  Temps moyen  : {np.mean(times)*1000:.1f} ms/requête")
    if not_found:
        print(f"\n  Echecs ({len(not_found)}) :")
        for q, t, r in not_found:
            print(f"    '{q}' → attendu '{t}', obtenu {r}")
    return {"p1": hit1/n, "p3": hit3/n, f"p{k}": hitk/n, "ms": np.mean(times)*1000}


# ─── Lancement ───────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  BENCHMARK — Correspondance de noms de famille")
print("="*55)

results = {}
results["Exact"]          = evaluate(search_exact,       "Lookup exact",        k=5)
results["Levenshtein"]    = evaluate(search_levenshtein, "Levenshtein",         k=5)
results["Double Metaphone"] = evaluate(search_metaphone, "Double Metaphone",    k=5)
results["N-grammes TF-IDF"] = evaluate(search_ngrams,   "N-grammes TF-IDF",    k=5)
results["Hybride"]        = evaluate(search_hybrid,      "Hybride (Phon+NGram)",k=5)

# ─── Récap tableau ───────────────────────────────────────────────────────────

print("\n\n" + "="*55)
print("  RECAPITULATIF")
print("="*55)
print(f"  {'Méthode':<25} {'P@1':>6} {'P@3':>6} {'P@5':>6} {'ms':>8}")
print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
for name, r in results.items():
    print(f"  {name:<25} {r['p1']*100:>5.1f}% {r['p3']*100:>5.1f}% {r['p5']*100:>5.1f}% {r['ms']:>7.1f}ms")
