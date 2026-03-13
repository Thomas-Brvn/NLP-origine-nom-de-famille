"""
Benchmark aléatoire — génération automatique de variantes sur un large échantillon.
Perturbations : fautes de frappe, consonnes doublées, substitutions phonétiques, particules.
"""

import json
import unicodedata
import re
import time
import random
from collections import defaultdict

import Levenshtein
from metaphone import doublemetaphone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from pathlib import Path
BASE = Path(__file__).parent.parent / "data"

with open(BASE / "names.json") as f:
    names_data = json.load(f)

ALL_NAMES = [n["name"] for n in names_data]
print(f"Base : {len(ALL_NAMES)} noms chargés")


# ─── Normalisation ────────────────────────────────────────────────────────────

PARTICULES = {"le", "la", "les", "de", "du", "des", "d", "l"}

def normalize(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[-''\\s]+", " ", name).strip()
    parts = name.split()
    while parts and parts[0] in PARTICULES:
        parts = parts[1:]
    return "".join(parts)


# ─── Index normalisé ─────────────────────────────────────────────────────────

NORM_TO_ORIGINAL: dict[str, str] = {}
for n in ALL_NAMES:
    nk = normalize(n)
    if nk not in NORM_TO_ORIGINAL:
        NORM_TO_ORIGINAL[nk] = n

NORM_NAMES = list(NORM_TO_ORIGINAL.keys())
NORM_SET   = set(NORM_NAMES)

# Clés de longueur minimale 4 pour que les perturbations aient du sens
ELIGIBLE = [k for k in NORM_NAMES if len(k) >= 4]


# ─── Générateurs de perturbations ────────────────────────────────────────────

def _apply_delete(name: str) -> str | None:
    """Supprime un caractère aléatoire (pas le premier)."""
    if len(name) < 4:
        return None
    i = random.randint(1, len(name) - 1)
    return name[:i] + name[i+1:]

def _apply_swap(name: str) -> str | None:
    """Intervertit deux caractères adjacents (milieu du mot, noms ≥ 6 lettres)."""
    if len(name) < 6:
        return None
    # Swap dans la zone centrale uniquement (évite les extrémités)
    i = random.randint(2, len(name) - 3)
    lst = list(name)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return "".join(lst)

def _apply_substitute(name: str) -> str | None:
    """Substitue un caractère par un voisin phonétiquement proche."""
    SUBS = {
        'a': 'e', 'e': 'a', 'i': 'y', 'o': 'u', 'u': 'o',
        'n': 'm', 'm': 'n', 's': 'z', 'z': 's',
        'c': 'k', 'k': 'c', 'b': 'p', 'p': 'b',
        'r': 'l', 'l': 'r', 't': 'd', 'd': 't',
        'f': 'v', 'v': 'f', 'g': 'j', 'j': 'g',
    }
    indices = [i for i, c in enumerate(name) if i > 0 and c in SUBS]
    if not indices:
        return None
    i = random.choice(indices)
    lst = list(name)
    lst[i] = SUBS[lst[i]]
    return "".join(lst)

def _apply_double_consonant(name: str) -> str | None:
    """Double ou dédouble une consonne."""
    CONSONANTS = set("bcdfghjklmnpqrstvwxz")
    # Dédoublement : ll→l, tt→t, etc.
    for pair in ["ll", "tt", "pp", "nn", "rr", "ss", "mm", "cc", "ff", "gg"]:
        if pair in name:
            # remplace la première occurrence
            return name.replace(pair, pair[0], 1)
    # Doublement : choisir une consonne unique
    indices = [i for i, c in enumerate(name) if i > 0 and c in CONSONANTS
               and (i == 0 or name[i-1] != c)]
    if not indices:
        return None
    i = random.choice(indices)
    return name[:i] + name[i] + name[i:]

def _apply_phonetic(name: str) -> str | None:
    """Substitutions phonétiques courantes en français."""
    PHONETIC = [
        ("eau", "o"),
        ("au",  "o"),
        ("ou",  "u"),
        ("ai",  "e"),
        ("ei",  "e"),
        ("ph",  "f"),
        ("th",  "t"),
        ("ch",  "sh"),
        ("qu",  "k"),
        ("gu",  "g"),
        ("x",   "s"),
    ]
    random.shuffle(PHONETIC)
    for old, new in PHONETIC:
        if old in name:
            return name.replace(old, new, 1)
    return None

def _apply_particle(name: str) -> str:
    """Ajoute une particule au nom normalisé. Testé sur la query brute (avant normalize)."""
    particle = random.choice(["le ", "du ", "de ", "la "])
    return particle + name


# Catalogue des perturbations avec leur label
PERTURBATIONS = [
    ("typo_delete",   _apply_delete),
    ("typo_swap",     _apply_swap),
    ("typo_subst",    _apply_substitute),
    ("doublement",    _apply_double_consonant),
    ("phonetique",    _apply_phonetic),
    ("particule",     _apply_particle),
]


# ─── Génération du jeu de test ────────────────────────────────────────────────

def build_test_cases(n_per_type: int = 50) -> list[tuple[str, str, str]]:
    """
    Retourne une liste de (query_perturbe, target_normalise, type_perturbation).
    Pour chaque type, on tire n_per_type noms au hasard et on applique la perturbation.
    On rejette si la perturbation produit le même nom ou un nom qui existe dans la base
    (ce serait ambigu — la cible serait incorrecte).
    Exception : les cas 'particule' normalisent toujours vers la cible (test de normalize()).
    """
    cases = []
    for ptype, pfn in PERTURBATIONS:
        pool = ELIGIBLE.copy()
        random.shuffle(pool)
        count = 0
        for norm_name in pool:
            if count >= n_per_type:
                break
            perturbed = pfn(norm_name)
            if perturbed is None:
                continue
            perturbed_norm = normalize(perturbed)
            if ptype == "particule":
                # La normalisation retire la particule → exact match garanti
                # On garde quand même pour valider le pipeline end-to-end
                if perturbed_norm != norm_name:
                    continue  # ne devrait pas arriver
            else:
                # Rejeter si identique après normalisation ou si c'est un autre nom existant
                if perturbed_norm == norm_name:
                    continue
                if perturbed_norm in NORM_SET and perturbed_norm != norm_name:
                    continue  # ambigu
            cases.append((perturbed, norm_name, ptype))
            count += 1
    return cases


TEST_CASES = build_test_cases(n_per_type=50)
print(f"Jeu de test généré : {len(TEST_CASES)} cas "
      f"({len(PERTURBATIONS)} types × ~50 noms aléatoires)\n")


# ─── Méthodes de recherche ────────────────────────────────────────────────────

def search_exact(query: str, k: int = 5):
    q = normalize(query)
    return [NORM_TO_ORIGINAL[q]] if q in NORM_TO_ORIGINAL else []

def search_levenshtein(query: str, k: int = 5):
    q = normalize(query)
    scores = sorted(NORM_NAMES, key=lambda n: Levenshtein.distance(q, n))
    return [NORM_TO_ORIGINAL[n] for n in scores[:k]]

METAPHONE_INDEX: dict[str, list[str]] = defaultdict(list)
for n in NORM_NAMES:
    for key in doublemetaphone(n):
        if key:
            METAPHONE_INDEX[key].append(n)

def search_metaphone(query: str, k: int = 5):
    q = normalize(query)
    candidates = set()
    for key in doublemetaphone(q):
        if key:
            candidates.update(METAPHONE_INDEX.get(key, []))
    if not candidates:
        return []
    scored = sorted(candidates, key=lambda n: Levenshtein.distance(q, n))
    return [NORM_TO_ORIGINAL[n] for n in scored[:k]]

print("Construction index TF-IDF...", end=" ", flush=True)
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

def search_hybrid(query: str, k: int = 5):
    exact = search_exact(query)
    if exact:
        return exact
    phon  = search_metaphone(query, k)
    ngram = search_ngrams(query, k)
    seen: dict[str, int] = {}
    for rank, name in enumerate(phon):
        seen[name] = seen.get(name, 0) + (k - rank)
    for rank, name in enumerate(ngram):
        seen[name] = seen.get(name, 0) + (k - rank)
    return sorted(seen, key=seen.__getitem__, reverse=True)[:k]


# ─── Évaluation ──────────────────────────────────────────────────────────────

METHODS = [
    ("Lookup exact",         search_exact),
    ("Levenshtein",          search_levenshtein),
    ("Double Metaphone",     search_metaphone),
    ("N-grammes TF-IDF",     search_ngrams),
    ("Hybride (Phon+NGram)", search_hybrid),
]

def evaluate_all():
    # (method_name → {ptype → [hit1, hit3, hit5]})
    stats: dict[str, dict[str, list]] = {
        mname: {ptype: [0, 0, 0] for ptype, _ in PERTURBATIONS}
        for mname, _ in METHODS
    }
    totals: dict[str, list[float]] = {mname: [] for mname, _ in METHODS}
    method_times: dict[str, list[float]] = {mname: [] for mname, _ in METHODS}

    # Pré-calcul des résultats pour chaque méthode (évite de répéter)
    all_results: dict[str, list] = {}
    for mname, mfn in METHODS:
        res = []
        for query, target, ptype in TEST_CASES:
            t0 = time.time()
            candidates = mfn(query, k=5)
            elapsed = time.time() - t0
            method_times[mname].append(elapsed)
            norm_cands = [normalize(c) for c in candidates]
            h1 = int(bool(norm_cands) and norm_cands[0] == target)
            h3 = int(target in norm_cands[:3])
            h5 = int(target in norm_cands[:5])
            res.append((h1, h3, h5, ptype))
        all_results[mname] = res

    # Affichage global
    print("\n" + "="*65)
    print("  BENCHMARK ALÉATOIRE — Résultats globaux")
    print("="*65)
    print(f"  {'Méthode':<26} {'P@1':>6} {'P@3':>6} {'P@5':>6} {'ms/req':>8}")
    print(f"  {'-'*26} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    n = len(TEST_CASES)
    for mname, _ in METHODS:
        res = all_results[mname]
        h1 = sum(r[0] for r in res)
        h3 = sum(r[1] for r in res)
        h5 = sum(r[2] for r in res)
        ms = np.mean(method_times[mname]) * 1000
        print(f"  {mname:<26} {h1/n*100:>5.1f}% {h3/n*100:>5.1f}% {h5/n*100:>5.1f}% {ms:>7.1f}ms")

    # Affichage par type de perturbation
    print("\n" + "="*65)
    print("  RÉSULTATS PAR TYPE DE PERTURBATION (P@5)")
    print("="*65)
    ptype_counts = {pt: sum(1 for _, _, p in TEST_CASES if p == pt) for pt, _ in PERTURBATIONS}
    header = f"  {'Type':<18}"
    for mname, _ in METHODS:
        short = mname.split()[0][:8]
        header += f" {short:>8}"
    print(header)
    print("  " + "-"*18 + ("-"*9) * len(METHODS))
    for ptype, _ in PERTURBATIONS:
        row = f"  {ptype:<18}"
        cnt = ptype_counts[ptype]
        if cnt == 0:
            continue
        for mname, _ in METHODS:
            h5 = sum(r[2] for r in all_results[mname] if r[3] == ptype)
            row += f" {h5/cnt*100:>7.1f}%"
        row += f"  (n={cnt})"
        print(row)

    # Affichage des échecs TF-IDF
    tfidf_results = all_results["N-grammes TF-IDF"]
    failures = [
        (TEST_CASES[i][0], TEST_CASES[i][1], TEST_CASES[i][2])
        for i, r in enumerate(tfidf_results) if r[2] == 0
    ]
    print(f"\n  Échecs TF-IDF hors top 5 : {len(failures)}/{n}")
    if failures:
        for q, t, pt in failures[:20]:
            print(f"    [{pt}] '{q}' → attendu '{t}'")
        if len(failures) > 20:
            print(f"    ... et {len(failures)-20} autres")


evaluate_all()
