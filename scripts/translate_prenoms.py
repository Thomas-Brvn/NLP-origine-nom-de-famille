"""
Traduit origins_prenoms.json (EN → FR) via Google Translate.
Génère origins_prenoms_fr.json. Reprend là où il s'était arrêté si interrompu.
"""

import json
import time
from pathlib import Path
from deep_translator import GoogleTranslator

BASE = Path(__file__).parent

with open(BASE / "origins_prenoms.json") as f:
    originals: dict[str, str] = json.load(f)

# Chargement d'un éventuel fichier partiel
out_path = BASE / "origins_prenoms_fr.json"
if out_path.exists():
    with open(out_path) as f:
        translated: dict[str, str] = json.load(f)
else:
    translated = {}

translator = GoogleTranslator(source="en", target="fr")

todo = {k: v for k, v in originals.items() if k not in translated}
total = len(originals)
done = len(translated)

print(f"Total : {total} | Déjà traduits : {done} | Restants : {len(todo)}")

BATCH = 1  # un à la fois pour éviter les coupures
SAVE_EVERY = 50

for i, (key, text) in enumerate(todo.items(), 1):
    try:
        result = translator.translate(text)
        translated[key] = result
    except Exception as e:
        print(f"  Erreur sur {key}: {e} — texte conservé en anglais")
        translated[key] = text

    if i % SAVE_EVERY == 0 or i == len(todo):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)
        pct = (done + i) / total * 100
        print(f"  {done + i}/{total} ({pct:.0f}%) — sauvegardé")

    # Petite pause pour ne pas surcharger l'API
    time.sleep(0.05)

print(f"\nTerminé. Fichier : {out_path}")
