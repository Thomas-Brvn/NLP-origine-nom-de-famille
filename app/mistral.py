"""
Résumé d'origine via l'API Mistral.
Appelé uniquement sur le premier résultat de chaque recherche.
"""

import os
import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

_API_KEY = os.getenv("MISTRAL_API_KEY", "")
_URL = "https://api.mistral.ai/v1/chat/completions"
_MODEL = "mistral-small-latest"

_SYSTEM = (
    "Tu es un expert en étymologie et onomastique française. "
    "On te fournit un ou plusieurs textes bruts expliquant l'origine d'un prénom ou d'un nom de famille. "
    "Rédige un seul paragraphe synthétique, fluide et informatif en français, en 3 à 5 phrases. "
    "Fusionne les informations complémentaires, supprime les répétitions. "
    "Ne commence pas par le nom lui-même. "
    "Réponds uniquement avec le texte synthétisé, sans titre ni introduction."
)


async def summarize_origin(origin_text: str, name: str) -> str:
    """Retourne un résumé Mistral du texte d'origine, ou le texte brut si échec."""
    if not _API_KEY or not origin_text.strip():
        return origin_text

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                _URL,
                headers={
                    "Authorization": f"Bearer {_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _MODEL,
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": f"Nom : {name}\n\nSource :\n{origin_text}"},
                    ],
                    "max_tokens": 350,
                    "temperature": 0.3,
                },
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    return origin_text
