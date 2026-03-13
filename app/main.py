import asyncio
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .search import search_nom, search_prenom
from .mistral import summarize_origin

app = FastAPI(title="Origine des noms et prénoms")

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/methode")
def methode():
    return FileResponse(STATIC / "methode.html")


@app.get("/api/search")
async def search(
    nom: str = Query("", description="Nom de famille"),
    prenom: str = Query("", description="Prénom"),
):
    result = {}

    if nom.strip():
        result["nom"] = search_nom(nom.strip(), k=5)

    if prenom.strip():
        result["prenom"] = search_prenom(prenom.strip(), k=3)

    # Résumé Mistral en parallèle sur le premier résultat de chaque liste
    tasks = {}
    if result.get("nom") and result["nom"][0].get("origin"):
        tasks["nom"] = summarize_origin(result["nom"][0]["origin"], result["nom"][0]["name"])
    if result.get("prenom") and result["prenom"][0].get("origin"):
        tasks["prenom"] = summarize_origin(result["prenom"][0]["origin"], result["prenom"][0]["name"])

    if tasks:
        keys = list(tasks.keys())
        summaries = await asyncio.gather(*[tasks[k] for k in keys])
        for k, summary in zip(keys, summaries):
            result[k][0]["origin"] = summary

    return result
