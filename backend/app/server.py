from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pathlib import Path

from langserve import add_routes
from .chains import build_chain

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Promtior RAG")

# LangServe
chain = build_chain()
add_routes(app, chain, path="/rag")

# Healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}

# Frontend (served at /ui)
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    @app.get("/")
    def root():
        return RedirectResponse(url="/ui")

else:
    @app.get("/")
    def root():
        return {
            "status": "frontend_not_built",
            "hint": "Run `npm install` and `npm run build` inside /frontend",
            "frontend_dist_expected_at": str(FRONTEND_DIST),
        }
