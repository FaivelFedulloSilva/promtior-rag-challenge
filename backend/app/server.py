from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path

from langserve import add_routes
from dotenv import load_dotenv

from .chains import build_chain

load_dotenv()

app = FastAPI(title="Promtior RAG")

chain = build_chain()
add_routes(app, chain, path="/rag")

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Frontend (Vite build)
# Expected inside container:
# /app/frontend/dist
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # ✅ /app
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
ASSETS_DIR = FRONTEND_DIST / "assets"

if FRONTEND_DIST.exists():
    if ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/ui")

else:
    @app.get("/", include_in_schema=False)
    def root():
        return {
            "status": "frontend_not_built",
            "hint": "Run `npm install` and `npm run build` inside /frontend",
            "frontend_dist_expected_at": str(FRONTEND_DIST),
        }
