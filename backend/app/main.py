"""FastAPI entry point for the local spelling-correction service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .correction.adapters import DemoCorrectionAdapter, UnavailableCorrectionAdapter
from .correction.model_adapter import HierarchicalModelAdapter
from .correction.service import CorrectionService
from .correction.types import CorrectionAdapter
from .schemas import CorrectionRequest, CorrectionResponse, HealthResponse
from .settings import PROJECT_ROOT, Settings


def build_adapter(settings: Settings) -> CorrectionAdapter:
    if settings.model_source == "demo":
        return DemoCorrectionAdapter()
    try:
        return HierarchicalModelAdapter(settings)
    except Exception as error:  # Keep the health endpoint useful when artifacts are absent or invalid.
        source = settings.hf_model_repo if settings.model_source == "huggingface" else str(settings.model_local_dir)
        return UnavailableCorrectionAdapter(source or settings.model_source, str(error))


def create_app(settings: Settings | None = None, adapter: CorrectionAdapter | None = None) -> FastAPI:
    active_settings = settings or Settings.from_env()
    active_adapter = adapter or build_adapter(active_settings)
    service = CorrectionService(active_adapter)
    app = FastAPI(
        title="Vietnamese Spelling Correction API",
        version="1.0.0",
        description="API cục bộ để phát hiện và sửa lỗi chính tả tiếng Việt.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        status = active_adapter.status()
        return HealthResponse(
            status="ready" if status.adapter != "unavailable" else "unavailable",
            adapter=status.adapter,
            source=status.source,
            model_loaded=status.model_loaded,
            device=status.device,
            detail=status.detail,
        )

    @app.post("/api/correct", response_model=CorrectionResponse)
    def correct(request: CorrectionRequest) -> CorrectionResponse:
        try:
            result = service.correct(request.text, request.mode)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return CorrectionResponse.model_validate(result)

    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


app = create_app()
