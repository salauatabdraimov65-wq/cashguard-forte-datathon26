"""Attach built static UI to an existing FastAPI model service, without wrapping it."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Match
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

SITE_CONFIG = {
    "synthetic": False,
    "apiProfile": "cashguard-v1",
    "contextAvailable": False,
}
SITE_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "content-security-policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; object-src 'none'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
}


class SiteHeaders:
    """UI policy plus private API no-store; preserve API auth, content and CSP."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        is_site = path in ("/", "/app-config.json", "/assets") or path.startswith(
            "/assets/"
        )
        is_private_api = path.startswith("/api/v1/")
        if scope["type"] != "http" or not (is_site or is_private_api):
            await self.app(scope, receive, send)
            return

        async def with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                existing = message.get("headers", [])
                replacement = {"cache-control": "no-store"}
                if is_site:
                    replacement.update(SITE_HEADERS)
                elif not any(
                    name.lower() == b"x-content-type-options" for name, _ in existing
                ):
                    replacement["x-content-type-options"] = "nosniff"
                headers = [
                    (name, value)
                    for name, value in existing
                    if name.decode("latin-1").lower() not in replacement
                ]
                headers.extend(
                    (name.encode("ascii"), value.encode("ascii"))
                    for name, value in replacement.items()
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, with_headers)


def install_frontend(app: FastAPI, dist: str | Path) -> None:
    """Add exactly `/`, `/assets`, `/app-config.json`; reject collisions early.

    Call once, during application construction and before its lifespan starts.
    No model imports, replacement app, startup handlers or SPA catch-all routes.
    Only index.html and files under assets are exposed, never adjacent files.
    """
    root = Path(dist).absolute()
    index = root / "index.html"
    assets = root / "assets"
    if not index.is_file() or not assets.is_dir():
        raise ValueError("Frontend dist requires index.html and an assets directory")
    if root.is_symlink() or index.is_symlink() or assets.is_symlink():
        raise ValueError("Frontend dist must not contain symlinks")
    if any(path.is_symlink() for path in assets.rglob("*")):
        raise ValueError("Frontend assets must not contain symlinks")
    root = root.resolve()
    for route in app.routes:
        path = getattr(route, "path", "")
        matches_ui = any(
            route.matches(
                {
                    "type": "http",
                    "method": "GET",
                    "path": probe,
                    "root_path": "",
                }
            )[0]
            == Match.FULL
            for probe in ("/", "/app-config.json", "/assets", "/assets/probe.js")
        )
        if (
            matches_ui
            or path in ("/", "/app-config.json", "/assets")
            or path.startswith("/assets/")
            or (path.startswith("/{") and ":path}" in path)
        ):
            raise ValueError(f"Frontend route collision: {path}")
    if app.middleware_stack is not None:
        raise ValueError("Install frontend before the application starts")

    app.add_middleware(SiteHeaders)

    @app.get("/", include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(root / "index.html", media_type="text/html")

    @app.get("/app-config.json", include_in_schema=False)
    async def frontend_config() -> JSONResponse:
        return JSONResponse(SITE_CONFIG, headers={"Cache-Control": "no-store"})

    app.mount("/assets", StaticFiles(directory=root / "assets"), name="frontend-assets")
