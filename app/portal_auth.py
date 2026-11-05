"""Login presentation and current-session logout on top of Gradio 6.30 auth."""

import base64
import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

ROOT = Path(__file__).resolve().parents[1]
LOGIN_TEMPLATE = Path(__file__).with_name("portal_login.html").read_text()
LOGO_DATA = "data:image/jpeg;base64," + base64.b64encode(
    (ROOT / "app/assets/team-logo.jpg").read_bytes()
).decode("ascii")


def cookie_names(app):
    return (f"access-token-{app.cookie_id}", f"access-token-unsecure-{app.cookie_id}")


def current_user(request):
    app = request.app
    secure_name, local_name = cookie_names(app)
    # Match Gradio's own get_current_user cookie precedence exactly.
    token = request.cookies.get(secure_name) or request.cookies.get(local_name)
    return app.tokens.get(token)


def login_response():
    nonce = secrets.token_urlsafe(24)
    html = LOGIN_TEMPLATE.replace("__NONCE__", nonce).replace("__LOGO_DATA__", LOGO_DATA)
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'nonce-"
            + nonce
            + "'; script-src 'nonce-"
            + nonce
            + "'; connect-src 'self'; form-action 'self'; frame-ancestors 'self' https://huggingface.co; base-uri 'none'",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "same-origin",
        },
    )


class PortalSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        app = request.app
        if (
            request.url.path == "/"
            and request.method == "GET"
            and app.auth is not None
            and current_user(request) is None
        ):
            return login_response()
        if request.url.path in ("/session/logout", "/logout"):
            if request.method != "POST":
                return JSONResponse(
                    {"detail": "Use POST to log out"}, status_code=405, headers={"Allow": "POST"}
                )
            # Only revoke this browser's tokens; the preview account may be shared.
            origin = request.headers.get("origin")
            allowed_hosts = {request.headers.get("host", "")}
            space_host = os.environ.get("SPACE_HOST", "")
            if space_host and space_host.endswith(".hf.space") and "/" not in space_host:
                allowed_hosts.add(space_host)
            share_url = getattr(app.get_blocks(), "share_url", None)
            if share_url:
                allowed_hosts.add(urlsplit(share_url).netloc)
            if request.headers.get("sec-fetch-site") == "cross-site" or (
                origin and urlsplit(origin).netloc not in allowed_hosts
            ):
                return JSONResponse({"detail": "Cross-site logout is not allowed"}, status_code=403)
            response = RedirectResponse("/", status_code=303, headers={"Cache-Control": "no-store"})
            for name in cookie_names(app):
                token = request.cookies.get(name)
                if token:
                    app.tokens.pop(token, None)
                response.delete_cookie(name, path="/")
            return response
        response = await call_next(request)
        if request.url.path == "/":
            response.headers["Cache-Control"] = "no-store"
        return response


def portal_app_kwargs():
    return {"middleware": [Middleware(PortalSessionMiddleware)]}
