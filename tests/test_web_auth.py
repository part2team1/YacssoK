"""Exercise the portal middleware with Gradio's real login and cookie handling."""

import pytest

gr = pytest.importorskip("gradio")

# The default training CI does not install the optional demo dependencies.
from fastapi.testclient import TestClient  # noqa: E402
from gradio.routes import App  # noqa: E402

from app.portal_auth import portal_app_kwargs  # noqa: E402


@pytest.fixture
def portal():
    with gr.Blocks(analytics_enabled=False) as blocks:
        gr.Textbox()
    blocks.auth = [("test-reviewer", "local-test-password")]
    blocks.auth_message = "Portal test"
    application = App.create_app(blocks, app_kwargs=portal_app_kwargs())
    with TestClient(application) as client:
        yield application, client


def login(client):
    response = client.post(
        "/login", data={"username": "test-reviewer", "password": "local-test-password"}
    )
    assert response.status_code == 200


def test_unauthenticated_login_page_does_not_expose_config(portal):
    _, client = portal
    response = client.get("/")
    assert response.status_code == 200
    assert "__NONCE__" not in response.text
    assert "앱 데모" in response.text
    assert "Content-Security-Policy" in response.headers
    assert client.get("/config").status_code == 401


def test_wrong_password_cannot_open_config(portal):
    _, client = portal
    assert (
        client.post("/login", data={"username": "test-reviewer", "password": "wrong"}).status_code
        == 400
    )
    assert client.get("/config").status_code == 401


def test_logout_revokes_only_this_session(portal):
    application, client = portal
    login(client)
    with TestClient(application) as other:
        login(other)
        assert client.get("/config").status_code == 200
        assert client.get("/session/logout").status_code == 405
        assert (
            client.post(
                "/session/logout", headers={"Origin": "https://unrelated.invalid"}
            ).status_code
            == 403
        )
        assert client.get("/config").status_code == 200
        assert client.post("/session/logout", follow_redirects=False).status_code == 303
        assert client.get("/config").status_code == 401
        assert other.get("/config").status_code == 200
