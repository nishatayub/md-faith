from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mdfaith.api import create_app

WEB = Path(__file__).resolve().parents[1] / "src" / "mdfaith" / "web"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(":memory:", WEB))


def test_app_shell_and_static_assets_are_served(client):
    r = client.get("/app")
    assert r.status_code == 200 and "MD-Faith" in r.text and "/app/static/app.js" in r.text
    for f, kind in [("app.js", "javascript"), ("charts.js", "javascript"), ("app.css", "css")]:
        a = client.get(f"/app/static/{f}")
        assert a.status_code == 200 and kind in a.headers["content-type"]


def test_root_serves_something_navigable(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (200, 307)


def test_app_js_has_no_native_replace_children_with_nullable_args():
    """Regression: native replaceChildren() stringifies null; nullable children must go through put()."""
    js = (WEB / "app" / "app.js").read_text()
    assert "const put =" in js and "filter(Boolean)" in js


def test_every_route_in_nav_is_defined_in_router():
    html = (WEB / "app" / "index.html").read_text()
    js = (WEB / "app" / "app.js").read_text()
    for name in ["home", "lab", "verify", "runs", "results", "about"]:
        assert f'data-r="{name}"' in html and f"async function {name}(" in js
