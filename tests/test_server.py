"""The HTTP surface: the three fixes (generation, 413, invalid image) and the local-only guard."""
from __future__ import annotations

import http.client
import io
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import search
from helpers import make_index, pdf_page, photo, unit


class FakeModel:
    """Stands in for Model: no weights, no torch, fixed vectors."""

    status = "ready"
    error = ""
    device = "cpu"

    def ready(self) -> bool:
        return True

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype="float32")

    def embed_images(self, paths, batch: int = 8) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0]], dtype="float32")


def build_index(directory: Path) -> search.Index:
    items = [photo(0, "a.jpg"), photo(1, "b.jpg"), photo(2, "c.jpg"), pdf_page(3, "p.pdf")]
    rows = [{"item": 0, "kind": "image"}, {"item": 1, "kind": "image"}, {"item": 2, "kind": "image"},
            {"item": 3, "kind": "text", "text": "documento sobre facturas del servidor"}]
    index = make_index(items, rows, [unit(c) for c in (0.90, 0.10, 0.20, 0.40)], folder=str(directory))
    (directory / "thumbs").mkdir(parents=True, exist_ok=True)
    (directory / "thumbs" / "00000.jpg").write_bytes(b"jpeg-bytes")
    return index


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "online", lambda: True)  # hermetic: no socket to 1.1.1.1
    server = ThreadingHTTPServer(("127.0.0.1", 0), search.make_handler(FakeModel(), build_index(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def call(server, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=10)
    sent = {"Host": f"127.0.0.1:{port}", **(headers or {})}
    connection.request(method, path, body=body, headers=sent)
    response = connection.getresponse()
    is_json = response.getheader("Content-Type", "").startswith("application/json")
    payload = response.read()
    connection.close()
    return response.status, json.loads(payload) if is_json and payload else payload


def test_search_returns_the_index_generation(served):
    status, payload = call(served, "GET", "/search?q=factura&n=2")

    assert status == 200
    assert isinstance(payload["generation"], int)
    assert [r["thumb"] for r in payload["results"]]


def test_status_exposes_the_generation_for_the_thumbnail_cache(served):
    status, payload = call(served, "GET", "/status")

    assert status == 200
    assert isinstance(payload["index"]["generation"], int)


def test_thumbnail_is_served_with_the_generation_query_ignored(served):
    status, _ = call(served, "GET", "/thumb/0?v=123")
    assert status == 200

    status, _ = call(served, "GET", "/file/0?v=123")
    assert status == 200


def test_an_oversized_body_is_refused_before_it_is_read(served):
    status, payload = call(served, "POST", "/search-image", body=b"x",
                           headers={"Content-Length": str(search.MAX_UPLOAD + 1)})

    assert status == 413
    assert "larger than" in payload["error"]


def test_a_body_that_is_not_an_image_is_a_400(served):
    status, payload = call(served, "POST", "/search-image", body=b"not an image")

    assert status == 400
    assert payload["error"] == "that is not an image"


def test_a_real_image_is_accepted_and_answers_with_results(served):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, "JPEG")

    status, payload = call(served, "POST", "/search-image", body=buffer.getvalue())

    assert status == 200
    assert payload["results"]
    assert payload["generation"] is not None


def test_an_image_over_the_pixel_budget_is_refused(served, monkeypatch):
    monkeypatch.setattr(search, "MAX_PIXELS", 1)  # 8x8 is already too big
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, "JPEG")

    status, payload = call(served, "POST", "/search-image", body=buffer.getvalue())

    assert status == 413
    assert payload["error"] == "image too large to decode"


def test_a_foreign_origin_is_forbidden(served):
    status, payload = call(served, "GET", "/status", headers={"Origin": "https://example.com"})
    assert (status, payload["error"]) == (403, "forbidden")


def test_a_foreign_host_is_forbidden(served):
    status, payload = call(served, "GET", "/status", headers={"Host": "evil.example:80"})
    assert (status, payload["error"]) == (403, "forbidden")


def test_a_body_without_a_numeric_length_is_a_400(served):
    status, payload = call(served, "POST", "/index", body=b"{}", headers={"Content-Length": "abc"})
    assert status == 400
    assert payload["error"] == "invalid Content-Length"


def test_an_image_pil_refuses_as_a_decompression_bomb_is_a_413(served, monkeypatch):
    def boom(*args, **kwargs):
        raise Image.DecompressionBombError("too many pixels")

    monkeypatch.setattr(Image, "open", boom)
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, "JPEG")

    status, payload = call(served, "POST", "/search-image", body=buffer.getvalue())

    assert status == 413
    assert payload["error"] == "image too large to decode"
