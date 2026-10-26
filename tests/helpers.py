"""Synthetic indexes for the ranking tests: no files, no model, full control of the cosines."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import search


def unit(cosine: float) -> np.ndarray:
    """A unit vector whose dot product with [1, 0, 0] is `cosine`."""
    return np.array([cosine, (1 - cosine**2) ** 0.5, 0.0], dtype="float32")


def make_index(items, rows, vectors, folder: str = "/tmp/folder") -> search.Index:
    """An Index holding the given items and rows, with nothing on disk."""
    index = search.Index()
    root = Path(folder)
    index.install(search.State(
        folder=root, directory=root, items=items, rows=rows,
        folded=[search.fold(row.get("text", "")) for row in rows],
        vectors=np.asarray(vectors, dtype="float32"),
        stats={"date": "2026-10-26 09:00"}, generation=1, thumbs=root / "thumbs",
    ))
    return index


def photo(item: int, name: str) -> dict:
    return {"type": "photo", "path": f"/tmp/folder/{name}", "rel": f"Photos/{name}",
            "name": name, "thumb": f"{item:05d}.jpg"}


def pdf_page(item: int, name: str, page: int = 1) -> dict:
    return {"type": "pdf", "path": f"/tmp/folder/{name}", "rel": f"Documents/{name}",
            "name": name, "page": page, "pages": 1, "thumb": f"{item:05d}.jpg"}
