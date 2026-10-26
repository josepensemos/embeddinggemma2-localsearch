"""Index.load(): each build owns its files, and the old format still opens."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import search
from helpers import photo


def write_index(d: Path, folder: Path, **extra) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.json").write_text(json.dumps({
        "folder": str(folder), "settings": search.settings(),
        "items": [photo(0, "a.jpg")], "rows": [{"item": 0, "kind": "image"}],
        "stats": {"date": "2026-10-26 09:00"}, **extra,
    }, ensure_ascii=False), encoding="utf-8")


def test_load_reads_the_generation_and_the_files_of_that_build(tmp_path, monkeypatch):
    folder = Path("/tmp/folder")
    monkeypatch.setattr(search, "INDEXES", tmp_path)
    d = search.index_dir(folder)
    write_index(d, folder, generation=42, thumbs_dir="thumbs-42", vectors_file="vectors-42.npy")
    np.save(d / "vectors-42.npy", np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    (d / "thumbs-42").mkdir()
    (d / "thumbs-42" / "00000.jpg").write_bytes(b"jpeg")

    index = search.Index()

    assert index.load(folder) is True
    assert index.generation == 42
    assert index.thumbs == d / "thumbs-42"
    assert index.state().vectors.shape == (1, 3)


def test_an_index_without_the_new_fields_still_loads(tmp_path, monkeypatch):
    folder = Path("/tmp/folder")
    monkeypatch.setattr(search, "INDEXES", tmp_path)
    d = search.index_dir(folder)
    write_index(d, folder)  # the format before this change: no generation, no thumbs_dir
    np.save(d / "vectors.npy", np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    (d / "thumbs").mkdir()
    (d / "thumbs" / "00000.jpg").write_bytes(b"jpeg")

    index = search.Index()

    assert index.load(folder) is True
    assert index.thumbs == d / "thumbs"
    assert index.generation == (d / "index.json").stat().st_mtime_ns  # falls back to the mtime


def test_load_says_no_when_the_index_is_missing_or_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "INDEXES", tmp_path)

    assert search.Index().load(Path("/tmp/never-indexed")) is False

    folder = Path("/tmp/folder")
    d = search.index_dir(folder)
    write_index(d, folder, generation=7, thumbs_dir="thumbs-7", vectors_file="vectors-7.npy")

    assert search.Index().load(folder) is False  # index.json points at vectors that are not there


def test_write_atomic_leaves_no_temporary_file_and_replaces_the_target(tmp_path):
    target = tmp_path / "index.json"
    target.write_text("old", encoding="utf-8")

    search.write_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["index.json"]
