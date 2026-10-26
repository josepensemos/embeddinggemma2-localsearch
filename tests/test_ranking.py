"""Index.rank(): modality scaling, snippets, code boost, filters and the state snapshot."""
from __future__ import annotations

from pathlib import Path

import pytest

import search
from helpers import make_index, pdf_page, photo, unit


def test_a_lower_cosine_wins_when_its_modality_is_tighter(query_vector):
    """The modality gap: text and image similarities are standardised on their own scale."""
    items = [photo(0, "a.jpg"), photo(1, "b.jpg"), photo(2, "c.jpg"), photo(3, "d.jpg"),
             pdf_page(4, "p1.pdf"), pdf_page(5, "p2.pdf"), pdf_page(6, "p3.pdf")]
    rows = [{"item": 0, "kind": "image"}, {"item": 1, "kind": "image"},
            {"item": 2, "kind": "image"}, {"item": 3, "kind": "image"},
            {"item": 4, "kind": "text", "text": "primera página del documento"},
            {"item": 5, "kind": "text", "text": "segunda página del documento"},
            {"item": 6, "kind": "text", "text": "tercera página del documento"}]
    index = make_index(items, rows, [unit(c) for c in (0.20, 0.21, 0.55, 0.56, 0.30, 0.30, 0.44)])

    results = index.rank(query_vector, 10)

    # 0.44 loses to 0.56 as a cosine, but the text scores are tightly grouped, so the
    # 0.44 chunk is the most unusual of its modality and comes first.
    assert results[0]["id"] == 6
    assert results[0]["similarity"] == pytest.approx(0.44, abs=0.01)
    assert results[0]["matched_by"] == "text"
    best_image = max(r["similarity"] for r in results if r["matched_by"] == "image")
    assert best_image > results[0]["similarity"]


def test_the_snippet_is_the_closest_text_chunk_even_when_the_image_row_wins(query_vector):
    items = [pdf_page(0, "contrato.pdf"), photo(1, "x.jpg"), photo(2, "y.jpg"), photo(3, "z.jpg")]
    rows = [{"item": 0, "kind": "image"},
            {"item": 0, "kind": "text", "text": "cláusula décima: se admite un perro pequeño"},
            {"item": 0, "kind": "text", "text": "fianza de dos mensualidades"},
            {"item": 0, "kind": "text", "text": "anexo primero: inventario"},
            {"item": 1, "kind": "image"}, {"item": 2, "kind": "image"}, {"item": 3, "kind": "image"}]
    index = make_index(items, rows, [unit(c) for c in (0.90, 0.10, 0.40, 0.12, 0.10, 0.10, 0.10)])

    [top] = index.rank(query_vector, 1)

    assert top["id"] == 0
    assert top["matched_by"] == "image"
    assert top["snippet"] == "fianza de dos mensualidades"


def test_a_code_in_the_query_lifts_the_chunk_that_contains_it_verbatim(query_vector):
    items = [pdf_page(0, "factura.pdf"), pdf_page(1, "otro.pdf")]
    rows = [{"item": 0, "kind": "text", "text": "Factura NR-2026-0931 importe total"},
            {"item": 1, "kind": "text", "text": "un documento distinto, sin identificadores"}]
    index = make_index(items, rows, [unit(0.30), unit(0.40)])

    assert index.rank(query_vector, 2)[0]["id"] == 1  # plain similarity prefers item 1

    results = index.rank(query_vector, 2, text="factura NR-2026-0931")

    assert results[0]["id"] == 0
    assert results[0]["snippet"] == "Factura NR-2026-0931 importe total"


def test_a_code_found_in_most_chunks_is_not_an_identifier(query_vector):
    items = [pdf_page(i, f"doc{i}.pdf") for i in range(4)]
    rows = [{"item": i, "kind": "text", "text": f"ejercicio 2026, documento {i}"} for i in range(4)]
    index = make_index(items, rows, [unit(c) for c in (0.20, 0.30, 0.40, 0.50)])

    results = index.rank(query_vector, 4, text="facturas del 2026")

    # "2026" is in every chunk, so it is a year, not an identifier: no boost, plain order.
    assert [r["id"] for r in results] == [3, 2, 1, 0]


def test_filters_apply_before_the_cut(query_vector):
    items = [photo(0, "a.jpg"), photo(1, "b.jpg"), photo(2, "c.jpg"),
             pdf_page(3, "p.pdf"), pdf_page(4, "q.pdf")]
    rows = [{"item": 0, "kind": "image"}, {"item": 1, "kind": "image"}, {"item": 2, "kind": "image"},
            {"item": 3, "kind": "text", "text": "documento sobre facturas"},
            {"item": 4, "kind": "text", "text": "otro documento sobre facturas"}]
    index = make_index(items, rows, [unit(c) for c in (0.90, 0.10, 0.20, 0.30, 0.40)])

    assert {r["type"] for r in index.rank(query_vector, 10, kind="photo")} == {"photo"}
    assert [r["type"] for r in index.rank(query_vector, 10, kind="pdf")] == ["pdf", "pdf"]
    assert len(index.rank(query_vector, 2)) == 2  # n caps the result list


def test_exclude_drops_one_item_from_the_ranking(query_vector):
    items = [photo(0, "a.jpg"), photo(1, "b.jpg"), photo(2, "c.jpg")]
    rows = [{"item": 0, "kind": "image"}, {"item": 1, "kind": "image"}, {"item": 2, "kind": "image"}]
    index = make_index(items, rows, [unit(c) for c in (0.90, 0.10, 0.20)])

    assert [r["id"] for r in index.rank(query_vector, 10, exclude=0)] == [2, 1]


def test_rank_uses_the_state_it_is_given(query_vector):
    """A rebuild between two reads must not mix two indexes; callers pass one snapshot."""
    index = make_index([photo(0, "a.jpg"), photo(1, "b.jpg"), photo(2, "c.jpg")],
                       [{"item": i, "kind": "image"} for i in range(3)],
                       [unit(0.90), unit(0.10), unit(0.20)])
    snapshot = index.state()

    index.install(search.State(folder=Path("/tmp/other"),
                               items=[photo(0, "z.jpg")],
                               rows=[{"item": 0, "kind": "image"}],
                               vectors=[unit(0.50)]))

    assert [r["name"] for r in index.rank(query_vector, 5, state=snapshot)] == ["a.jpg", "c.jpg", "b.jpg"]
    assert [r["name"] for r in index.rank(query_vector, 5)] == ["z.jpg"]


def test_image_vector_returns_the_stored_image_row(query_vector):
    index = make_index([pdf_page(0, "p.pdf")],
                       [{"item": 0, "kind": "image"},
                        {"item": 0, "kind": "text", "text": "el texto de la página"}],
                       [unit(0.70), unit(0.20)])

    assert index.image_vector(0)[0] == pytest.approx(0.70, abs=0.01)
    with pytest.raises(KeyError):
        index.image_vector(1)
