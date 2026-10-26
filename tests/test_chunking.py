"""chunk_text(), fold() and codes(): the pure text helpers behind PDF indexing and ranking."""
from __future__ import annotations

import search


# ------------------------------------------------------------------ chunk_text


def test_text_too_short_to_be_worth_a_vector_is_dropped():
    assert search.chunk_text("") == []
    assert search.chunk_text("   ") == []
    assert search.chunk_text("x" * 24) == []


def test_text_just_over_the_threshold_becomes_one_chunk():
    assert search.chunk_text("x" * 25) == ["x" * 25]
    assert search.chunk_text("x" * search.CHUNK) == ["x" * search.CHUNK]


def test_runs_of_whitespace_are_folded():
    text = "Quinta —   Gastos\ty suministros\n\n\nTotal: 120"
    assert search.chunk_text(text) == ["Quinta — Gastos y suministros\nTotal: 120"]


def test_a_long_text_is_split_into_overlapping_chunks():
    text = "".join(chr(ord("a") + i % 26) for i in range(search.CHUNK * 3))

    chunks = search.chunk_text(text)

    step = search.CHUNK - search.OVERLAP
    assert all(len(chunk) <= search.CHUNK for chunk in chunks)
    assert chunks[0] == text[:search.CHUNK]
    assert chunks[1] == text[step:step + search.CHUNK]
    # Consecutive chunks share their overlap, so no character falls between two chunks.
    for left, right in zip(chunks, chunks[1:]):
        assert left[-search.OVERLAP:] == right[:search.OVERLAP]
    assert text.endswith(chunks[-1])
    # Every chunk contributes: a dropped chunk would leave the text uncovered.
    assert sum(len(chunk) for chunk in chunks) > len(text)


def test_chunking_covers_every_character():
    text = "".join(str(i % 10) for i in range(search.CHUNK * 2 + 1))
    step = search.CHUNK - search.OVERLAP

    covered = set()
    for i, chunk in enumerate(search.chunk_text(text)):
        assert chunk == text[i * step:i * step + search.CHUNK]  # each chunk is where it should be
        covered.update(range(i * step, i * step + len(chunk)))

    assert covered >= set(range(len(text)))  # no character falls between two chunks


# ------------------------------------------------------------------ fold and codes


def test_fold_strips_accents_and_case():
    assert search.fold("Factura Nº") == "factura no"
    assert search.fold("CAFÉ ÓRDEN") == "cafe orden"
    assert search.fold("a—b") == "ab"  # non-ASCII punctuation goes too


def test_codes_are_words_with_a_digit_and_three_characters():
    assert search.codes("factura NR-2026-0931 del 2026") == ["nr-2026-0931", "2026"]
    assert search.codes("el perro dormido en el sofá") == []
    assert search.codes("at 09") == []  # two characters: not a code


def test_codes_keep_the_separators_inside_them():
    assert search.codes("FAC-000214 y 4321 BCD") == ["fac-000214", "4321"]
