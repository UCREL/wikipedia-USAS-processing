from pathlib import Path

import pytest

from wikipedia_processing.utils import (
    get_usas_language_processing_information,
    truncate_to_255_bytes,
)


@pytest.mark.parametrize(
    ("string_to_truncate", "expected"),
    [
        # Empty string stays empty.
        ("", ""),
        # Single ASCII character, well under the byte limit.
        ("a", "a"),
        # Exactly 255 ASCII bytes is left untouched.
        ("a" * 255, "a" * 255),
        # 256 ASCII bytes gets truncated by exactly one character.
        ("a" * 256, "a" * 255),
        # Whitespace-only input is short enough to pass through unchanged.
        ("   ", "   "),
        # A 3-byte-per-character string (e.g. CJK) that lands exactly on a
        # character boundary at the 255 byte cut.
        ("字" * 85, "字" * 85),
        # A 3-byte-per-character string where the 255 byte cut falls mid
        # character; the incomplete trailing bytes are dropped.
        ("字" * 86, "字" * 85),
    ],
    ids=[
        "empty-string",
        "single-ascii-char",
        "exactly-255-ascii-bytes",
        "256-ascii-bytes-truncated",
        "whitespace-only",
        "multibyte-on-boundary",
        "multibyte-mid-character-cut",
    ],
)
def test_truncate_to_255_bytes(string_to_truncate: str, expected: str) -> None:
    assert truncate_to_255_bytes(string_to_truncate) == expected


def _write_language_data_file(tmp_path: Path) -> Path:
    language_data_file = tmp_path / "languages.yaml"
    language_data_file.write_text(
        """
languages:
  - language: English
    iso_639_3: eng
    wikipedia_code: en
    training: true
    data_trove_language: english
  - language: Dutch
    iso_639_3: nld
    wikipedia_code: nl
    training: true
    data_trove_language: dutch
""",
        encoding="utf-8",
    )
    return language_data_file


def test_get_usas_language_processing_information_returns_matching_language(tmp_path: Path) -> None:
    language_data_file = _write_language_data_file(tmp_path)
    assert get_usas_language_processing_information("nl", language_data_file) == {
        "language": "Dutch",
        "iso_639_3": "nld",
        "wikipedia_code": "nl",
        "training": True,
        "data_trove_language": "dutch",
    }


def test_get_usas_language_processing_information_unknown_code_raises_value_error(tmp_path: Path) -> None:
    language_data_file = _write_language_data_file(tmp_path)
    with pytest.raises(ValueError, match="xx"):
        get_usas_language_processing_information("xx", language_data_file)


def test_get_usas_language_processing_information_default_file_loads_packaged_data() -> None:
    # No language_data_file given -> falls back to the packaged
    # wikipedia_processing/data/usas_wikipedia_processing.yaml. Uses a language
    # other than English/Dutch so a match here can't be coincidentally
    # satisfied by the _write_language_data_file fixture data used above.
    result = get_usas_language_processing_information("es")
    assert result == {
        "language": "Spanish",
        "iso_639_3": "spa",
        "wikipedia_code": "es",
        "training": True,
        "data_trove_language": "spanish",
    }
