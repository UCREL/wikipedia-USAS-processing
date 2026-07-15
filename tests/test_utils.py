import pytest

from wikipedia_processing.utils import truncate_to_255_bytes


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
