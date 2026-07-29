import time
from pathlib import Path
from unittest.mock import patch

import pytest
from datatrove.data import Document
from datatrove.utils.logging import logger as data_trove_logger

from wikipedia_processing.utils import (
    create_sub_directory,
    get_hashes_per_bucket,
    get_progress_logger_function,
    get_usas_language_processing_information,
    get_valid_usas_language_processing_wikipedia_codes,
    time_elapsed,
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
        "data_trove_language": "spa",
    }


def test_get_valid_usas_language_processing_wikipedia_codes_returns_all_codes(tmp_path: Path) -> None:
    language_data_file = _write_language_data_file(tmp_path)
    assert get_valid_usas_language_processing_wikipedia_codes(language_data_file) == ["en", "nl"]


def test_get_valid_usas_language_processing_wikipedia_codes_default_file_loads_packaged_data() -> None:
    # No language_data_file given -> falls back to the packaged
    # wikipedia_processing/data/usas_wikipedia_processing.yaml, which is
    # expected to include Spanish's "es" code among others.
    assert "es" in get_valid_usas_language_processing_wikipedia_codes()


def test_create_sub_directory_creates_new_directory(tmp_path: Path) -> None:
    sub_directory = create_sub_directory(tmp_path, "sub")
    assert Path(sub_directory) == (tmp_path / "sub").resolve()
    assert Path(sub_directory).is_dir()


def test_create_sub_directory_existing_directory_left_untouched(tmp_path: Path) -> None:
    existing = tmp_path / "sub"
    existing.mkdir()
    marker = existing / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")

    sub_directory = create_sub_directory(tmp_path, "sub")

    assert Path(sub_directory) == existing.resolve()
    assert marker.read_text(encoding="utf-8") == "keep me"


def test_create_sub_directory_creates_missing_parents(tmp_path: Path) -> None:
    # "a/b" has no existing intermediate directory "a" under tmp_path.
    sub_directory = create_sub_directory(tmp_path, "a/b")
    assert Path(sub_directory) == (tmp_path / "a" / "b").resolve()
    assert Path(sub_directory).is_dir()


def test_create_sub_directory_returns_resolved_absolute_path(tmp_path: Path) -> None:
    sub_directory = create_sub_directory(tmp_path, "sub")
    assert Path(sub_directory).is_absolute()


@pytest.mark.parametrize(
    ("num_buckets", "threshold", "expected"),
    [
        # Docstring example.
        (14, 0.72, 8),
        # Smallest valid num_buckets, threshold at the midpoint.
        (2, 0.5, 1),
        # A threshold near 1 requires many hashes per bucket to distinguish sets.
        (2, 0.99, 69),
        # A very loose threshold can round down to 0, as noted in the docstring.
        (2, 0.001, 0),
    ],
    ids=[
        "docstring-example",
        "minimal-num-buckets",
        "threshold-near-one",
        "loose-threshold-rounds-to-zero",
    ],
)
def test_get_hashes_per_bucket(num_buckets: int, threshold: float, expected: int) -> None:
    assert get_hashes_per_bucket(num_buckets, threshold) == expected


def test_get_hashes_per_bucket_num_buckets_below_two_raises_value_error() -> None:
    with pytest.raises(ValueError, match="num_buckets must be >= 2"):
        get_hashes_per_bucket(1, 0.5)


@pytest.mark.parametrize(
    "threshold",
    [0.0, 1.0, -0.5, 1.5],
    ids=["zero", "one", "negative", "greater-than-one"],
)
def test_get_hashes_per_bucket_threshold_outside_open_interval_raises_value_error(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        get_hashes_per_bucket(num_buckets=14, threshold=threshold)


def test_get_progress_logger_function_yields_documents_unchanged() -> None:
    documents = [Document(text="a", id="1"), Document(text="b", id="2")]
    log_progress = get_progress_logger_function("reading")

    with patch.object(data_trove_logger, "info"):
        result = list(log_progress(iter(documents), 0, 1))  # ty: ignore[invalid-argument-type]

    assert result == documents


def test_get_progress_logger_function_logs_progress_at_log_every_interval() -> None:
    documents = [Document(text=str(i), id=str(i)) for i in range(5)]
    log_progress = get_progress_logger_function("reading", log_every=2)

    with patch.object(data_trove_logger, "info") as mock_info:
        list(log_progress(iter(documents), 0, 1))  # ty: ignore[invalid-argument-type]

    # Progress lines fire at counts 2 and 4, then a final "finished" line.
    assert mock_info.call_count == 3
    assert "processed 2 documents so far" in mock_info.call_args_list[0].args[0]
    assert "processed 4 documents so far" in mock_info.call_args_list[1].args[0]
    assert "finished, processed 5 documents total" in mock_info.call_args_list[2].args[0]


def test_get_progress_logger_function_includes_step_name_and_rank_in_messages() -> None:
    documents = [Document(text="a", id="1")]
    log_progress = get_progress_logger_function("annotate")

    with patch.object(data_trove_logger, "info") as mock_info:
        list(log_progress(iter(documents), 3, 8))  # ty: ignore[invalid-argument-type]

    mock_info.assert_called_once()
    message = mock_info.call_args.args[0]
    assert "[annotate]" in message
    assert "rank=3" in message
    assert "finished, processed 1 documents total" in message


def test_get_progress_logger_function_data_none_processes_zero_documents() -> None:
    log_progress = get_progress_logger_function("reading")

    with patch.object(data_trove_logger, "info") as mock_info:
        result = list(log_progress(None, 0, 1))  # ty: ignore[invalid-argument-type]

    assert result == []
    mock_info.assert_called_once()
    assert "finished, processed 0 documents total" in mock_info.call_args.args[0]


def test_get_progress_logger_function_empty_generator_processes_zero_documents() -> None:
    log_progress = get_progress_logger_function("reading")

    with patch.object(data_trove_logger, "info") as mock_info:
        result = list(log_progress(iter([]), 0, 1))  # ty: ignore[invalid-argument-type]

    assert result == []
    mock_info.assert_called_once()
    assert "finished, processed 0 documents total" in mock_info.call_args.args[0]


def test_time_elapsed_computes_difference_from_start_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deterministic perf_counter reading so the elapsed time is exact, not
    # just "close to" some wall-clock delta.
    monkeypatch.setattr(time, "perf_counter", lambda: 10.0)
    assert time_elapsed(4.0) == 6.0


def test_time_elapsed_zero_when_start_time_is_now(monkeypatch: pytest.MonkeyPatch) -> None:
    # Boundary case: start_time equal to the current reading yields no elapsed time.
    monkeypatch.setattr(time, "perf_counter", lambda: 5.0)
    assert time_elapsed(5.0) == 0.0
