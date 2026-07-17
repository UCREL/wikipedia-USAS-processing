import sys
from unittest.mock import Mock, patch

import pytest
import typer

from wikipedia_processing.models_install import (
    LANGUAGE_2_PYMUSAS_SPACY_MODEL,
    LANGUAGE_2_SPACY_MODEL,
    PYMUSAS_SPACY_MODEL_2_URL,
    SPACY_DESCRIPTIONS,
    SPACY_MODEL_2_URL,
    Languages,
    install,
    pip_install_model,
)


def test_pip_install_model_skips_already_installed_package(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("wikipedia_processing.models_install.spacy.util.is_package", return_value=True),
        patch("wikipedia_processing.models_install.subprocess.run") as mock_run,
    ):
        pip_install_model("http://example.com/model.whl", "en_core_web_sm")

    mock_run.assert_not_called()
    assert "already installed" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("returncode", "expected_substring"),
    [
        # pip install succeeded.
        (0, "installed successfully"),
        # pip install failed; stderr is included in the failure message.
        (1, "Failed to install"),
    ],
    ids=["success", "failure"],
)
def test_pip_install_model_reports_subprocess_outcome(
    returncode: int, expected_substring: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch("wikipedia_processing.models_install.spacy.util.is_package", return_value=False),
        patch(
            "wikipedia_processing.models_install.subprocess.run",
            return_value=Mock(returncode=returncode, stderr="boom"),
        ) as mock_run,
    ):
        pip_install_model("http://example.com/model.whl", "en_core_web_sm")

    mock_run.assert_called_once_with(
        [sys.executable, "-m", "pip", "install", "http://example.com/model.whl"],
        capture_output=True,
        text=True,
    )
    assert expected_substring in capsys.readouterr().out

@pytest.mark.parametrize(
    "languages",
    [None, []],
    ids=["no-language", "empty-language-list"],
)
def test_install_raises_when_no_languages_selected(languages: None | list[Languages]) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        install(languages=languages, all_languages=False, describe=False)

    assert exc_info.value.exit_code == 1

def test_install_describe_prints_descriptions_without_installing(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("wikipedia_processing.models_install.pip_install_model") as mock_install,
        pytest.raises(typer.Exit) as exc_info,
    ):
        install(languages=[Languages.en], all_languages=False, describe=True)

    assert exc_info.value.exit_code == 0
    mock_install.assert_not_called()
    output = capsys.readouterr().out
    for spacy_model in LANGUAGE_2_SPACY_MODEL[Languages.en]:
        assert SPACY_DESCRIPTIONS[spacy_model] in output


def test_install_all_languages_overrides_explicit_language_list() -> None:
    with patch("wikipedia_processing.models_install.pip_install_model") as mock_install:
        install(languages=[Languages.en], all_languages=True, describe=False)

    expected_call_count = sum(len(models) for models in LANGUAGE_2_SPACY_MODEL.values()) + len(
        LANGUAGE_2_PYMUSAS_SPACY_MODEL
    )
    assert mock_install.call_count == expected_call_count
    # Danish was never in the explicit `languages` list, proving `--all` won.
    danish_model = LANGUAGE_2_SPACY_MODEL[Languages.da][0]
    mock_install.assert_any_call(SPACY_MODEL_2_URL[danish_model], spacy_model_name=danish_model.value)


def test_install_selected_language_installs_its_models() -> None:
    with patch("wikipedia_processing.models_install.pip_install_model") as mock_install:
        install(languages=[Languages.en], all_languages=False, describe=False)

    expected_call_count = len(LANGUAGE_2_SPACY_MODEL[Languages.en]) + 1  # + 1 PyMUSAS model
    assert mock_install.call_count == expected_call_count
    pymusas_model_name = LANGUAGE_2_PYMUSAS_SPACY_MODEL[Languages.en]
    mock_install.assert_any_call(
        PYMUSAS_SPACY_MODEL_2_URL[pymusas_model_name], spacy_model_name=pymusas_model_name
    )
