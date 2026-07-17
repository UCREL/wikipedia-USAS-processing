import subprocess
import sys
from enum import Enum
from typing import Annotated, List

import spacy
import typer
from rich import print as rprint


class Languages(str, Enum):
    """Wikipedia languages supported by this pipeline's model installation.

    Each member's value is the human-readable language name used as a
    ``--languages``/``-l`` choice on the :func:`install` CLI and as a key
    into the language-to-model mapping dictionaries in this module.
    """

    zh = "Chinese"
    da = "Danish"
    nl = "Dutch"
    en = "English"
    fi = "Finnish"
    it = "Italian"
    pt = "Portuguese"
    es = "Spanish"

class SpacyModel(str, Enum):
    """spaCy model package names installable via :func:`pip_install_model`.

    Each member's value is the spaCy model's package/model name, used as a
    key into :data:`SPACY_MODEL_2_URL` and :data:`SPACY_DESCRIPTIONS`.
    """

    zh_sm = "zh_core_web_sm"
    zh_md = "zh_core_web_md"
    zh_trf = "zh_core_web_trf"

    da_sm = "da_core_news_sm"
    da_trf = "da_core_news_trf"

    nl_md = "nl_core_news_md"
    nl_lg = "nl_core_news_lg"

    en_sm  = "en_core_web_sm"
    en_trf = "en_core_web_trf"

    fi_sm = "fi_core_news_sm"
    fi_lg = "fi_core_news_lg"

    it_sm = "it_core_news_sm"
    it_lg = "it_core_news_lg"

    pt_sm = "pt_core_news_sm"
    pt_lg = "pt_core_news_lg"
    
    es_sm = "es_core_news_sm"
    es_trf = "es_dep_news_trf"


LANGUAGE_2_SPACY_MODEL: dict[Languages, List[SpacyModel]] = {
    Languages.zh: [SpacyModel.zh_sm, SpacyModel.zh_md, SpacyModel.zh_trf],
    Languages.da: [SpacyModel.da_sm, SpacyModel.da_trf],
    Languages.nl: [SpacyModel.nl_md, SpacyModel.nl_lg],
    Languages.en: [SpacyModel.en_sm, SpacyModel.en_trf],
    Languages.fi: [SpacyModel.fi_sm, SpacyModel.fi_lg],
    Languages.it: [SpacyModel.it_sm, SpacyModel.it_lg],
    Languages.pt: [SpacyModel.pt_sm, SpacyModel.pt_lg],
    Languages.es: [SpacyModel.es_sm, SpacyModel.es_trf]
}

LANGUAGE_2_PYMUSAS_SPACY_MODEL: dict[Languages, str] = {
    Languages.zh: "cmn_dual_upos2usas_contextual_none",
    Languages.da: "da_dual_none_contextual_none",
    Languages.nl: "nl_single_upos2usas_contextual_none",
    Languages.en: "en_dual_none_contextual_none",
    Languages.fi: "fi_single_upos2usas_contextual_none",
    Languages.it: "it_dual_upos2usas_contextual_none",
    Languages.pt: "pt_dual_upos2usas_contextual_none",
    Languages.es: "es_dual_upos2usas_contextual_none",
}

PYMUSAS_SPACY_MODEL_2_URL: dict[str, str] = {
    "cmn_dual_upos2usas_contextual_none": "https://github.com/UCREL/pymusas-models/releases/download/cmn_dual_upos2usas_contextual_none-0.4.0/cmn_dual_upos2usas_contextual_none-0.4.0-py3-none-any.whl",
    "da_dual_none_contextual_none": "https://github.com/UCREL/pymusas-models/releases/download/da_dual_none_contextual_none-0.4.1/da_dual_none_contextual_none-0.4.1-py3-none-any.whl",
    "nl_single_upos2usas_contextual_none": "https://github.com/UCREL/pymusas-models/releases/download/nl_single_upos2usas_contextual_none-0.4.0/nl_single_upos2usas_contextual_none-0.4.0-py3-none-any.whl",
    "en_dual_none_contextual_none": "https://github.com/UCREL/pymusas-models/releases/download/en_dual_none_contextual_none-0.4.0/en_dual_none_contextual_none-0.4.0-py3-none-any.whl",
    "fi_single_upos2usas_contextual_none": "https://github.com/UCREL/pymusas-models/releases/download/fi_single_upos2usas_contextual_none-0.4.0/fi_single_upos2usas_contextual_none-0.4.0-py3-none-any.whl",
    "it_dual_upos2usas_contextual_none": "https://github.com/UCREL/pymusas-models/releases/download/it_dual_upos2usas_contextual_none-0.4.0/it_dual_upos2usas_contextual_none-0.4.0-py3-none-any.whl",
    "pt_dual_upos2usas_contextual_none": "https://github.com/UCREL/pymusas-models/releases/download/pt_dual_upos2usas_contextual_none-0.4.0/pt_dual_upos2usas_contextual_none-0.4.0-py3-none-any.whl",
    "es_dual_upos2usas_contextual_none": "https://github.com/UCREL/pymusas-models/releases/download/es_dual_upos2usas_contextual_none-0.4.0/es_dual_upos2usas_contextual_none-0.4.0-py3-none-any.whl"
}

    
SPACY_DESCRIPTIONS: dict[SpacyModel, str] = {
    SpacyModel.zh_sm:  "Chinese - Small (46MB)",
    SpacyModel.zh_md:  "Chinese - Medium (74MB)",
    SpacyModel.zh_trf: "Chinese - Transformer-based (396MB)",
    SpacyModel.da_sm:  "Danish - Small (11MB)",
    SpacyModel.da_trf: "Danish - Transformer-based (420MB)",
    SpacyModel.nl_md:  "Dutch - Medium (40MB)",
    SpacyModel.nl_lg:  "Dutch - Large (541MB)",
    SpacyModel.en_sm:  "English - Small (12MB)",
    SpacyModel.en_trf: "English - Transformer-based (438MB)",
    SpacyModel.fi_sm:  "Finnish - Small (13MB)",
    SpacyModel.fi_lg:  "Finnish - Large (220MB)",
    SpacyModel.it_sm:  "Italian - Small (12MB)",
    SpacyModel.it_lg:  "Italian - Large (541MB)",
    SpacyModel.pt_sm:  "Portuguese - Small (12MB)",
    SpacyModel.pt_lg:  "Portuguese - Large (541MB)",
    SpacyModel.es_sm:  "Spanish - Small (12MB)",
    SpacyModel.es_trf:  "Spanish - Transformer-based (388MB)",
}

PYMUSAS_SPACY_MODELS_DESCRIPTIONS: dict[Languages, str] = {
    Languages.zh: "PyMUSAS Chinese Rule Based Model - (1.28MB)",
    Languages.da: "PyMUSAS Danish Rule Based Model - (0.82MB)",
    Languages.nl: "PyMUSAS Dutch Rule Based Model - (0.15MB)",
    Languages.en: "PyMUSAS English Rule Based Model - (0.86MB)",
    Languages.fi: "PyMUSAS Finnish Rule Based Model - (0.64MB)",
    Languages.it: "PyMUSAS Italian Rule Based Model - (0.50MB)",
    Languages.pt: "PyMUSAS Portuguese Rule Based Model - (0.27MB)",
    Languages.es: "PyMUSAS Spanish Rule Based Model - (0.26MB)",
}

SPACY_MODEL_2_URL: dict[SpacyModel, str] = {
    SpacyModel.zh_sm: 'https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl',
    SpacyModel.zh_md: 'https://github.com/explosion/spacy-models/releases/download/zh_core_web_md-3.8.0/zh_core_web_md-3.8.0-py3-none-any.whl',
    SpacyModel.zh_trf: 'https://github.com/explosion/spacy-models/releases/download/zh_core_web_trf-3.8.0/zh_core_web_trf-3.8.0-py3-none-any.whl',
    SpacyModel.da_sm: 'https://github.com/explosion/spacy-models/releases/download/da_core_news_sm-3.8.0/da_core_news_sm-3.8.0-py3-none-any.whl',
    SpacyModel.da_trf: 'https://github.com/explosion/spacy-models/releases/download/da_core_news_trf-3.8.0/da_core_news_trf-3.8.0-py3-none-any.whl',
    SpacyModel.nl_md: 'https://github.com/explosion/spacy-models/releases/download/nl_core_news_md-3.8.0/nl_core_news_md-3.8.0-py3-none-any.whl',
    SpacyModel.nl_lg: 'https://github.com/explosion/spacy-models/releases/download/nl_core_news_lg-3.8.0/nl_core_news_lg-3.8.0-py3-none-any.whl',
    SpacyModel.en_sm: 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl',
    SpacyModel.en_trf: 'https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl',
    SpacyModel.fi_sm: 'https://github.com/explosion/spacy-models/releases/download/fi_core_news_sm-3.8.0/fi_core_news_sm-3.8.0-py3-none-any.whl',
    SpacyModel.fi_lg: 'https://github.com/explosion/spacy-models/releases/download/fi_core_news_lg-3.8.0/fi_core_news_lg-3.8.0-py3-none-any.whl',
    SpacyModel.it_sm: 'https://github.com/explosion/spacy-models/releases/download/it_core_news_sm-3.8.0/it_core_news_sm-3.8.0-py3-none-any.whl',
    SpacyModel.it_lg: 'https://github.com/explosion/spacy-models/releases/download/it_core_news_lg-3.8.0/it_core_news_lg-3.8.0-py3-none-any.whl',
    SpacyModel.pt_sm: 'https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.8.0/pt_core_news_sm-3.8.0-py3-none-any.whl',
    SpacyModel.pt_lg: 'https://github.com/explosion/spacy-models/releases/download/pt_core_news_lg-3.8.0/pt_core_news_lg-3.8.0-py3-none-any.whl',
    SpacyModel.es_sm: 'https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl',
    SpacyModel.es_trf: 'https://github.com/explosion/spacy-models/releases/download/es_dep_news_trf-3.8.0/es_dep_news_trf-3.8.0-py3-none-any.whl',
}

def pip_install_model(wheel_url: str, spacy_model_name: str | None = None) -> None:
    """Install a model package from a wheel URL via pip.

    If `spacy_model_name` is provided and is already installed as a spaCy
    package, installation is skipped.

    Args:
        wheel_url: URL or path to the wheel file to install via pip.
        spacy_model_name: Name of the spaCy model package to check before
            installing. If None, no pre-check is performed and the wheel
            is always installed.

    Returns:
        None. Prints status messages indicating whether the model was
        already installed, installed successfully, or failed to install.
    """
    rprint(f"Pip installing the following wheel: {wheel_url}")
    
    if spacy_model_name is not None and spacy.util.is_package(spacy_model_name):
        rprint(f"[green]✓ {spacy_model_name} is already installed[/green]")
        return
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", wheel_url],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        rprint(f"[green]✓ installed successfully: {wheel_url} [/green]")
    else:
        rprint(f"[red]✗ Failed to install: {wheel_url} {result.stderr}[/red]")


def install(
    languages: Annotated[list[Languages] | None, typer.Option("--languages", "-l", help="Install the language specific models for the given languages.")] = None,
    all_languages: Annotated[bool, typer.Option("--all", "-a", help="Install all language specific models.")] = False,
    describe: Annotated[bool, typer.Option("--describe", "-d", help="Describe the models that will be installed and exit.")] = False
    ):
    """Typer CLI entry point to install spaCy and PyMUSAS models per language.

    You can either select the languages you want to install with
    ``--languages``/``-l`` (repeatable) or use the ``--all`` flag to install
    every supported language's models. Pass ``--describe`` to list the
    models that would be installed instead of installing them.

    Args:
        languages: Languages to install models for. May be passed multiple
            times (e.g. ``-l English -l Dutch``). Ignored if all_languages
            is True.
        all_languages: If True, install models for every supported
            language, overriding languages.
        describe: If True, print the models that would be installed for
            the selected languages and exit without installing anything.

    Raises:
        typer.Exit: With code 1 if neither languages nor all_languages
            selects any language. With code 0 after printing the
            description, if describe is True.

    Examples:
        To install all language specific models::

            $ python models_install.py --all

        To install only the English and Dutch language specific models::

            $ python models_install.py -l English -l Dutch

        To describe the models that will be installed for English::

            $ python models_install.py -l English --describe
    """
    selected = languages
    if all_languages:
        selected = list(Languages)
    if selected is None or len(selected) == 0:
        rprint("No languages selected, either use --languages or --all")
        raise typer.Exit(1)

    if describe:
        rprint("Describing the models that will be installed:")
        for language in selected:
            rprint(f"Models that will be installed for this language {language}:")
            if LANGUAGE_2_SPACY_MODEL.get(language):
                rprint("spaCy models that will be installed:")
                for model in LANGUAGE_2_SPACY_MODEL[language]:
                    rprint(f" {SPACY_DESCRIPTIONS[model]}")
            if PYMUSAS_SPACY_MODELS_DESCRIPTIONS.get(language):
                rprint("PyMUSAS spaCy models that will be installed:")
                rprint(f"{PYMUSAS_SPACY_MODELS_DESCRIPTIONS[language]}")
            rprint()
        rprint("Done")
                
        raise typer.Exit(0)

    rprint(f"Installing {len(selected)} language specific models, some languages do use more than one model")
    for language in selected:
        rprint(f"Installing {language} specific models")
        if LANGUAGE_2_SPACY_MODEL.get(language):
            for spacy_model in LANGUAGE_2_SPACY_MODEL[language]:
                pip_install_model(SPACY_MODEL_2_URL[spacy_model], spacy_model_name=spacy_model.value)
        if LANGUAGE_2_PYMUSAS_SPACY_MODEL.get(language):
            pymusas_model_name = LANGUAGE_2_PYMUSAS_SPACY_MODEL[language]
            pip_install_model(PYMUSAS_SPACY_MODEL_2_URL[pymusas_model_name], spacy_model_name=pymusas_model_name)
        rprint(f"Done installing {language} specific models")
    rprint(f"Done installing {len(selected)} language specific models")
        
        

if __name__ == "__main__":
    typer.run(install)