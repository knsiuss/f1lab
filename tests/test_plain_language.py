# -*- coding: utf-8 -*-
"""Guards for the plain-language design of the pages.

The app is read by people who do not follow Formula 1, so every page has to
explain itself: a plain-language line under the section titles, a sentence that
says what the numbers add up to, and a glossary at the bottom. These tests are
about behaviour the reader sees, not about how the page is implemented.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

PAGES_DIR = ROOT / 'src' / 'pages'


def _pages():
    return sorted(PAGES_DIR.glob('*.py'))


def test_every_page_exists():
    pages = _pages()
    assert len(pages) == 7, f'expected 7 pages, found {len(pages)}'


def test_no_page_uses_a_dataframe():
    """st.dataframe brings its own look; the app has its own table."""
    for page in _pages():
        source = page.read_text(encoding='utf-8')
        assert 'st.dataframe' not in source, (
            f'{page.name} uses st.dataframe; use ui.data_table(...)'
        )


def test_every_page_explains_itself():
    """Each page explains the words it uses and says what the numbers mean."""
    for page in _pages():
        source = page.read_text(encoding='utf-8')
        assert 'ui.glossary(' in source or 'ui.explain(' in source, (
            f'{page.name} has no glossary and no explanation of its terms'
        )
        assert 'ui.steps(' in source, (
            f'{page.name} has no "how to read this page" block'
        )
        assert 'ui.verdict(' in source, (
            f'{page.name} never says what its numbers add up to'
        )


def test_every_page_has_a_plain_language_section_hint():
    """A section explains itself in one line under its title."""
    for page in _pages():
        source = page.read_text(encoding='utf-8')
        assert _has_section_hint(source), (
            f'{page.name} has no section with a plain-language hint'
        )


def _has_section_hint(source: str) -> bool:
    """True when at least one ``ui.section(...)`` call passes a ``hint``.

    The call is inspected as text rather than with a strict pattern, because a
    title is allowed to contain a comma ("Points, round by round") and the
    second argument may be positional or a keyword.
    """
    index = source.find('ui.section(')
    while index != -1:
        end = source.find(')', index)
        call = source[index:end if end != -1 else len(source)]
        if 'hint=' in call:
            return True
        index = source.find('ui.section(', index + 1)
    return False


def test_the_glossary_covers_the_words_the_app_uses():
    """Every term a page asks the glossary for exists."""
    import glossary

    asked = set()
    for page in _pages():
        source = page.read_text(encoding='utf-8')
        for block in re.findall(r'ui\.glossary\(\s*\[(.*?)\]', source, re.S):
            for name in re.findall(r"['\"]([^'\"]+)['\"]", block):
                asked.add(name.strip().lower())
    for term in sorted(asked):
        assert term in glossary.TERMS, (
            f'the pages ask the glossary for {term!r}, which it does not define'
        )


def test_the_glossary_is_written_for_a_newcomer():
    """Every definition is a sentence, not a word, and not empty."""
    import glossary

    for term, entry in glossary.TERMS.items():
        for key in ('short', 'long'):
            text = entry.get(key, '')
            assert text.strip(), f'{term} has no {key} definition'
            assert text.strip().endswith('.'), (
                f'the {key} definition of {term} is not a sentence'
            )
            assert len(text.split()) >= 5, (
                f'the {key} definition of {term} is too short to explain anything'
            )


def test_the_sidebar_offers_a_guide_for_newcomers():
    """The app has a first-time visitor's guide in the sidebar."""
    source = (ROOT / 'src' / 'f1.py').read_text(encoding='utf-8')
    assert 'New to F1' in source, (
        'the sidebar must offer a guide for people who do not know the sport'
    )
