# -*- coding: utf-8 -*-
"""
test_glossary_language.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the bilingual glossary.

The English catalogue is the source of the wording, the Indonesian one is a
translation keyed exactly like it. These tests keep the two in step: a term
added to one without the other is a bug, and so is a "translation" that is
still the English sentence.
"""

import pytest

import glossary
import glossary_id


ENGLISH = glossary.TERMS
INDONESIAN = glossary_id.TERMS


class TestCatalogueParity:
    def test_every_english_term_has_an_indonesian_one(self):
        missing = sorted(set(ENGLISH) - set(INDONESIAN))
        assert missing == [], f"no Indonesian wording for: {missing}"

    def test_no_indonesian_term_is_unknown(self):
        extra = sorted(set(INDONESIAN) - set(ENGLISH))
        assert extra == [], f"Indonesian terms the app never asks for: {extra}"

    @pytest.mark.parametrize('term', sorted(ENGLISH))
    def test_entry_has_a_display_name_short_and_long(self, term):
        entry = INDONESIAN[term]
        for field in ('term', 'short', 'long'):
            value = entry.get(field, '')
            assert isinstance(value, str) and value.strip(), f"{term}.{field} is empty"

    @pytest.mark.parametrize('term', sorted(ENGLISH))
    def test_the_translation_is_not_the_english_sentence(self, term):
        assert INDONESIAN[term]['short'] != ENGLISH[term]['short']
        assert INDONESIAN[term]['long'] != ENGLISH[term]['long']

    def test_the_long_version_says_more_than_the_short_one(self):
        for term, entry in INDONESIAN.items():
            assert len(entry['long']) > len(entry['short']), term


class TestLanguageHelpers:
    def test_english_is_the_default(self):
        assert glossary.language(None) == 'en'
        assert glossary.language('') == 'en'
        assert glossary.language('fr') == 'en'

    def test_a_known_code_is_kept(self):
        assert glossary.language('id') == 'id'
        assert glossary.language(' ID ') == 'id'

    def test_short_and_long_follow_the_language(self):
        assert glossary.short('pit stop') == ENGLISH['pit stop']['short']
        assert glossary.short('pit stop', lang='id') == INDONESIAN['pit stop']['short']
        assert glossary.long('pit stop', lang='id') == INDONESIAN['pit stop']['long']

    def test_an_unknown_term_returns_the_default(self):
        assert glossary.short('nonsense', default='-', lang='id') == '-'
        assert glossary.long('nonsense', default='-', lang='id') == '-'

    def test_terms_for_uses_the_display_name_of_the_language(self):
        english = glossary.terms_for(['pit stop', 'tyre'])
        indonesian = glossary.terms_for(['pit stop', 'tyre'], lang='id')
        assert [entry['term'] for entry in english] == ['Pit Stop', 'Tyre']
        assert [entry['term'] for entry in indonesian] == ['Pit stop', 'Ban']

    def test_terms_for_keeps_the_order_and_skips_unknown_terms(self):
        entries = glossary.terms_for(['tyre', 'nonsense', 'race'], lang='id')
        assert [entry['term'] for entry in entries] == ['Ban', 'Balapan']

    def test_all_entries_lists_the_whole_vocabulary_in_both_languages(self):
        english = glossary.all_entries()
        indonesian = glossary.all_entries(lang='id')
        assert len(english) == len(ENGLISH)
        assert len(indonesian) == len(ENGLISH)
        assert [entry['term'] for entry in indonesian] != [
            entry['term'] for entry in english
        ]

    def test_h2h_keeps_its_short_name(self):
        names = [entry['term'] for entry in glossary.all_entries()]
        assert 'H2H' in names
