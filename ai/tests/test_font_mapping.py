"""Recover embedded semantics without guessing private-use character meanings."""

import io

import pymupdf
import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable

from paperless_ai.agents.font_mapping import (
    _embedded_mapping,
    _reader_mapping,
    recover_embedded_font_unicode,
    set_font_unicode,
)


def symbol_font(*, names=True, unicode_map=None):
    """Build a tiny embedded font with explicit symbol-code-to-glyph relations."""
    builder = FontBuilder(1000, isTTF=True)
    glyphs = [".notdef", "A", "hyphen"]
    builder.setupGlyphOrder(glyphs)
    builder.setupCharacterMap(unicode_map or {})
    builder.setupGlyf({name: TTGlyphPen(None).glyph() for name in glyphs})
    builder.setupHorizontalMetrics({name: (500, 0) for name in glyphs})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "SymbolFixture", "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost(keepGlyphNames=names)
    symbol = CmapSubtable.newSubtable(4)
    symbol.platformID, symbol.platEncID, symbol.language = 3, 0, 0
    symbol.cmap = {0xF041: "A", 0xF02D: "hyphen"}
    builder.font["cmap"].tables.append(symbol)
    data = io.BytesIO()
    builder.save(data)
    return data.getvalue()


@pytest.mark.parametrize(
    "names,unicode_map", [(True, {}), (False, {65: "A", 45: "hyphen"})]
)
def test_embedded_names_or_unicode_map_resolve_symbol_codes(names, unicode_map):
    """Either trusted font source can supply semantics without a PDF encoding."""
    with TTFont(io.BytesIO(symbol_font(names=names, unicode_map=unicode_map))) as font:
        assert _embedded_mapping(font) == {45: "-", 65: "A"}


@pytest.mark.parametrize("unicode_map", [{}, {0xF041: "A", 0xF02D: "hyphen"}])
def test_private_use_without_names_remains_ambiguous(unicode_map):
    """Private-use labels describe glyph slots, not Unicode characters."""
    with TTFont(io.BytesIO(symbol_font(names=False, unicode_map=unicode_map))) as font:
        assert _embedded_mapping(font) == {}


def test_conflicting_font_evidence_remains_ambiguous():
    """Do not choose between contradictory glyph names and Unicode mappings."""
    with TTFont(io.BytesIO(symbol_font(unicode_map={66: "A", 45: "hyphen"}))) as font:
        assert _embedded_mapping(font) == {}


@pytest.mark.parametrize("flags", [4, 0, 32, 36])
def test_embedded_recovery_connects_codes_to_glyphs_and_keeps_existing_maps(flags):
    """Read embedded bytes, install Unicode and preserve authoritative maps."""
    with pymupdf.open() as document:
        page = document.new_page()
        font = page.insert_font(fontname="fixture", fontbuffer=symbol_font())
        document.xref_set_key(font, "Subtype", "/TrueType")
        document.xref_set_key(font, "Encoding", "null")
        document.xref_set_key(font, "ToUnicode", "null")
        # insert_font uses a CID wrapper; attach its embedded descriptor directly.
        descendant = int(
            document.xref_get_key(font, "DescendantFonts")[1].split()[0][1:]
        )
        descriptor = document.xref_get_key(descendant, "FontDescriptor")[1]
        document.xref_set_key(font, "FontDescriptor", descriptor)
        document.xref_set_key(int(descriptor.split()[0]), "Flags", str(flags))
        if flags != 4:
            assert not recover_embedded_font_unicode(document)
            return
        assert recover_embedded_font_unicode(document)
        mapping = document.xref_get_key(font, "ToUnicode")
        assert mapping[0] == "xref"
        assert not recover_embedded_font_unicode(document)
        assert document.xref_get_key(font, "ToUnicode") == mapping


def test_unembedded_font_remains_unresolved():
    """No system-font substitution is evidence of the original font semantics."""
    with pymupdf.open() as document:
        page = document.new_page()
        font = page.insert_font(fontname="helv")
        document.xref_set_key(font, "Encoding", "null")
        assert not recover_embedded_font_unicode(document)
        set_font_unicode(document, font, {65: "A"})
        assert not recover_embedded_font_unicode(document)


@pytest.mark.parametrize(
    "values,expected",
    [([65, 45], {65: "A", 45: "-"}), ([65, 66], {}), ([0xFFFD, 45], {})],
)
def test_reader_guesses_require_consistent_meanings(values, expected):
    """Preserve valid used-glyph guesses, rejecting contradictions and replacements."""

    class Page:
        def get_fonts(self):
            return [(7, "ttf", "TrueType", "ABCDEF+Fixture")]

        def get_texttrace(self):
            glyphs = [1, 2] if values != [65, 66] else [1, 1]
            return [
                {
                    "font": "Fixture",
                    "chars": [
                        (value, glyph, (), ()) for value, glyph in zip(values, glyphs)
                    ],
                }
            ]

    with TTFont(io.BytesIO(symbol_font(names=False))) as font:
        assert _reader_mapping([Page()], 7, font) == expected


def test_reader_guesses_reject_duplicate_font_names():
    """Do not attach a reader guess when font identity is ambiguous."""

    class Page:
        def get_fonts(self):
            return [
                (7, "ttf", "TrueType", "ABCDEF+Fixture"),
                (8, "ttf", "TrueType", "UVWXYZ+Fixture"),
            ]

        def get_texttrace(self):
            return [{"font": "Fixture", "chars": [(65, 1, (), ())]}]

    with TTFont(io.BytesIO(symbol_font(names=False))) as font:
        assert _reader_mapping([Page()], 7, font) == {}
