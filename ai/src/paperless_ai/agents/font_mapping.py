"""Recover character semantics only when embedded font data is unambiguous."""

import io
import struct
import unicodedata

from fontTools.agl import toUnicode
from fontTools.ttLib import TTFont, TTLibError


def set_font_unicode(document, xref: int, mapping: dict[int, str]) -> None:
    """Attach a one-byte character-to-Unicode map to a simple PDF font."""
    entries = [
        f"<{code:02x}> <{text.encode('utf-16-be').hex()}>"
        for code, text in sorted(mapping.items())
    ]
    cmap = (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /RecoveredUnicode def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<00> <ff>\nendcodespacerange\n"
    )
    for offset in range(0, len(entries), 100):
        chunk = entries[offset : offset + 100]
        cmap += f"{len(chunk)} beginbfchar\n" + "\n".join(chunk) + "\nendbfchar\n"
    cmap += "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    stream = document.get_new_xref()
    document.update_object(stream, "<<>>")
    document.update_stream(stream, cmap.encode("ascii"))
    document.xref_set_key(xref, "ToUnicode", f"{stream} 0 R")


def _symbol_characters(font: TTFont) -> dict[int, str]:
    """Read Windows symbol character codes without treating them as Unicode."""
    tables = [t for t in font["cmap"].tables if (t.platformID, t.platEncID) == (3, 0)]
    if len(tables) != 1:
        return {}
    result = {}
    for code, glyph in tables[0].cmap.items():
        if 0xF000 <= code <= 0xF0FF:
            code -= 0xF000
        if glyph == ".notdef":
            continue
        if not 0 <= code <= 255 or code in result and result[code] != glyph:
            return {}
        result[code] = glyph
    return result


def _semantic_text(text: str) -> bool:
    """Reject private-use, control and unassigned values as semantic evidence."""
    return bool(text) and all(
        not unicodedata.category(char).startswith("C") for char in text
    )


def _embedded_mapping(font: TTFont) -> dict[int, str]:
    """Resolve symbol glyphs from real Unicode tables or descriptive glyph names."""
    characters = _symbol_characters(font)
    if not characters:
        return {}
    unicode_by_glyph = {}
    for table in font["cmap"].tables:
        if table.isUnicode():
            for code, glyph in table.cmap.items():
                text = chr(code)
                if _semantic_text(text):
                    unicode_by_glyph.setdefault(glyph, set()).add(text)
    mapping = {}
    named = "post" in font and font["post"].formatType != 3
    for code, glyph in characters.items():
        meanings = unicode_by_glyph.get(glyph, set()).copy()
        if named:
            text = toUnicode(glyph)
            if _semantic_text(text):
                meanings.add(text)
        if len(meanings) != 1:
            return {}
        mapping[code] = meanings.pop()
    return mapping


def _reader_mapping(document, xref: int, font: TTFont) -> dict[int, str]:
    """Preserve MuPDF's consistent guesses for glyphs used by an ambiguous font."""
    codes = {
        code: font.getGlyphID(name) for code, name in _symbol_characters(font).items()
    }
    meanings = {}
    for page in document:
        names = {}
        for reference in page.get_fonts():
            names.setdefault(reference[3].split("+")[-1], set()).add(reference[0])
        for span in page.get_texttrace():
            refs = names.get(span["font"].split("+")[-1], set())
            if xref not in refs:
                continue
            if refs != {xref}:
                return {}
            for value, glyph, _, _ in span["chars"]:
                if not 0 <= value <= 0x10FFFF or value == 0xFFFD:
                    return {}
                text = chr(value)
                if not _semantic_text(text) or glyph not in codes.values():
                    return {}
                if meanings.setdefault(glyph, text) != text:
                    return {}
    return {code: meanings[glyph] for code, glyph in codes.items() if glyph in meanings}


def recover_embedded_font_unicode(document) -> bool:
    """Prefer embedded semantics, then preserve valid source-reader guesses."""
    changed = False
    fonts = {
        font[0]
        for page in document
        for font in page.get_fonts()
        if font[2] == "TrueType"
    }
    for xref in fonts:
        if (
            document.xref_get_key(xref, "ToUnicode")[0] != "null"
            or document.xref_get_key(xref, "Encoding")[0] != "null"
        ):
            continue
        kind, flags = document.xref_get_key(xref, "FontDescriptor/Flags")
        if kind != "int" or not int(flags) & 4 or int(flags) & 32:
            continue
        data = document.extract_font(xref)[3]
        if not data:
            continue
        try:
            with TTFont(io.BytesIO(data)) as font:
                mapping = _embedded_mapping(font) or _reader_mapping(
                    document, xref, font
                )
        except KeyError, ValueError, TTLibError, struct.error:
            continue
        if mapping:
            set_font_unicode(document, xref, mapping)
            changed = True
    return changed
