"""Create archival PDFs while preserving original page content."""

import hashlib
import html
import math
import os
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from paperless_ai.agents.font_mapping import (
    recover_embedded_font_unicode,
    set_font_unicode,
)


class _BlockText(HTMLParser):
    """Keep table cell and paragraph boundaries when reading Paddle HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"}:
            self.parts.append(" ")
        elif tag in {"p", "div", "tr"}:
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self.parts.append("\n")


def _block_text(content: str) -> str:
    """Convert table HTML to text without interpreting ordinary angle brackets."""
    if "<table" not in content.lower():
        return content.strip()
    parser = _BlockText()
    parser.feed(content)
    return "".join(parser.parts).strip()


def create_searchable_archive(source: Path, output: dict, destination: Path) -> None:
    """Convert the original PDF to validated PDF/A-2u without rebuilding pages.

    Args:
        source: Original PDF whose SHA-256 is recorded in the OCR output.
        output: Validated Paddle output identifying the source and its pages.
        destination: Archive file to publish only after successful conversion.

    Raises:
        ValueError: Source identity, page coverage or converted output is invalid.
        subprocess.CalledProcessError: Ghostscript cannot produce compliant PDF/A.
    """
    import pymupdf

    source_bytes = source.read_bytes()
    if output.get("source_sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("OCR output does not identify this original PDF")
    if source.resolve() == destination.resolve():
        raise ValueError("The archive must not replace the original PDF")

    with pymupdf.open(stream=source_bytes, filetype="pdf") as original:
        pages = output.get("pages")
        if (
            not isinstance(pages, list)
            or len(pages) != len(original)
            or output.get("page_count") != len(original)
        ):
            raise ValueError("OCR archive page coverage does not match the original")
        if any(record.get("page_index") != index for index, record in enumerate(pages)):
            raise ValueError("OCR archive pages must be in source order")
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            temporary = Path(temporary)
            original_copy = temporary / "original.pdf"
            changed = _add_known_font_unicode(original)
            changed = recover_embedded_font_unicode(original) or changed
            if changed:
                original.save(original_copy)
            else:
                original_copy.write_bytes(source_bytes)
            converted = temporary / "archive.pdf"
            _convert_to_pdfa(original_copy, converted, temporary)
            with pymupdf.open(converted) as checked:
                if len(checked) != len(original) or "pdfaid:part='2'" not in (
                    checked.get_xml_metadata().replace('"', "'")
                ):
                    raise ValueError(
                        "Archive conversion did not produce a complete PDF/A-2"
                    )
                # Ghostscript can corrupt ToUnicode maps; add embedded OCR fonts afterward.
                for index, page in enumerate(checked):
                    if original[index].get_text().strip():
                        continue
                    _add_ocr_layer(page, pages[index]["prunedResult"])
                # PDF/A requires the identity mapping MuPDF otherwise leaves implicit.
                for xref in range(1, checked.xref_length()):
                    if (
                        checked.xref_get_key(xref, "Subtype")
                        == ("name", "/CIDFontType2")
                        and checked.xref_get_key(xref, "CIDToGIDMap")[0] == "null"
                    ):
                        checked.xref_set_key(xref, "CIDToGIDMap", "/Identity")
                ElementTree.register_namespace(
                    "pdfaid", "http://www.aiim.org/pdfa/ns/id/"
                )
                metadata = ElementTree.fromstring(checked.get_xml_metadata())
                conformance = "{http://www.aiim.org/pdfa/ns/id/}conformance"
                for node in metadata.iter():
                    if conformance in node.attrib:
                        node.set(conformance, "U")
                    elif node.tag == conformance:
                        node.text = "U"
                checked.set_xml_metadata(
                    ElementTree.tostring(metadata, encoding="unicode")
                )
                finished = temporary / "finished.pdf"
                checked.save(finished, garbage=4, deflate=True)
            _validate_pdfa_2u(finished)
            os.replace(finished, destination)


def _add_known_font_unicode(document) -> bool:
    """Add missing Unicode maps only for explicitly declared WinAnsi fonts."""
    fonts = {
        font[0]
        for page in document
        for font in page.get_fonts()
        if font[2] in {"Type1", "TrueType"}
        and document.xref_get_key(font[0], "Encoding") == ("name", "/WinAnsiEncoding")
        and document.xref_get_key(font[0], "ToUnicode")[0] == "null"
    }
    if not fonts:
        return False
    # PDF WinAnsi differs from Windows-1252 at these glyph positions (PDF Annex D).
    overrides = {
        127: "•",
        129: "•",
        141: "•",
        143: "•",
        144: "•",
        157: "•",
        160: " ",
        173: "-",
    }
    mappings = {}
    for code in range(32, 256):
        mappings[code] = overrides.get(code) or bytes([code]).decode("cp1252")
    for font in fonts:
        set_font_unicode(document, font, mappings)
    return True


def _add_ocr_layer(page, result: dict) -> None:
    """Place an invisible OCR layer over the unchanged original page."""
    import pymupdf

    width, height = result.get("width"), result.get("height")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in (width, height)
    ):
        raise ValueError("Invalid OCR dimensions")
    preprocessing = result.get("doc_preprocessor_res", {})
    if preprocessing.get("model_settings", {}).get("use_doc_unwarping"):
        raise ValueError("Unwarped OCR cannot align with original page images")
    angle = preprocessing.get("angle", 0)
    if angle not in (0, 90, 180, 270, -1):
        raise ValueError("Invalid OCR page orientation")
    if angle == -1:
        angle = 0
    blocks = result.get("parsing_res_list")
    if not isinstance(blocks, list):
        raise TypeError("Missing OCR blocks")
    with pymupdf.open() as overlay:
        layer = overlay.new_page(width=width, height=height)
        for block in blocks:
            content = block.get("block_content", "")
            if not isinstance(content, str):
                raise TypeError("Invalid OCR block text")
            text = _block_text(content)
            if not text:
                continue
            bbox = block.get("block_bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in bbox
                )
                or not (
                    0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height
                )
            ):
                raise ValueError("Invalid OCR bounding box")
            # shortcut: block-level layout is approximate; use word boxes for precise selection.
            spare_height, _ = layer.insert_htmlbox(
                pymupdf.Rect(bbox),
                html.escape(text).replace("\n", "<br>"),
                css="* { margin: 0; padding: 0; } body { font-size: 24pt; line-height: 1.8; }",
                opacity=0,
                scale_low=0,
            )
            if spare_height < 0:
                raise ValueError("OCR block text cannot fit on the page")
        if layer.get_contents():
            overlay.subset_fonts()
            page.show_pdf_page(
                page.rect * page.derotation_matrix,
                overlay,
                rotate=(page.rotation - angle) % 360,
                keep_proportion=False,
            )


def _validate_pdfa_2u(source: Path) -> None:
    """Publish only PDFs that pass veraPDF's Unicode conformance profile."""
    result = subprocess.run(
        ["verapdf", "--flavour", "2u", "--format", "xml", str(source)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    reports = ElementTree.fromstring(result.stdout).findall(".//validationReport")
    if len(reports) != 1 or reports[0].get("isCompliant") != "true":
        raise ValueError("Archive failed PDF/A-2u validation")


def _convert_to_pdfa(source: Path, destination: Path, temporary: Path) -> None:
    """Convert to PDF/A, omitting incompatible settings and rejecting PDF errors."""
    profile = Path("/usr/share/color/icc/ghostscript/srgb.icc")
    definition = temporary / "pdfa.ps"
    definition.write_text(
        "/ICCProfile (" + str(profile) + ") def\n"
        "[/_objdef {icc_PDFA} /type /stream /OBJ pdfmark\n"
        "[{icc_PDFA} << /N 3 >> /PUT pdfmark\n"
        "[{icc_PDFA} ICCProfile (r) file /PUT pdfmark\n"
        "[/_objdef {OutputIntent_PDFA} /type /dict /OBJ pdfmark\n"
        "[{OutputIntent_PDFA} << /Type /OutputIntent /S /GTS_PDFA1 "
        "/DestOutputProfile {icc_PDFA} /OutputConditionIdentifier (sRGB) >> /PUT pdfmark\n"
        "[{Catalog} << /OutputIntents [{OutputIntent_PDFA}] >> /PUT pdfmark\n"
    )
    subprocess.run(
        [
            "gs",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-dPDFA=2",
            "-dPDFACompatibilityPolicy=1",
            "-dPDFSTOPONERROR",
            "-sDEVICE=pdfwrite",
            "-sColorConversionStrategy=RGB",
            "-dAutoRotatePages=/None",
            "-dDownsampleColorImages=false",
            "-dDownsampleGrayImages=false",
            "-dDownsampleMonoImages=false",
            f"--permit-file-read={profile}",
            f"-sOutputFile={destination}",
            str(definition),
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )
