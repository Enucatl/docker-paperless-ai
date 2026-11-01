"""Build a searchable archival PDF from the pages recognized by external OCR."""

import hashlib
import html
import math
import os
import subprocess
import tempfile
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path


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


def create_searchable_archive(
    source: Path,
    output: dict,
    destination: Path,
    *,
    corrected_images: Sequence[bytes],
) -> None:
    """Create PDF/A-2b with a searchable text layer aligned to recognized pages.

    Args:
        source: Original PDF whose SHA-256 is recorded in the OCR output.
        output: Validated Paddle output containing page dimensions and block boxes.
        destination: Archive file to publish only after successful conversion.
        corrected_images: Exact corrected page images from the OCR pipeline, in
            page order, including any rotation and unwarping applied before OCR.

    Raises:
        ValueError: Source identity, geometry, page coverage or output is invalid.
        TypeError: OCR blocks or their text have invalid types.
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
            or len(corrected_images) != len(original)
        ):
            raise ValueError("OCR archive page coverage does not match the original")
        with pymupdf.open() as archive:
            text_layers = []
            for index, record in enumerate(pages):
                text_blocks = []
                text_layers.append(text_blocks)
                if record.get("page_index") != index:
                    raise ValueError("OCR archive pages must be in source order")
                result = record["prunedResult"]
                dimensions = (result.get("width"), result.get("height"))
                if any(
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                    for value in dimensions
                ):
                    raise ValueError(f"Invalid OCR dimensions on page {index}")
                width, height = dimensions
                preprocessing = result.get("doc_preprocessor_res", {})
                angle = preprocessing.get("angle", 0)
                if angle not in (0, 90, 180, 270, -1):
                    raise ValueError(f"Invalid OCR page orientation on page {index}")
                original_page = original[index]
                image = pymupdf.Pixmap(corrected_images[index])
                if (image.width, image.height) != dimensions:
                    raise ValueError(
                        f"Corrected image dimensions differ on page {index}"
                    )
                page_width, page_height = (
                    original_page.rect.width,
                    original_page.rect.height,
                )
                if angle in (90, 270):
                    page_width, page_height = page_height, page_width
                page = archive.new_page(width=page_width, height=page_height)
                page.insert_image(
                    page.rect, stream=corrected_images[index], keep_proportion=False
                )

                blocks = result.get("parsing_res_list")
                if not isinstance(blocks, list):
                    raise TypeError(f"Missing OCR blocks on page {index}")
                for block in blocks:
                    content = block.get("block_content", "")
                    if not isinstance(content, str):
                        raise TypeError(f"Invalid OCR block text on page {index}")
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
                            0 <= bbox[0] < bbox[2] <= width
                            and 0 <= bbox[1] < bbox[3] <= height
                        )
                    ):
                        raise ValueError(f"Invalid OCR bounding box on page {index}")
                    rect = pymupdf.Rect(
                        bbox[0] * page.rect.width / width,
                        bbox[1] * page.rect.height / height,
                        bbox[2] * page.rect.width / width,
                        bbox[3] * page.rect.height / height,
                    )
                    text_blocks.append((rect, text))

            with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                temporary = Path(temporary)
                searchable = temporary / "searchable.pdf"
                converted = temporary / "archive.pdf"
                archive.save(searchable, garbage=4, deflate=True)
                _convert_to_pdfa(searchable, converted, temporary)
                with pymupdf.open(converted) as checked:
                    if len(checked) != len(
                        original
                    ) or "pdfaid:part='2'" not in checked.get_xml_metadata().replace(
                        '"', "'"
                    ):
                        raise ValueError(
                            "Archive conversion did not produce a complete PDF/A-2"
                        )
                    # Ghostscript can corrupt ToUnicode maps; add embedded fonts afterward.
                    for page, text_blocks in zip(checked, text_layers):
                        for rect, text in text_blocks:
                            spare_height, _ = page.insert_htmlbox(
                                rect,
                                html.escape(text).replace("\n", "<br>"),
                                css="* { margin: 0; padding: 0; } body { font-size: 12pt; line-height: 1.8; }",
                                opacity=0,
                                scale_low=0,
                            )
                            if spare_height < 0:
                                raise ValueError(
                                    "OCR block text cannot fit on the page"
                                )
                    checked.subset_fonts()
                    # PDF/A requires the identity mapping MuPDF otherwise leaves implicit.
                    for xref in range(1, checked.xref_length()):
                        if (
                            checked.xref_get_key(xref, "Subtype")
                            == ("name", "/CIDFontType2")
                            and checked.xref_get_key(xref, "CIDToGIDMap")[0] == "null"
                        ):
                            checked.xref_set_key(xref, "CIDToGIDMap", "/Identity")
                    finished = temporary / "finished.pdf"
                    checked.save(finished, garbage=4, deflate=True)
                os.replace(finished, destination)


def _convert_to_pdfa(source: Path, destination: Path, temporary: Path) -> None:
    """Convert without silently retaining features forbidden by PDF/A."""
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
            "-dPDFACompatibilityPolicy=2",
            "-sDEVICE=pdfwrite",
            "-sColorConversionStrategy=RGB",
            "-dAutoRotatePages=/None",
            f"--permit-file-read={profile}",
            f"-sOutputFile={destination}",
            str(definition),
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=300,
    )
