"""Verify archives retain native PDF content and publish only valid conversions."""

import hashlib
import io
import subprocess
from xml.etree import ElementTree

import pymupdf
import pytest
from PIL import Image

from paperless_ai.agents.archive import _validate_pdfa_2u, create_searchable_archive


@pytest.fixture
def archive_input(tmp_path, monkeypatch):
    """Provide a mixed document with native text, vectors and a high resolution scan."""
    monkeypatch.setattr(
        "paperless_ai.agents.archive._validate_pdfa_2u", lambda source: None
    )
    source = tmp_path / "original.pdf"
    image = io.BytesIO()
    Image.new("RGB", (1500, 2000), "#ffeecc").save(image, format="PNG")
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        page.insert_text((20, 30), "Original stays untouched")
        page.draw_rect(pymupdf.Rect(20, 40, 280, 380))
        page.insert_image(pymupdf.Rect(20, 50, 50, 90), stream=image.getvalue())
        page = document.new_page(width=300, height=400)
        page.insert_image(page.rect, stream=image.getvalue())
        document.save(source)
    output = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "page_count": 2,
        "pages": [
            {
                "page_index": index,
                "prunedResult": {
                    "width": 600,
                    "height": 800,
                    "doc_preprocessor_res": {"angle": 0},
                    "parsing_res_list": [
                        {
                            "block_content": "Recognized scan",
                            "block_bbox": [40, 80, 560, 180],
                        }
                    ],
                },
            }
            for index in range(2)
        ],
    }
    return source, output, tmp_path / "archive.pdf"


def test_pdfa_preserves_native_text_vectors_and_image_resolution(archive_input):
    """Real conversion retains mixed pages and adds OCR only to the scan."""
    source, output, destination = archive_input
    original = source.read_bytes()
    create_searchable_archive(source, output, destination)
    assert source.read_bytes() == original
    with pymupdf.open(destination) as document:
        assert len(document) == 2
        assert document[0].get_text().strip() == "Original stays untouched"
        assert document[1].get_text().strip() == "Recognized scan"
        assert document[0].get_drawings()
        for page in document:
            assert page.rect == pymupdf.Rect(0, 0, 300, 400)
            assert [(entry[2], entry[3]) for entry in page.get_images()] == [
                (1500, 2000)
            ]
        assert document[1].get_pixmap().pixel(60, 60) == (255, 238, 204)
        assert (
            document.xref_get_key(document.pdf_catalog(), "OutputIntents")[0] == "array"
        )


@pytest.mark.parametrize("failure", ["source", "coverage", "count", "order"])
def test_wrong_source_or_pages_cannot_be_archived(archive_input, failure):
    """Keep the original and previous archive when OCR no longer matches them."""
    source, output, destination = archive_input
    destination.write_bytes(b"existing archive")
    if failure == "source":
        output["source_sha256"] = "0" * 64
    elif failure == "coverage":
        output["pages"].pop()
    elif failure == "count":
        output["page_count"] = 3
    else:
        output["pages"][0]["page_index"] = 1
    with pytest.raises(ValueError):
        create_searchable_archive(source, output, destination)
    assert destination.read_bytes() == b"existing archive"


def test_original_cannot_be_destination(archive_input):
    """Never overwrite the original with its converted archive."""
    source, output, _ = archive_input
    original = source.read_bytes()
    with pytest.raises(ValueError, match="replace the original"):
        create_searchable_archive(source, output, source)
    assert source.read_bytes() == original


def test_converter_failure_preserves_existing_archive(archive_input, monkeypatch):
    """Do not publish incomplete output if strict PDF/A conversion fails."""
    source, output, destination = archive_input
    destination.write_bytes(b"existing archive")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "gs")

    monkeypatch.setattr("paperless_ai.agents.archive.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        create_searchable_archive(source, output, destination)
    assert destination.read_bytes() == b"existing archive"


def test_invalid_conversion_preserves_existing_archive(archive_input, monkeypatch):
    """Reject a valid PDF that is missing the required archive metadata."""
    source, output, destination = archive_input
    destination.write_bytes(b"existing archive")

    def copy_original(source, destination, temporary):
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr("paperless_ai.agents.archive._convert_to_pdfa", copy_original)
    with pytest.raises(ValueError, match="complete PDF/A-2"):
        create_searchable_archive(source, output, destination)
    assert destination.read_bytes() == b"existing archive"


def test_scan_unicode_overlay_is_invisible(archive_input, monkeypatch):
    """Searchable multilingual text does not alter the visible scan pixels."""
    monkeypatch.setattr(
        "paperless_ai.agents.archive._validate_pdfa_2u", _validate_pdfa_2u
    )
    source, output, destination = archive_input
    output["pages"][1]["prunedResult"]["parsing_res_list"] = [
        {
            "block_content": "Résumé Ελληνικά 日本語 发票",
            "block_bbox": [40, 80, 560, 180],
        },
        {
            "block_content": "<table><tr><td>Café</td><td>€10</td></tr></table>",
            "block_bbox": [40, 200, 560, 300],
        },
    ]
    create_searchable_archive(source, output, destination)
    with pymupdf.open(source) as original, pymupdf.open(destination) as archive:
        assert original[1].get_pixmap().samples == archive[1].get_pixmap().samples
        text = archive[1].get_text()
        for word in ("Résumé", "Ελληνικά", "日本語", "发票", "Café", "€10"):
            assert word in text
        assert original[0].get_text() == archive[0].get_text()
        metadata = ElementTree.fromstring(archive.get_xml_metadata())
        conformance = "{http://www.aiim.org/pdfa/ns/id/}conformance"
        assert any(node.get(conformance) == "U" for node in metadata.iter())


@pytest.mark.parametrize(
    "angle, rotation",
    [(0, 0), (90, 0), (180, 0), (270, 0), (90, 90), (90, 180), (270, 270)],
)
def test_ocr_rotation_maps_back_to_original(archive_input, angle, rotation):
    """Corrected OCR boxes return to the original image and PDF rotation."""
    source, output, destination = archive_input
    with pymupdf.open(source) as document:
        document[1].set_rotation(rotation)
        document.saveIncr()
    output["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result = output["pages"][1]["prunedResult"]
    result["doc_preprocessor_res"]["angle"] = angle
    width, height = (600, 800) if (rotation + angle) % 180 == 0 else (800, 600)
    result.update(width=width, height=height)
    result["parsing_res_list"] = [
        {
            "block_content": "Recognized",
            "block_bbox": [width * 0.1, height * 0.2, width * 0.4, height * 0.3],
        }
    ]
    # These are the inverse counterclockwise rotations in normalized image space.
    expected = {
        0: (0.1, 0.2, 0.4, 0.3),
        90: (0.7, 0.1, 0.8, 0.4),
        180: (0.6, 0.7, 0.9, 0.8),
        270: (0.2, 0.6, 0.3, 0.9),
    }[angle]
    create_searchable_archive(source, output, destination)
    with pymupdf.open(source) as original, pymupdf.open(destination) as archive:
        page = archive[1]
        expected_rect = pymupdf.Rect(expected) * pymupdf.Matrix(
            page.rect.width, page.rect.height
        )
        boxes = page.search_for("Recognized")
        assert len(boxes) == 1
        displayed_box = boxes[0] * page.rotation_matrix
        assert (expected_rect + (-0.01, -0.01, 0.01, 0.01)).contains(displayed_box)
        assert original[1].get_pixmap().samples == page.get_pixmap().samples


@pytest.mark.parametrize(
    "failure", ["bbox", "dimensions", "angle", "unwarp", "blocks", "text"]
)
def test_invalid_overlay_preserves_previous_archive(archive_input, failure):
    """Invalid OCR geometry or text cannot publish a partial archive."""
    source, output, destination = archive_input
    destination.write_bytes(b"existing archive")
    result = output["pages"][1]["prunedResult"]
    if failure == "bbox":
        result["parsing_res_list"][0]["block_bbox"] = [0, 0, float("nan"), 20]
    elif failure == "dimensions":
        result["width"] = 0
    elif failure == "angle":
        result["doc_preprocessor_res"]["angle"] = 45
    elif failure == "unwarp":
        result["doc_preprocessor_res"]["model_settings"] = {"use_doc_unwarping": True}
    elif failure == "blocks":
        result["parsing_res_list"] = None
    else:
        result["parsing_res_list"][0]["block_content"] = 123
    with pytest.raises((TypeError, ValueError)):
        create_searchable_archive(source, output, destination)
    assert destination.read_bytes() == b"existing archive"


def test_existing_invisible_text_is_not_duplicated(archive_input):
    """An existing scanned-page OCR layer takes precedence over fresh OCR."""
    source, output, destination = archive_input
    with pymupdf.open(source) as document:
        document[1].insert_text((20, 30), "Existing OCR", render_mode=3)
        document.saveIncr()
    output["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    create_searchable_archive(source, output, destination)
    with pymupdf.open(destination) as document:
        assert document[1].get_text().strip() == "Existing OCR"


def test_pdfa_retains_text_with_overprint_mode(tmp_path):
    """Omit forbidden overprint settings without discarding subsequent page content."""
    source = tmp_path / "overprint.pdf"
    destination = tmp_path / "archive.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((20, 30), "Keep this native text")
        resources = int(document.xref_get_key(page.xref, "Resources")[1].split()[0])
        document.xref_set_key(
            resources, "ExtGState/Overprint", "<< /OPM 1 /op true /OP true >>"
        )
        stream = page.get_contents()[0]
        document.update_stream(
            stream, b"/Overprint gs\n" + document.xref_stream(stream)
        )
        document.save(source)
    output = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "page_count": 1,
        "pages": [{"page_index": 0}],
    }
    create_searchable_archive(source, output, destination)
    with pymupdf.open(source) as original, pymupdf.open(destination) as archive:
        assert archive[0].get_text() == original[0].get_text()
        assert archive[0].get_pixmap().samples == original[0].get_pixmap().samples


def test_pdf_errors_do_not_publish_partial_conversion(tmp_path):
    """A malformed content operator must fail rather than publish a partial page."""
    source = tmp_path / "malformed.pdf"
    destination = tmp_path / "archive.pdf"
    destination.write_bytes(b"existing archive")
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((20, 30), "Keep this native text")
        stream = page.get_contents()[0]
        document.update_stream(stream, b"null cm\n" + document.xref_stream(stream))
        document.save(source)
    output = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "page_count": 1,
        "pages": [{"page_index": 0}],
    }
    with pytest.raises(subprocess.CalledProcessError):
        create_searchable_archive(source, output, destination)
    assert destination.read_bytes() == b"existing archive"


def test_declared_winansi_survives_pdfa_conversion(tmp_path):
    """Explicit encoding restores punctuation without changing visible glyphs."""
    source = tmp_path / "winansi.pdf"
    destination = tmp_path / "archive.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        font = page.insert_font(fontname="helv")
        document.xref_set_key(font, "Encoding", "/WinAnsiEncoding")
        page.insert_text((20, 30), "placeholder")
        document.update_stream(
            page.get_contents()[0], b"BT /helv 12 Tf 20 30 Td <959680a0ad> Tj ET"
        )
        document.save(source)
    original_bytes = source.read_bytes()
    output = {
        "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "page_count": 1,
        "pages": [{"page_index": 0}],
    }
    create_searchable_archive(source, output, destination)
    assert source.read_bytes() == original_bytes
    with pymupdf.open(source) as original, pymupdf.open(destination) as archive:
        assert archive[0].get_text().strip() == "•–€ -"
        assert archive[0].get_pixmap().samples == original[0].get_pixmap().samples


@pytest.mark.parametrize(
    "encoding",
    [
        None,
        "/MacRomanEncoding",
        "<< /BaseEncoding /WinAnsiEncoding /Differences [149 /A] >>",
    ],
)
def test_unknown_or_custom_encoding_is_not_inferred(encoding):
    """Leave missing and custom font encodings to the existing PDF reader."""
    from paperless_ai.agents.archive import _add_known_font_unicode

    with pymupdf.open() as document:
        page = document.new_page()
        font = page.insert_font(fontname="helv")
        document.xref_set_key(font, "Encoding", encoding or "null")
        assert not _add_known_font_unicode(document)
        assert document.xref_get_key(font, "ToUnicode")[0] == "null"


def test_existing_unicode_map_is_preserved():
    """Do not override a font's explicit character semantics."""
    from paperless_ai.agents.archive import _add_known_font_unicode

    with pymupdf.open() as document:
        page = document.new_page()
        font = page.insert_font(fontname="helv")
        document.xref_set_key(font, "Encoding", "/WinAnsiEncoding")
        mapping = document.get_new_xref()
        document.update_object(mapping, "<<>>")
        document.update_stream(mapping, b"existing map")
        document.xref_set_key(font, "ToUnicode", f"{mapping} 0 R")
        assert not _add_known_font_unicode(document)
        assert document.xref_stream(mapping) == b"existing map"


@pytest.mark.parametrize(
    "report",
    [
        b'<report><validationReport isCompliant="false"/></report>',
        b"<report/>",
        b"<report><validationReport/></report>",
        b'<report><validationReport isCompliant="true"/><validationReport isCompliant="true"/></report>',
        b"not XML",
    ],
)
def test_invalid_unicode_report_preserves_previous_archive(
    archive_input, monkeypatch, report
):
    """Reject incomplete or negative validation even when the CLI exits successfully."""
    source, output, destination = archive_input
    destination.write_bytes(b"existing archive")
    original_run = subprocess.run

    def run(command, **kwargs):
        if command[0] == "verapdf":
            assert command[1:3] == ["--flavour", "2u"]
            return subprocess.CompletedProcess(command, 0, stdout=report)
        return original_run(command, **kwargs)

    monkeypatch.setattr(
        "paperless_ai.agents.archive._validate_pdfa_2u", _validate_pdfa_2u
    )
    monkeypatch.setattr("paperless_ai.agents.archive.subprocess.run", run)
    with pytest.raises((ValueError, ElementTree.ParseError)):
        create_searchable_archive(source, output, destination)
    assert destination.read_bytes() == b"existing archive"
