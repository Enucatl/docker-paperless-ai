"""Verify searchable archives use corrected page pixels and complete OCR text."""

import hashlib
import io
import subprocess

import pymupdf
import pytest
from PIL import Image

from paperless_ai.agents.archive import create_searchable_archive


@pytest.fixture
def archive_input(tmp_path):
    """Provide original pages and differently colored corrected page images."""
    source = tmp_path / "original.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=400)
        page.insert_text((20, 30), "Original stays untouched")
        document.new_page(width=300, height=400)
        document.save(source)
    images = []
    for color in ("#ffeecc", "#cceeff"):
        image = io.BytesIO()
        Image.new("RGB", (600, 800), color).save(image, format="PNG")
        images.append(image.getvalue())
    output = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "page_count": 2,
        "pages": [
            {
                "page_index": index,
                "prunedResult": {
                    "width": 600,
                    "height": 800,
                    "doc_preprocessor_res": {
                        "angle": 0,
                        "model_settings": {"use_doc_unwarping": True},
                    },
                    "parsing_res_list": [],
                },
            }
            for index in range(2)
        ],
    }
    return source, output, images, tmp_path / "archive.pdf"


def test_pdfa_preserves_corrected_pixels_unicode_and_box_position(archive_input):
    """Real Ghostscript conversion keeps invisible multilingual text and images."""
    source, output, images, destination = archive_input
    output["pages"][0]["prunedResult"]["parsing_res_list"] = [
        {
            "block_content": "Résumé Ελληνικά 日本語 发票",
            "block_bbox": [60, 80, 540, 180],
        },
        {
            "block_content": "<table><tr><td>Café</td><td>€10</td></tr></table>",
            "block_bbox": [60, 400, 540, 500],
        },
    ]
    original = source.read_bytes()
    create_searchable_archive(source, output, destination, corrected_images=images)
    assert source.read_bytes() == original
    with pymupdf.open(destination) as document:
        assert len(document) == 2
        text = document[0].get_text()
        for word in ("Résumé", "Ελληνικά", "日本語", "发票", "Café", "€10"):
            assert word in text
        assert "Original stays untouched" not in text
        assert not document[1].get_text().strip()
        boxes = document[0].search_for("Résumé")
        assert boxes and pymupdf.Rect(30, 40, 270, 90).contains(boxes[0])
        pixmap = document[0].get_pixmap()
        assert all(
            abs(a - b) <= 3 for a, b in zip(pixmap.pixel(60, 60), (255, 238, 204))
        )
        assert (
            document.xref_get_key(document.pdf_catalog(), "OutputIntents")[0] == "array"
        )


def test_quarter_turn_correction_swaps_page_size(archive_input):
    """The corrected image and text share upright page coordinates."""
    source, output, images, destination = archive_input
    output["pages"][0]["prunedResult"]["doc_preprocessor_res"]["angle"] = 90
    output["pages"][0]["prunedResult"]["width"] = 800
    output["pages"][0]["prunedResult"]["height"] = 600
    corrected = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(corrected, format="PNG")
    images[0] = corrected.getvalue()
    output["pages"][0]["prunedResult"]["parsing_res_list"] = [
        {"block_content": "Upright", "block_bbox": [80, 60, 720, 180]}
    ]
    create_searchable_archive(source, output, destination, corrected_images=images)
    with pymupdf.open(destination) as document:
        assert document[0].rect == pymupdf.Rect(0, 0, 400, 300)
        assert "Upright" in document[0].get_text()


@pytest.mark.parametrize(
    "bbox",
    [[-1, 0, 20, 20], [0, 0, 601, 20], [20, 0, 10, 20], [0, 0, float("nan"), 20], None],
)
def test_invalid_coordinates_do_not_publish_archive(archive_input, bbox):
    """Reject malformed coordinates before replacing any existing archive."""
    source, output, images, destination = archive_input
    output["pages"][0]["prunedResult"]["parsing_res_list"] = [
        {"block_content": "Text", "block_bbox": bbox}
    ]
    destination.write_bytes(b"existing archive")
    with pytest.raises(ValueError, match="bounding box"):
        create_searchable_archive(source, output, destination, corrected_images=images)
    assert destination.read_bytes() == b"existing archive"


@pytest.mark.parametrize(
    "failure", ["source", "coverage", "images", "dimensions", "order"]
)
def test_wrong_source_or_pages_cannot_be_archived(archive_input, failure):
    """Keep the original and previous archive when OCR no longer matches them."""
    source, output, images, destination = archive_input
    if failure == "source":
        output["source_sha256"] = "0" * 64
    elif failure == "coverage":
        output["pages"].pop()
    elif failure == "images":
        images.pop()
    elif failure == "dimensions":
        output["pages"][0]["prunedResult"]["width"] = 900
    else:
        output["pages"][0]["page_index"] = 1
    with pytest.raises(ValueError):
        create_searchable_archive(source, output, destination, corrected_images=images)
    assert not destination.exists()


def test_converter_failure_preserves_existing_archive(archive_input, monkeypatch):
    """Do not publish incomplete output if strict PDF/A conversion fails."""
    source, output, images, destination = archive_input
    destination.write_bytes(b"existing archive")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "gs")

    monkeypatch.setattr("paperless_ai.agents.archive.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        create_searchable_archive(source, output, destination, corrected_images=images)
    assert destination.read_bytes() == b"existing archive"
