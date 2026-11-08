"""Real HTTP archive persistence against the disposable Paperless test service."""

from hashlib import sha256

import niquests
import pytest

from tests.conftest import _make_test_pdf, _upload_document


@pytest.mark.live
async def test_external_archive_roundtrip_preserves_original(paperless_client) -> None:
    """Attach PDF/text together and reject stale output without changing either."""
    client = paperless_client
    original = _make_test_pdf()
    archive = _make_test_pdf()
    document_id = await _upload_document(client, original)
    try:
        downloaded_original = await client.download_original(document_id)
        assert downloaded_original == original
        await client.upload_archive(
            document_id,
            archive,
            "External searchable transcript",
            sha256(original).hexdigest(),
        )
        document = await client.get_document_with_content(document_id)
        assert document["content"] == "External searchable transcript"
        response = await client._client.get(f"/api/documents/{document_id}/download/")
        response.raise_for_status()
        assert response.content == archive
        assert await client.download_original(document_id) == original
        with pytest.raises(niquests.HTTPError) as error:
            await client.upload_archive(
                document_id, original, "Stale transcript", "0" * 64
            )
        assert error.value.response.status_code == 409
        document = await client.get_document_with_content(document_id)
        assert document["content"] == "External searchable transcript"
        response = await client._client.get(f"/api/documents/{document_id}/download/")
        response.raise_for_status()
        assert response.content == archive
        assert await client.download_original(document_id) == original
    finally:
        response = await client._client.delete(f"/api/documents/{document_id}/")
        response.raise_for_status()
