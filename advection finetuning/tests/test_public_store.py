import http.client
from unittest.mock import MagicMock, patch

import pytest

from public_store import PublicGCSStore


def response(value):
    result = MagicMock()
    if isinstance(value, Exception):
        result.__enter__.return_value.read.side_effect = value
    else:
        result.__enter__.return_value.read.return_value = value
    return result


def test_incomplete_chunk_is_retried_and_partial_bytes_discarded():
    with patch("public_store.urllib.request.urlopen", side_effect=[
        response(http.client.IncompleteRead(b"partial", 10)), response(b"complete")
    ]) as fetch, patch("public_store.time.sleep") as sleep:
        assert PublicGCSStore("gs://bucket/store")["field/0.0"] == b"complete"
        assert fetch.call_count == 2
        sleep.assert_called_once_with(1)


def test_repeated_incomplete_chunks_stop_after_bounded_retries():
    with patch("public_store.urllib.request.urlopen", side_effect=lambda *a, **kw:
        response(http.client.IncompleteRead(b"partial", 10))
    ) as fetch, patch("public_store.time.sleep"):
        with pytest.raises(http.client.IncompleteRead):
            PublicGCSStore("gs://bucket/store")["field/0.0"]
        assert fetch.call_count == 4
