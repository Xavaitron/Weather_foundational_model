"""Read public GCS Zarr objects over HTTPS, without credentials or bucket listing."""
from collections.abc import MutableMapping
import http.client
import time
import urllib.error
import urllib.parse
import urllib.request


class PublicGCSStore(MutableMapping):
    def __init__(self, url):
        if not url.startswith("gs://"):
            raise ValueError("Expected a gs:// URL")
        self.base = "https://storage.googleapis.com/" + url[5:].rstrip("/")

    def __getitem__(self, key):
        url = self.base + "/" + urllib.parse.quote(key, safe="/.")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    raise KeyError(key) from error
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                    raise
            except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead, ConnectionError):
                if attempt == 3:
                    raise
            time.sleep(2 ** attempt)

    def __setitem__(self, key, value):
        raise TypeError("Public source store is read-only")

    def __delitem__(self, key):
        raise TypeError("Public source store is read-only")

    def __iter__(self):
        raise NotImplementedError("Bucket listing is disabled; consolidated metadata is required")

    def __len__(self):
        raise NotImplementedError("Bucket listing is disabled; consolidated metadata is required")
