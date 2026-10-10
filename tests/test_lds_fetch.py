import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "scripts/fetch_lds_weights.py"
spec = importlib.util.spec_from_file_location("fetch", path)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class Response:
    def __init__(self, status, content_range, chunks):
        self.status_code, self.headers, self.chunks = status, {"Content-Range": content_range}, chunks
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def raise_for_status(self): pass
    def iter_content(self, size): yield from self.chunks


class RangeTests(unittest.TestCase):
    def call(self, status, content_range, chunks):
        with patch.object(fetch.requests, "get", return_value=Response(status, content_range, chunks)):
            return fetch.fetch_range("https://example.invalid/file", 10, 12)
    def test_exact_range(self):
        self.assertEqual(self.call(206, "bytes 10-12/100", [b"ab", b"c"]), b"abc")
    def test_refuse_full_shard(self):
        with self.assertRaises(RuntimeError): self.call(200, "", [])
    def test_wrong_range(self):
        with self.assertRaises(RuntimeError): self.call(206, "bytes 0-2/100", [b"abc"])
    def test_truncated(self):
        with self.assertRaises(RuntimeError): self.call(206, "bytes 10-12/100", [b"ab"])
    def test_oversize(self):
        with self.assertRaises(RuntimeError): self.call(206, "bytes 10-12/100", [b"abcd"])


if __name__ == "__main__":
    unittest.main()
