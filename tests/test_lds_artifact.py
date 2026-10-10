"""Independent scalar decoder tests for the persisted stream format (CPU only)."""
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def decode(path):
    blob = path.read_bytes()
    magic, version, p, rows, cols, tr, tc, lanes, tiles, raw, payload, reserved = struct.unpack_from("<8s8I3Q", blob)
    assert (magic, version, p, tr, tc, lanes, reserved) == (b"LDSANS1\0", 1, 12, 64, 128, 64, 0)
    freq = struct.unpack_from("<256I", blob, 64)
    lut, cumulative = [], 0
    for symbol, f in enumerate(freq):
        lut.extend([(symbol, f, cumulative)] * f)
        cumulative += f
    assert cumulative == 4096
    start = 1088 + tiles * 64 * 16
    assert len(blob) == start + payload
    output = bytearray(tiles * 8192)
    for stream in range(tiles * 64):
        offset, length, state = struct.unpack_from("<QII", blob, 1088 + stream * 16)
        pos = 0
        for i in range(128):
            symbol, f, cdf = lut[state & 4095]
            output[(stream // 64) * 8192 + i * 64 + stream % 64] = symbol
            state = f * (state >> 12) + (state & 4095) - cdf
            while state < (1 << 23):
                assert pos < length
                state = (state << 8) | blob[start + offset + pos]
                pos += 1
        assert pos == length and state == (1 << 23)
    unpadded = bytearray(raw)
    tile_columns = (cols + 127) // 128
    for t in range(tiles):
        for i in range(8192):
            r, c = (t // tile_columns) * 64 + i // 128, (t % tile_columns) * 128 + i % 128
            value = output[t * 8192 + i]
            if r < rows and c < cols:
                unpadded[r * cols + c] = value
            else:
                assert value == 0
    return bytes(unpadded), freq


class ArtifactTests(unittest.TestCase):
    def pack(self, data, rows, cols):
        with tempfile.TemporaryDirectory(prefix="lds-test-") as tmp:
            base = Path(tmp)
            (base / "raw").write_bytes(data)
            stats = json.loads(subprocess.check_output([str(ROOT / "bin/lds_pack"), str(base / "raw"), str(rows), str(cols), str(base / "ans")]))
            recovered, freq = decode(base / "ans")
            self.assertEqual(recovered, data)
            self.assertEqual(stats["artifact_bytes"], (base / "ans").stat().st_size)
            self.assertTrue(stats["cpu_roundtrip"])
            return freq

    def test_constant_full_frequency_4096(self):
        freq = self.pack(bytes([147]) * 8192, 64, 128)
        self.assertEqual(freq[147], 4096)

    def test_all_symbols_multiple_tiles(self):
        self.pack(bytes(range(256)) * 128, 128, 256)

    def test_random_edge_padding(self):
        rng = random.Random(2026)
        self.pack(bytes(rng.randrange(256) for _ in range(67 * 139)), 67, 139)

    def test_skew_and_renormalization(self):
        rng = random.Random(0)
        self.pack(bytes(rng.choices(range(256), weights=[3000] + [1] * 255, k=8192)), 64, 128)

    def test_single_element(self):
        self.pack(b"\xff", 1, 1)

    def test_reject_reference_size(self):
        with tempfile.TemporaryDirectory(prefix="lds-test-") as tmp:
            base = Path(tmp)
            (base / "raw").write_bytes(b"wrong size")
            result = subprocess.run([str(ROOT / "bin/lds_pack"), str(base / "raw"), "64", "128", str(base / "ans")], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((base / "ans").exists())


if __name__ == "__main__":
    unittest.main()
