"""CPU contract tests; numerical correctness is tested inside the GPU allocation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ArtifactTests(unittest.TestCase):
    def test_cli_rejects_wrong_codec(self):
        with tempfile.TemporaryDirectory(prefix='hipans-contract-') as tmp:
            path = Path(tmp) / 'artifact.json'
            path.write_text(json.dumps({'schema': 'lds-baseline-summary-v1'}))
            p = subprocess.run([sys.executable, str(ROOT / 'scripts/run_lds_bench.py'),
                                '--manifest', str(path)], text=True, capture_output=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('not a native hipANS artifact', p.stderr)

    def test_sha(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from prepare_lds_artifact import sha
        with tempfile.TemporaryDirectory(prefix='hipans-sha-') as tmp:
            path = Path(tmp) / 'bytes'
            path.write_bytes(b'abc')
            self.assertEqual(sha(path), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_actual_upstream_template_used(self):
        code = (ROOT / 'bench/hipans_lds_bench.cpp').read_text()
        self.assertIn('#include "hipans/ans/GpuANSDecode.h"', code)
        self.assertIn('ha::ANSDecodeWarpFullBlock<LDSWriter,10,ha::kDefaultBlockSize,false>::decode', code)
        self.assertIn('ha::ansEncodeBatchPointer(', code)
        self.assertNotIn('lds_format.h', code)

    def test_common_interval_joins_workers(self):
        code = (ROOT / 'bench/hipans_lds_bench.cpp').read_text()
        self.assertIn('hipStreamWaitEvent(stream,begin,0)', code)
        self.assertIn('hipStreamWaitEvent(control,finished[s],0)', code)

if __name__ == '__main__':
    unittest.main()
