"""Verify the published snapshot without loading datasets or checkpoints."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
entries = (ROOT/'MANIFEST.sha256').read_text().splitlines()
for line in entries:
    expected, name = line.split('  ', 1)
    path = ROOT/name
    assert path.is_file() and not path.is_symlink(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
print(f'{len(entries)} files match MANIFEST.sha256')
