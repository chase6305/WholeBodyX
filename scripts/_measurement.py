"""Shared RSS sampling and source receipts for standalone measurement tools."""

import hashlib
from pathlib import Path

SOURCE = Path(__file__).resolve()


def rss_kib(pid="self"):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def source_hashes(root, paths):
    root = Path(root)
    return {str(path.relative_to(root) if path.is_relative_to(root) else path):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in (SOURCE, *paths)}


def sources_unchanged(root, hashes):
    root = Path(root)
    return all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items())
