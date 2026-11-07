"""Download only pinned source/Retail files, verifying each against its upstream Git blob hash.

No provider access is needed. The external dependency is never copied into the casepilot package.
"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPOSITORY = "sierra-research/tau2-bench"
REVISION = "033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f"
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT.parent / ".dependencies" / f"tau2-{REVISION[:12]}"


def git_blob_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def download_file(destination: Path, relative: str, expected: str) -> None:
    target = (destination / relative).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"Source manifest path escapes destination: {relative}")
    if target.is_file() and git_blob_hash(target.read_bytes()) == expected:
        return
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{relative}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = response.read()
            if git_blob_hash(data) != expected:
                raise ValueError(f"Pinned source integrity mismatch: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(0.5 * 2**attempt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    files = json.loads((PROJECT / "configs/benchmark-files.json").read_text(encoding="utf-8"))
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".casepilot-source.json"
    marker.unlink(missing_ok=True)
    # Create parents before threaded writes to avoid Windows realpath races.
    for relative in files:
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Source manifest path escapes destination: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(download_file, destination, name, sha) for name, sha in files.items()
        ]
        for count, result in enumerate(as_completed(futures), 1):
            result.result()
            if count % 50 == 0:
                print(f"Verified {count}/{len(files)} files", flush=True)
    marker.write_text(json.dumps({"revision": REVISION, "files": len(files)}), encoding="utf-8")
    print(f"Pinned Retail source ready: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
