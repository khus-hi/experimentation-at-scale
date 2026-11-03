"""Download the Upworthy Research Archive from its public OSF project.

Nothing is hard-coded except the OSF node id. The script walks the node's
osfstorage tree with the public OSF v2 API (following folders, pagination and
child components), prints every file it sees, and downloads the ones that
belong to a released part (exploratory / confirmatory / holdout) plus any
codebook, README or randomization-update file.

Downloads land in data/raw/ and are recorded in data/raw/manifest.json.
A file is skipped when its local sha256 already matches.

Usage:
    .venv/bin/python -m ingest.download            # list + download
    .venv/bin/python -m ingest.download --list-only
    .venv/bin/python -m ingest.download --dump-raw  # print raw API JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.osf.io/v2"
NODE_ID = "jd64p"
RAW_DIR = Path("data/raw")
MANIFEST = RAW_DIR / "manifest.json"
USER_AGENT = (
    "experimentation-at-scale-ingest/0.1 (+https://github.com/khus-hi/experimentation-at-scale)"
)

PARTS = ("exploratory", "confirmatory", "holdout")
DOC_PATTERN = re.compile(r"codebook|readme|data[ _-]?dictionary", re.I)
UPDATE_PATTERN = re.compile(r"randomi[sz]ation|imbalance|problem|update", re.I)
DATA_EXT = (".csv", ".csv.gz", ".zip", ".tsv")

Fetch = Callable[[str], dict]


class UnexpectedLayout(RuntimeError):
    """The OSF API answered with JSON this script does not understand."""

    def __init__(self, url: str, payload: object) -> None:
        self.url = url
        self.payload = payload
        super().__init__(f"Unexpected OSF API response from {url}")


@dataclass
class OsfFile:
    osf_id: str
    name: str
    path: str  # materialized path inside the node, e.g. /folder/file.csv
    node: str
    size: int | None
    modified: str | None
    download_url: str
    osf_sha256: str | None
    osf_md5: str | None

    @property
    def category(self) -> str | None:
        """Return the part name, 'doc', 'update' or None (not downloaded)."""
        lowered = f"{self.path} {self.name}".lower()
        name = self.name.lower()
        if name.endswith(DATA_EXT):
            for part in PARTS:
                if part in lowered:
                    return part
        if DOC_PATTERN.search(name):
            return "doc"
        if UPDATE_PATTERN.search(name):
            return "update"
        return None


# --------------------------------------------------------------------------- HTTP


def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.5,  # 0, 3, 6, 12, 24, 48 s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def json_fetcher(session: requests.Session, dump_raw: bool = False) -> Fetch:
    def fetch(url: str) -> dict:
        resp = session.get(url, timeout=60, headers={"Accept": "application/vnd.api+json"})
        resp.raise_for_status()
        payload = resp.json()
        if dump_raw:
            print(f"\n=== GET {url}\n{json.dumps(payload, indent=2)[:20000]}")
        return payload

    return fetch


# ------------------------------------------------------------------- tree walking


def iter_pages(fetch: Fetch, url: str) -> Iterator[dict]:
    """Yield every item of a paginated JSON:API listing."""
    while url:
        payload = fetch(url)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise UnexpectedLayout(url, payload)
        yield from payload["data"]
        url = (payload.get("links") or {}).get("next")


def walk_files(fetch: Fetch, node: str, url: str | None = None) -> Iterator[OsfFile]:
    url = url or f"{API}/nodes/{node}/files/osfstorage/"
    for item in iter_pages(fetch, url):
        attrs = item.get("attributes") or {}
        kind = attrs.get("kind")
        if kind == "folder":
            child = (
                ((item.get("relationships") or {}).get("files") or {})
                .get("links", {})
                .get("related", {})
            )
            child_url = child.get("href") if isinstance(child, dict) else child
            if not child_url:
                raise UnexpectedLayout(url, item)
            yield from walk_files(fetch, node, child_url)
        elif kind == "file":
            hashes = (attrs.get("extra") or {}).get("hashes") or {}
            links = item.get("links") or {}
            download = links.get("download")
            if not download:
                raise UnexpectedLayout(url, item)
            yield OsfFile(
                osf_id=item.get("id", ""),
                name=attrs.get("name", ""),
                path=attrs.get("materialized_path") or "/" + attrs.get("name", ""),
                node=node,
                size=attrs.get("size"),
                modified=attrs.get("date_modified") or attrs.get("modified_utc"),
                download_url=download,
                osf_sha256=hashes.get("sha256"),
                osf_md5=hashes.get("md5"),
            )
        else:
            raise UnexpectedLayout(url, item)


def walk_node(fetch: Fetch, node: str) -> Iterator[OsfFile]:
    """Files of the node and, recursively, of its child components."""
    yield from walk_files(fetch, node)
    for child in iter_pages(fetch, f"{API}/nodes/{node}/children/"):
        if child.get("id"):
            yield from walk_node(fetch, child["id"])


# ---------------------------------------------------------------------- download


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"source": f"https://osf.io/{NODE_ID}/", "files": {}}


def local_name(f: OsfFile) -> str:
    """Flatten the OSF path into one file name.

    Part CSVs keep their plain name. Docs and other files get their folder as a
    prefix, because several folders hold a README.md that would overwrite each other.
    """
    parts = [p for p in f.path.strip("/").split("/")[:-1] if p]
    name = f.name
    if f.category not in PARTS and parts:
        name = "__".join([*parts, name])
    return name if f.node == NODE_ID else f"{f.node}__{name}"


def download(session: requests.Session, f: OsfFile, manifest: dict) -> str:
    dest = RAW_DIR / local_name(f)
    entry = manifest["files"].get(dest.name)
    expected = f.osf_sha256 or (entry or {}).get("sha256")
    if dest.exists() and expected and sha256_of(dest) == expected:
        return "skip (sha256 match)"

    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    nbytes = 0
    with session.get(f.download_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                h.update(chunk)
                nbytes += len(chunk)
    digest = h.hexdigest()
    if f.osf_sha256 and digest != f.osf_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {f.name}: got {digest}, OSF says {f.osf_sha256}")
    tmp.replace(dest)

    manifest["files"][dest.name] = {
        "url": f.download_url,
        "osf_id": f.osf_id,
        "osf_node": f.node,
        "osf_path": f.path,
        "category": f.category,
        "bytes": nbytes,
        "sha256": digest,
        "osf_modified": f.modified,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if dest.suffix == ".zip":
        manifest["files"][dest.name]["extracted"] = extract_zip(dest)
    return f"downloaded {nbytes:,} bytes"


def extract_zip(path: Path) -> list[str]:
    """Extract CSVs from an archive into data/raw/ (flat, no path traversal)."""
    out = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name or name.startswith("."):
                continue
            target = RAW_DIR / name
            with zf.open(info) as src, target.open("wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
            out.append(name)
    return out


# --------------------------------------------------------------------------- main


def fmt_size(n: int | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return str(n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--list-only", action="store_true", help="list files, download nothing")
    ap.add_argument("--dump-raw", action="store_true", help="print raw OSF API JSON responses")
    args = ap.parse_args(argv)

    session = make_session()
    fetch = json_fetcher(session, dump_raw=args.dump_raw)
    try:
        files = list(walk_node(fetch, NODE_ID))
    except requests.RequestException as exc:
        print(f"could not reach the OSF API after retries: {exc}", file=sys.stderr)
        return 3
    except UnexpectedLayout as exc:
        print(f"OSF API layout not understood at {exc.url}. Raw response:", file=sys.stderr)
        print(json.dumps(exc.payload, indent=2)[:20000], file=sys.stderr)
        return 2

    print(f"OSF node {NODE_ID}: {len(files)} files")
    print(f"{'category':<12} {'size':>10}  {'modified':<20}  path")
    for f in sorted(files, key=lambda x: (x.node, x.path)):
        node = "" if f.node == NODE_ID else f"[{f.node}]"
        mod = (f.modified or "?")[:19]
        print(f"{f.category or '-':<12} {fmt_size(f.size):>10}  {mod:<20}  {node}{f.path}")

    wanted = [f for f in files if f.category]
    found_parts = {f.category for f in wanted if f.category in PARTS}
    for part in PARTS:
        if part not in found_parts:
            print(f"note: no data file found for part '{part}' (not released?)")
    if args.list_only:
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    for f in wanted:
        status = download(session, f, manifest)
        print(f"{local_name(f)}: {status}")
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {MANIFEST} ({len(manifest['files'])} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
