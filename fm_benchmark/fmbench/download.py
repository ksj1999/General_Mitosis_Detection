"""Resumable HTTP downloads with checksum verification and archive extraction (stdlib only)."""
from __future__ import annotations

import gzip
import hashlib
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .tasks.base import DownloadSpec

USER_AGENT = "fmbench-downloader/0.1"
CHUNK = 1 << 20


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n and int(n) > 0 else None
    except Exception:  # noqa: BLE001 - some servers (figshare/S3) refuse HEAD; the GET still works
        return None


def _total_from_response(r, have: int) -> int | None:
    m = re.match(r"bytes \d+-\d+/(\d+)", r.headers.get("Content-Range", "") or "")
    if m:
        return int(m.group(1))
    n = r.headers.get("Content-Length")
    if n and r.status == 200:
        return int(n)
    if n and r.status == 206:
        return have + int(n)
    return None


def file_hash(path: Path, algo: str = "md5") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK * 8), b""):
            h.update(block)
    return h.hexdigest()


def md5sum(path: Path) -> str:
    return file_hash(path, "md5")


def verify(path: Path, spec: DownloadSpec, log=print) -> None:
    for algo, want in (("md5", spec.md5), ("sha256", spec.sha256)):
        if want:
            log(f"  {spec.filename}: checking {algo}")
            got = file_hash(path, algo)
            if got.lower() != want.lower():
                raise RuntimeError(f"{spec.filename}: {algo} {got} != expected {want}; delete {path} and retry")


def fetch(spec: DownloadSpec, dest_dir: Path, retries: int = 8, log=print) -> Path:
    """Download ``spec.url`` to ``dest_dir/spec.filename``; resume a partial file if possible.

    A file already at the destination (for example copied by hand) is verified and accepted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / spec.filename
    done_marker = out.with_name(out.name + ".ok")
    if out.exists() and done_marker.exists():
        log(f"  {spec.filename}: already downloaded")
        return out
    if out.exists():
        verify(out, spec, log)
        done_marker.write_text("local file\n")
        return out
    part = out.with_name(out.name + ".part")
    total = _remote_size(spec.url)
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:
            break
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(spec.url, headers=headers), timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                total = total or _total_from_response(r, have)
                t0, got, last = time.time(), have, time.time()
                with part.open(mode) as f:
                    while True:
                        block = r.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        got += len(block)
                        if time.time() - last > 30:
                            pct = f" {100 * got / total:.1f}%" if total else ""
                            log(f"  {spec.filename}: {got / 1e9:.2f} GB{pct} ({(got - have) / 1e6 / (time.time() - t0):.1f} MB/s)")
                            last = time.time()
            if total is None or part.stat().st_size >= total:
                break
            log(f"  {spec.filename}: stream ended at {part.stat().st_size} of {total} bytes; resuming")
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:
                # Range not satisfiable: the .part already holds the whole file (figshare/S3 refuse HEAD,
                # so ``total`` was unknown). Accept it when a checksum can confirm it; otherwise restart.
                m = re.match(r"bytes \*/(\d+)", (e.headers.get("Content-Range", "") if e.headers else "") or "")
                if (m and int(m.group(1)) == have) or (not m and (spec.md5 or spec.sha256)):
                    break
                part.unlink(missing_ok=True)
                log(f"  {spec.filename}: server refused to resume at {have} bytes; restarting")
                continue
            log(f"  {spec.filename}: attempt {attempt + 1} failed ({e}); retrying")
            time.sleep(min(60, 2 ** attempt))
        except Exception as e:  # noqa: BLE001 - network errors: back off and resume
            log(f"  {spec.filename}: attempt {attempt + 1} failed ({e}); retrying")
            time.sleep(min(60, 2 ** attempt))
    else:
        raise RuntimeError(f"could not download {spec.url}")
    verify(part, spec, log)
    part.replace(out)
    done_marker.write_text(spec.url + "\n")
    log(f"  {spec.filename}: done ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def _wanted(name: str, members: tuple[str, ...]) -> bool:
    name = name[2:] if name.startswith("./") else name
    return not members or any(name.startswith(m) for m in members)


def extract(archive: Path, dest_dir: Path, members: tuple[str, ...] = (), log=print) -> Path:
    """Extract ``archive`` into ``dest_dir`` (only members starting with one of ``members``, if given)."""
    marker = dest_dir / f".extracted-{archive.name}"
    if marker.exists():
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    log(f"  extracting {archive.name} -> {dest_dir}")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            names = [m for m in z.namelist() if _wanted(m, members)]
            for m in names:  # refuse path traversal
                if Path(m).is_absolute() or ".." in Path(m).parts:
                    raise RuntimeError(f"unsafe path in {archive}: {m}")
            z.extractall(dest_dir, members=names)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            sel = [m for m in t.getmembers() if _wanted(m.name, members)]
            if hasattr(tarfile, "data_filter"):  # Python 3.12+, 3.11.4+, 3.10.12+
                t.extractall(dest_dir, members=sel, filter="data")
            else:
                root = dest_dir.resolve()
                for m in sel:
                    mp = Path(m.name)
                    if mp.is_absolute() or ".." in mp.parts or not (m.isfile() or m.isdir()):
                        raise RuntimeError(f"unsafe or unsupported member in {archive}: {m.name}")
                    if not (dest_dir / m.name).resolve().is_relative_to(root):
                        raise RuntimeError(f"unsafe path in {archive}: {m.name}")
                t.extractall(dest_dir, members=sel)
    elif name.endswith(".gz"):
        with gzip.open(archive, "rb") as src, (dest_dir / archive.name[:-3]).open("wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK * 8)
    else:
        raise ValueError(f"do not know how to extract {archive}")
    marker.write_text("ok\n")
    return dest_dir
