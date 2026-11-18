"""Read a ZIP, a directory, or individual files into parsed episodes. Zips are read in memory (no extraction)."""
from __future__ import annotations
import io, re, zipfile
from pathlib import Path
from .models import Episode, IngestReport
from .parser import parse_episode

MAX_FILE_BYTES = 5_000_000


def _decode(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("latin-1")


def _num_from_name(name: str) -> int | None:
    m = re.search(r"episode[-_ ]*0*(\d+)", Path(name).name, re.I)
    return int(m.group(1)) if m else None


def _collect(inputs) -> tuple[list[tuple[str, bytes]], IngestReport]:
    rep = IngestReport()
    items: list[tuple[str, bytes]] = []

    def add_zip(data: bytes, label: str):
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            rep.errors.append(f"{label}: not a valid ZIP"); return
        for info in zf.infolist():
            nm = info.filename
            base = Path(nm).name
            if info.is_dir() or "__MACOSX" in nm or base.startswith("."):
                continue
            if not base.lower().endswith(".txt"):
                rep.skipped.append(f"{nm}: not a .txt file"); continue
            if info.file_size > MAX_FILE_BYTES:
                rep.skipped.append(f"{nm}: larger than {MAX_FILE_BYTES} bytes"); continue
            items.append((base, zf.read(info)))

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    for inp in inputs:
        if isinstance(inp, (bytes, bytearray)):
            add_zip(bytes(inp), "<bytes>"); continue
        if isinstance(inp, tuple) and len(inp) == 2:     # (filename, bytes) from an upload widget
            name, data = inp
            if name.lower().endswith(".zip"):
                add_zip(data, name)
            else:
                items.append((Path(name).name, data))
            continue
        p = Path(inp)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    if f.suffix.lower() == ".txt":
                        items.append((f.name, f.read_bytes()))
                    elif f.suffix.lower() == ".zip":
                        add_zip(f.read_bytes(), f.name)
        elif p.is_file():
            if p.suffix.lower() == ".zip":
                add_zip(p.read_bytes(), p.name)
            else:
                items.append((p.name, p.read_bytes()))
        else:
            rep.errors.append(f"{inp}: path not found")
    return items, rep


def load_episodes(inputs, aliases=None, hidden=None) -> tuple[list[Episode], IngestReport]:
    items, rep = _collect(inputs)
    rep.files_seen = len(items)
    episodes: dict[int, Episode] = {}
    for name, data in sorted(items, key=lambda x: (_num_from_name(x[0]) or 10**6, x[0])):
        if not data.strip():
            rep.skipped.append(f"{name}: empty file"); continue
        try:
            ep = parse_episode(_decode(data), name, _num_from_name(name), aliases, hidden)
        except Exception as e:                          # parser should not raise; belt and braces
            rep.errors.append(f"{name}: parser crashed: {e!r}"); continue
        if ep.number == 0 or not ep.blocks:
            rep.errors.append(f"{name}: no usable episode content (number={ep.number}, blocks={len(ep.blocks)})")
            continue
        if ep.number in episodes:
            rep.skipped.append(f"{name}: duplicate episode number {ep.number}; kept {episodes[ep.number].filename}")
            continue
        episodes[ep.number] = ep
    rep.episodes_parsed = len(episodes)
    return [episodes[k] for k in sorted(episodes)], rep
