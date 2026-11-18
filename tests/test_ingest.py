import io, zipfile
from helpers import *

def mkzip(files):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        for n, t in files.items(): z.writestr(n, t if isinstance(t, bytes) else t.encode())
    return bio.getvalue()

def test_zip_with_junk_and_bad_file():
    good = lambda n: script(n, dialogue_rows(6), scene_end=20)
    data = mkzip({"show/episode-001-A.txt": good(1), "show/episode-002-B.txt": good(2), "__MACOSX/._x.txt": "junk",
                  "show/readme.md": "hi", "show/episode-003-C.txt": "", "show/episode-004-D.txt": "not a script"})
    eps, rep = load_episodes(data)
    assert [e.number for e in eps] == [1, 2]
    assert rep.episodes_parsed == 2 and any("empty" in s for s in rep.skipped) and any("episode-004" in e for e in rep.errors)

def test_invalid_zip_and_missing_path():
    eps, rep = load_episodes([b"not a zip", "/no/such/path"])
    assert eps == [] and len(rep.errors) == 2

def test_duplicate_episode_number_kept_once():
    t = script(1, dialogue_rows(6), scene_end=20)
    eps, rep = load_episodes([("episode-001-a.txt", t.encode()), ("episode-001-b.txt", t.encode())])
    assert len(eps) == 1 and any("duplicate" in s for s in rep.skipped)

def test_single_files_and_latin1():
    t = script(7, dialogue_rows(6), scene_end=20).replace("test logline", "caf\xe9 logline")
    eps, _ = load_episodes([("ep7.txt", t.replace("\u2013","-").encode("latin-1"))])
    assert eps and eps[0].number == 7 and "caf" in eps[0].logline

def test_episode_number_from_filename_when_header_missing():
    t = script(1, dialogue_rows(6), scene_end=20).replace("EPISODE NUMBER: Episode 1", "")
    eps, _ = load_episodes([("episode-009-x.txt", t.encode())])
    assert eps[0].number == 9 and any("EPISODE NUMBER" in w for w in eps[0].warnings)
