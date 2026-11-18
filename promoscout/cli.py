from __future__ import annotations
import argparse, os, sys
from pathlib import Path


def _load_dotenv():
    p = Path(".env")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv=None):
    _load_dotenv()
    ap = argparse.ArgumentParser(prog="promoscout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a ZIP / folder / files")
    r.add_argument("inputs", nargs="+"); r.add_argument("--out", default="outputs/run")
    r.add_argument("--llm", default="none", choices=["none", "gemini", "mock"])
    r.add_argument("--config", default=None); r.add_argument("--title", default="Untitled microdrama")
    sub.add_parser("llm-check", help="verify the Gemini key/model with one tiny call")
    a = ap.parse_args(argv)
    from .config import load_config
    cfg = load_config(a.config if a.cmd == "run" else None)
    if a.cmd == "llm-check":
        from .llm import make_client
        c = make_client("gemini", cfg, cache_dir=None)
        print("model:", c.model_id())
        print(c.generate_json("Reply with JSON only.", 'Return {"ok": true}', tag="check"))
        return 0
    from .llm import make_client
    from .pipeline import run_pipeline
    from .export import write_outputs
    llm = make_client(a.llm, cfg)
    res = run_pipeline(a.inputs, cfg, llm, a.title, progress=lambda i, n: print(f"  judged {i}/{n}", file=sys.stderr))
    paths = write_outputs(res, a.out)
    print(f"mode={res.info.get('mode')} episodes={len(res.episodes)} candidates={len(res.candidates)} approved={len(res.approved)}")
    for n in res.notes[:10]: print("  note:", n)
    for k, v in paths.items(): print("  wrote", v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
