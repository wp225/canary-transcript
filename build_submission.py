"""Package the full app as an anonymized zip for double-blind submission.

The online copy only replays pre-baked samples; this zip is the whole thing,
segmentation and transcription included, so a reviewer can run it on their own
recordings. Every file is scanned before it goes in, and the build fails on an
identifying string or on a Git LFS pointer standing in for a real model file.

    ./.venv/bin/python build_submission.py   ->   dist/canary-transcript.zip
"""
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dist" / "canary-transcript.zip"
TOP = "canary-transcript"  # folder the zip extracts to
N_SAMPLES = 3  # GET /api/samples serves the first three

# Identifying terms (names, logins, machine and project names), one per line. The
# file is kept out of git: this repository is what the anonymous mirror copies, so
# the list itself must not be in it. It doubles as the mirror's word list.
TERMS_FILE = ROOT / ".anon-terms"

FILES = [
    "app.py",
    "requirements.txt",
    "static/index.html",
    "static/demo.css",
    "static/demo.js",
    "static/Canary_blink.gif",
    "models/conv_rnn.py",
    "models/extract_features_temporal.py",
    "models/pooled_all.pt",
    "models/matches_temporal.pkl",
    "tests/test_segmentation.py",
]

README = """\
<p align="center"><img src="static/Canary_blink.gif" alt="" height="150" /></p>

# Canary Transcript

Upload a canary recording and watch its syllables unfold in time.

- **Segmentation**: a conv-RNN marks syllable boundaries on the spectrogram
- **Transcription**: each syllable gets a readable label such as `díiiin` or `tib`
- **Manifold**: every syllable becomes a 26-D feature vector, projected onto the
  corpus's first three principal components and linked in song order

## Running it

Tested with Python 3.13. The server needs about 2 GB of free memory.

```bash
python3 -m venv .venv
# Optional, Linux: CPU-only PyTorch, which skips ~3 GB of CUDA libraries this app never uses
.venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app
```

On Windows, use `.venv\\Scripts\\pip` and `.venv\\Scripts\\python` instead.

Then open http://127.0.0.1:8000. Three sample recordings load on the page; use
**Upload your recording** to run the pipeline on your own audio (WAV, MP3, FLAC
or OGG; the first 30 seconds are analysed). Click a syllable to hear it, drag
to scroll, and `Ctrl`/`⌘` + scroll to zoom.

## Contents

```
app.py          FastAPI server: segmentation, transcription, PCA, spectrograms
static/         the front end: one HTML, one CSS and one JS file
models/         segmentation model, feature extractor, clustering artifact
demo_samples/   the three sample recordings shown on the page
tests/          unittest suite
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```
"""


def identifying_pattern() -> re.Pattern:
    if not TERMS_FILE.exists():
        raise SystemExit(f"Missing {TERMS_FILE.name}: list the identifying terms, one per line.")
    terms = [t.strip() for t in TERMS_FILE.read_text().splitlines()
             if t.strip() and not t.lstrip().startswith("#")]
    # Absolute home directories identify whoever built the files, on any OS.
    alternatives = [re.escape(t.encode()) for t in terms] + [rb"/home/", rb"/Users/", rb"C:\\Users"]
    return re.compile(b"|".join(alternatives), re.I)


def check(name: str, data: bytes, identifying: re.Pattern) -> None:
    hit = identifying.search(data)
    if hit:
        raise SystemExit(f"Refusing to package {name}: contains {hit.group().decode(errors='replace')!r}")
    if data.startswith(b"version https://git-lfs"):
        raise SystemExit(f"Refusing to package {name}: it is a Git LFS pointer; run `git lfs pull`.")


def main() -> None:
    entries = {path: (ROOT / path).read_bytes() for path in FILES}

    with (ROOT / "demo_samples" / "all_samples.json").open(encoding="utf-8") as fh:
        samples = json.load(fh)[:N_SAMPLES]
    # Only the fields the server and the page read.
    samples = [{k: s[k] for k in ("id", "bird_name", "filename", "audio")} for s in samples]
    entries["demo_samples/all_samples.json"] = json.dumps(samples, indent=2).encode()
    for s in samples:
        entries[s["audio"]] = (ROOT / s["audio"]).read_bytes()

    entries["README.md"] = README.encode()

    identifying = identifying_pattern()
    for name, data in entries.items():
        check(name, data, identifying)

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(entries):
            info = zipfile.ZipInfo(f"{TOP}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, entries[name])

    raw = sum(len(d) for d in entries.values())
    print(f"{OUT.relative_to(ROOT)}: {len(entries)} files, "
          f"{raw / 1e6:.1f} MB unpacked, {OUT.stat().st_size / 1e6:.1f} MB zipped")


if __name__ == "__main__":
    main()
