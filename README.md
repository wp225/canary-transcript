<p align="center">
  <img src="static/Canary_blink.gif" alt="" height="150" />
</p>

<h1 align="center">Canary Transcript</h1>

<p align="center">
  Upload a canary recording and watch its syllables unfold in time.
</p>

---

Birdsong is made of short, repeated syllables. This demo segments a recording
into those syllables, gives each one a readable label, and plots them together
so the shape of a song is visible at a glance.

- **Segmentation** — a conv-RNN marks syllable boundaries on the spectrogram
- **Transcription** — each syllable gets a label like `díiiin` or `tib`
- **Manifold** — every syllable becomes a 26-D feature vector, projected onto
  the corpus's first three principal components and linked in song order

Click a syllable to hear it, drag to scroll, `Ctrl`/`⌘` + scroll to zoom.

## Running it

The model and the clustering artifact are stored with [Git LFS](https://git-lfs.com),
so run `git lfs install` before cloning this repository (or `git lfs pull` after).
Then, from the repository root:

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m uvicorn app:app --reload
```

Then open http://127.0.0.1:8000.

## Layout

```
app.py             FastAPI server: segmentation, transcription, PCA, spectrograms
build_site.py      bakes the static GitHub Pages copy into site/
models/            segmentation model, feature extractor, clustering artifact (Git LFS)
static/            the whole front end — one HTML file, one CSS file, one JS file
site/              built Pages artifact (committed; see below)
tests/             unittest suite for the segmentation front end
demo_samples/      source recordings, kept local (large media, not in git)
```

## GitHub Pages

Pages serves static files only, so the deployed copy is baked ahead of time:
the bundled samples are transcribed into `site/api/`, spectrograms are written
to `site/spectrograms/`, and every path is relative because project Pages are
served from `/<repo>/`. **Upload is disabled there** — it needs the model, so it
works only against a local server.

`site/` is committed rather than built in CI, because the recordings it bakes
live in `demo_samples/`, which is kept out of git as large local media. After
changing `static/`, `app.py`, `models/`, or the samples, rebuild and commit:

```bash
./.venv/bin/python build_site.py
```

Pushing to `main` deploys `site/` via `.github/workflows/deploy-pages.yml`, which
fails the build if `site/static/` has drifted from `static/`, so a forgotten
rebuild is caught rather than silently shipping a stale page.

To preview exactly what Pages will serve, subpath included:

```bash
mkdir -p /tmp/pages && ln -sfn "$PWD/site" /tmp/pages/canary-transcript
cd /tmp/pages && python -m http.server 8093
# open http://127.0.0.1:8093/canary-transcript/
```

Serve it over HTTP rather than opening `site/index.html` directly — browsers
block `fetch()` on `file://` URLs, so the baked data never loads.

## Tests

```bash
./.venv/bin/python -m unittest discover -s tests
```
