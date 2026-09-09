# canary-transcript

Upload a canary recording and see the syllables unfold in time: segmentation,
per-syllable transcription, and a 3-D PCA view of the syllable manifold.

## Run locally (full demo, including upload)

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m uvicorn app:app --reload
```

The segmentation model and the clustering artifact live in the sibling
`birdtranscript` research repo, not here. Point at a checkout elsewhere with:

```bash
BIRDTRANSCRIPT_ROOT=/path/to/birdtranscript ./.venv/bin/python -m uvicorn app:app
```

## GitHub Pages

Pages serves static files only, so the deployed site is a baked copy:

- the three bundled samples are transcribed ahead of time into `site/api/`
- spectrograms are written to `site/spectrograms/` instead of inlined as base64
- **upload is disabled** on Pages — it needs the model, so it only works locally

`site/` is committed rather than built in CI, because CI has no access to the
model. After changing anything in `static/`, `app.py`, or the samples, rebuild
and commit the result:

```bash
./.venv/bin/python build_site.py
```

Rebuilding needs the model *and* the source recordings in `demo_samples/`, which
are kept out of git as large local media — so only a machine that has both can
regenerate `site/`.

Pushing to `main` then deploys `site/` via `.github/workflows/deploy-pages.yml`.
The workflow fails the build if `site/static/` has drifted from `static/`, so a
forgotten rebuild is caught rather than silently shipping a stale page.

To preview exactly what Pages will serve, including the `/<repo>/` subpath:

```bash
mkdir -p /tmp/pages && ln -sfn "$PWD/site" /tmp/pages/canary-transcript
cd /tmp/pages && python -m http.server 8093
# open http://127.0.0.1:8093/canary-transcript/
```
