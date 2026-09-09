"""Bake the demo into a static site for GitHub Pages.

Pages runs no Python, and the segmentation model lives in the sibling birdtranscript
repo, so CI cannot do this: run it locally, commit site/, and let the workflow upload.

Each sample is transcribed through the same code path the server uses, and the result
is written where the client would otherwise POST. Upload stays server-only.
"""
import base64
import json
import shutil

import librosa

import app  # loads the segmentation model and the clustering artifact

OUT = app.BASE_DIR / "site"
N_SAMPLES = 3  # mirrors the items[:3] slice in GET /api/samples


def main() -> None:
    with app.SAMPLE_INDEX_PATH.open(encoding="utf-8") as fh:
        items = json.load(fh)[:N_SAMPLES]
    if not items:
        raise SystemExit("No samples in the index — nothing to build.")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "api" / "samples").mkdir(parents=True)
    (OUT / "spectrograms").mkdir()

    # Static assets. Project Pages are served from /<repo>/, so a leading slash
    # would escape the site root: every reference has to be relative.
    shutil.copytree(app.BASE_DIR / "static", OUT / "static")
    index = (app.BASE_DIR / "static" / "index.html").read_text()
    index = index.replace('href="/static/', 'href="static/').replace('src="/static/', 'src="static/')
    index = index.replace("<script src=", "<script>window.STATIC_BUILD = true;</script>\n<script src=", 1)
    (OUT / "index.html").write_text(index)
    (OUT / "static" / "index.html").unlink()  # only the root copy is served
    (OUT / ".nojekyll").touch()

    listing = []
    for item in items:
        audio_rel = item["audio"].replace("\\", "/")
        src = app.BASE_DIR / audio_rel
        if not src.is_file():
            raise SystemExit(f"Missing sample audio: {src}")

        audio, _ = librosa.load(src, sr=app.SR, mono=True, duration=app.MAX_AUDIO_S)
        payload = app.transcribe_audio(audio)
        payload["source"] = {
            "bird_name": item["bird_name"],
            "filename": item["filename"],
            "id": item["id"],
        }

        # A base64 data URI inside the JSON costs a third extra and defeats caching,
        # so the spectrogram is written as an ordinary file the browser can cache.
        png = base64.b64decode(payload["spectrogram"].pop("png"))
        (OUT / "spectrograms" / f"{item['id']}.png").write_bytes(png)
        payload["spectrogram"]["url"] = f"spectrograms/{item['id']}.png"

        (OUT / "api" / "samples" / f"{item['id']}.json").write_text(json.dumps(payload))

        dst = OUT / audio_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        listing.append({**item, "audio": audio_rel})
        print(f"  {item['id']:10} {len(payload['segments']):3} syllables")

    (OUT / "api" / "samples.json").write_text(json.dumps(listing))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nsite/ built: {len(listing)} samples, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
