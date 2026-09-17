"""Verify the committed site/ still matches the sample set app.py serves.

CI cannot rebuild site/ -- that needs the segmentation model and the local-only
recordings in demo_samples/ -- so it cannot diff the baked payloads against freshly
computed ones. What it can do is catch the drift that actually happens: someone edits
DEMO_SAMPLE_IDS, or adds a sample, and forgets to re-run build_site.py. Every id, and
every file the page will fetch for it, is checked to exist.

app.py is read with ast rather than imported, so this stays runnable in CI without
torch, librosa, or the vendored model.

    python check_site.py        # exits non-zero on drift
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


def declared_sample_ids() -> list[str]:
    tree = ast.parse((ROOT / "app.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEMO_SAMPLE_IDS" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise SystemExit("app.py no longer defines DEMO_SAMPLE_IDS")


def main() -> None:
    expected = declared_sample_ids()
    listing_path = SITE / "api" / "samples.json"
    if not listing_path.is_file():
        raise SystemExit(f"missing {listing_path.relative_to(ROOT)} -- run build_site.py")

    listing = json.loads(listing_path.read_text())
    got = [s["id"] for s in listing]
    if got != expected:
        raise SystemExit(
            f"site/ is stale: app.py serves {expected}, site/api/samples.json has {got}."
            "\nRun `python build_site.py` and commit the result."
        )

    missing = []
    for sample in listing:
        if not sample.get("display_name"):
            missing.append(f"{sample['id']}: no display_name in the baked listing")
        payload_path = SITE / "api" / "samples" / f"{sample['id']}.json"
        if not payload_path.is_file():
            missing.append(str(payload_path.relative_to(ROOT)))
            continue
        payload = json.loads(payload_path.read_text())
        for rel in (sample["audio"].replace("\\", "/"), payload["spectrogram"].get("url")):
            if rel and not (SITE / rel).is_file():
                missing.append(f"{sample['id']} -> site/{rel}")

    if missing:
        raise SystemExit("site/ is incomplete:\n  " + "\n  ".join(missing))

    print(f"site/ matches app.py: {len(listing)} samples, every payload and asset present")


if __name__ == "__main__":
    main()
