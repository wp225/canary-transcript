import base64
import io
import os
import pickle
import sys
import tempfile
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from scipy import signal
from sklearn.decomposition import PCA
from starlette.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
# Models and the clustering artifact live in the sibling research repo. Override with
# BIRDTRANSCRIPT_ROOT to run this demo from a checkout somewhere else.
BIRDTRANSCRIPT_ROOT = Path(
    os.environ.get("BIRDTRANSCRIPT_ROOT", "/supernova/data/home/george/codeBase/birdtranscript")
)
ARTIFACT_PATH = BIRDTRANSCRIPT_ROOT / "notebooks/bpa_pipeline_final/results/bpa_debug/matches_temporal.pkl"
EXTRACTOR_PATH = BIRDTRANSCRIPT_ROOT / "scripts/extract_features_temporal.py"
SEGMENT_MODEL_PATH = BIRDTRANSCRIPT_ROOT / "saved_models/pooled/pooled_all.pt"
SEGMENT_MODEL_MODULE_PATH = BIRDTRANSCRIPT_ROOT / "models/conv_rnn.py"
INDEX_HTML = BASE_DIR / "static" / "index.html"
SAMPLE_INDEX_PATH = BASE_DIR / "demo_samples" / "all_samples.json"

# Preprocessing constants, fixed by how pooled_all.pt was trained
# (birdtranscript/dataset.py CanariesSegmentationDataset).
SR = 44100
N_FFT = 512
HOP = 64
TOP_DB = 80.0
HIGHPASS_HZ = 500.0
HIGHPASS_ORDER = 40
FREQ_BINS = N_FFT // 2  # DC bin is dropped, leaving 256 rows for the conv stack

SEG_THRESHOLD = 0.5
WINDOW_FRAMES = int(5.0 * SR / HOP)  # 5 s, the training sample duration
OVERLAP_FRAMES = int(0.5 * SR / HOP)
# Real syllables run shorter than the 15 ms MIN_DUR_BIRD_MS gate that the corpus
# feature extraction used; 5 ms is the floor below which a run is not a syllable.
MIN_SYLLABLE_MS = 5.0
FEATURE_PAD_S = 0.01  # the corpus features were cut with 10 ms of context on each side
MAX_AUDIO_S = 30.0

# Display spectrogram. Rendered without axes so that client-side x == time exactly;
# the frequency axis is sent as tick fractions and drawn in an HTML gutter instead.
DISPLAY_N_MELS = 160
DISPLAY_FMIN = 300.0
DISPLAY_FMAX = 16000.0
DISPLAY_HOP = 128
DISPLAY_HEIGHT_PX = 340
DISPLAY_PX_PER_S = 200  # native resolution; the client may scale beyond this
DISPLAY_MAX_WIDTH_PX = 12000
FREQ_TICKS_HZ = (500, 1000, 2000, 4000, 8000, 16000)
# Cream paper -> warm mid -> ink, so quiet background matches the page ground.
# Stops are weighted toward the dark end so mid energy is already legible ink.
SPEC_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "sonoma",
    [(0.0, "#f2ede4"), (0.15, "#e0d6c4"), (0.35, "#a8907a"),
     (0.6, "#68503f"), (0.8, "#33282a"), (1.0, "#14141c")],
)

# PCA of the 26-d syllable features. Fitted once on the pipeline's own corpus so the
# axes mean the same thing for every upload, rather than being refit per recording.
PCA_FIT_SAMPLE = 50000
PCA_REFERENCE_POINTS = 1200

sys.path.insert(0, str(BIRDTRANSCRIPT_ROOT))


def load_external_module(module_name: str, module_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extractor = load_external_module("birdtranscript_feature_extractor", EXTRACTOR_PATH)
segmentor_model = None


def load_segmentation_model() -> torch.nn.Module:
    global segmentor_model
    if segmentor_model is None:
        segmentor_module = load_external_module("birdtranscript_conv_rnn", SEGMENT_MODEL_MODULE_PATH)
        model = segmentor_module.ConvRNNSegmentor(p_dropout=0.2)
        model.load_state_dict(torch.load(SEGMENT_MODEL_PATH, map_location="cpu"), strict=True)
        model.eval()
        segmentor_model = model
    return segmentor_model


app = FastAPI(title="Bird Transcript Demo", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/demo_samples", StaticFiles(directory=str(BASE_DIR / "demo_samples")), name="demo_samples")


@lru_cache(maxsize=1)
def load_pipeline() -> Dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Missing pipeline artifact: {ARTIFACT_PATH}")

    with ARTIFACT_PATH.open("rb") as fh:
        payload = pickle.load(fh)

    bpa = dict(payload["bpa_config"])
    # transcribe.py derives the vowel thresholds from cv_pool but never saves them.
    v_durs = [r["duration_ms"] for r in payload["cv_pool"] if r["unit_type"] in ("V", "VV")]
    bpa["dur_thresholds_v"] = [float(np.percentile(v_durs, p)) for p in (20, 40, 60, 80)]
    payload["bpa_config"] = bpa
    return payload


@lru_cache(maxsize=1)
def load_projection() -> Dict[str, Any]:
    """PCA fitted on a sample of the corpus, plus a reference cloud to plot behind
    the uploaded syllables. Coordinates are scaled into roughly [-1, 1] so the
    client can draw without knowing anything about the feature space."""
    pipeline = load_pipeline()
    records = pipeline["matched_records"]

    rng = np.random.default_rng(0)
    size = min(PCA_FIT_SAMPLE, len(records))
    idx = rng.choice(len(records), size=size, replace=False)
    feats = pipeline["scaler"].transform(np.stack([records[i]["features"] for i in idx]))

    pca = PCA(n_components=3, random_state=0).fit(feats)
    reference = pca.transform(feats[:PCA_REFERENCE_POINTS])
    # Scale on a high percentile rather than the max: a handful of outliers would
    # otherwise squeeze the whole cloud into the middle of the canvas.
    scale = float(np.percentile(np.abs(reference), 98)) or 1.0

    # How strongly each of the 26 features drives the 3-D projection, for the
    # coefficient readout beside the plot.
    loadings = np.sqrt((pca.components_ ** 2).sum(axis=0))
    loadings = loadings / (loadings.max() or 1.0)

    return {
        "pca": pca,
        "scale": scale,
        "reference": np.round(reference / scale, 4).tolist(),
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "feature_names": list(pipeline["feature_names"]),
        "loadings": [round(float(v), 3) for v in loadings],
    }


# ── BPA labelling (mirrors scripts/transcribe.py) ─────────────────────────────
RISING_ACCENT = {"i": "í", "e": "é", "a": "á", "o": "ó", "u": "ú"}
FALLING_ACCENT = {"i": "ì", "e": "è", "a": "à", "o": "ò", "u": "ù"}
CONS_SIMPLIFY = {"hh": "h", "th": "t", "dh": "d"}
MAX_REPS = 5


def make_bpa_label(cv_label: str, dur_ms: float, f0_slope: float, cfg: Dict[str, Any]) -> str:
    vowel_nucleus = cfg["vowel_nucleus"]
    vowel_tail = cfg["vowel_tail"]
    vowels = set(vowel_nucleus)
    parts = cv_label.split("+")

    def accent(nuc: str) -> str:
        if f0_slope > cfg["slope_thresh"]:
            return RISING_ACCENT.get(nuc, nuc)
        if f0_slope < -cfg["slope_thresh"]:
            return FALLING_ACCENT.get(nuc, nuc)
        return nuc

    def reps(thresholds: List[float]) -> int:
        return min(1 + sum(dur_ms >= t for t in thresholds), MAX_REPS)

    if len(parts) == 1:
        vow = parts[0]
        nucleus = vowel_nucleus.get(vow, vow[0] if vow else "?")
        tail = vowel_tail.get(vow, "")
        n_reps = reps(cfg["dur_thresholds_v"])
        return accent(nucleus) + nucleus * (n_reps - 1) + tail

    if len(parts) == 2 and parts[0] in vowels:
        nuc1 = vowel_nucleus.get(parts[0], parts[0][0])
        nuc2 = vowel_nucleus.get(parts[1], parts[1][0])
        tail2 = vowel_tail.get(parts[1], "")
        n_reps = reps(cfg["dur_thresholds_v"])
        return accent(nuc1) + nuc2 * max(1, n_reps - 1) + tail2

    cons = CONS_SIMPLIFY.get(parts[0], parts[0])
    vow = parts[1]
    coda = CONS_SIMPLIFY.get(parts[2], parts[2]) if len(parts) == 3 else ""
    thresholds = cfg["dur_thresholds_cvc"] if len(parts) == 3 else cfg["dur_thresholds_cv"]
    nucleus = vowel_nucleus.get(vow, vow[0] if vow else "?")
    n_reps = reps(thresholds)
    return cons + accent(nucleus) + nucleus * (n_reps - 1) + coda


def simplify_short(label: str, dur_ms: float, cfg: Dict[str, Any]) -> str:
    if dur_ms >= cfg["short_ms"]:
        return label
    for asp in ("th", "dh", "ph", "kh"):
        label = label.replace(asp, asp[0])
    return label


# ── Segmentation ──────────────────────────────────────────────────────────────
def _highpass(audio: np.ndarray) -> np.ndarray:
    sos = signal.butter(
        HIGHPASS_ORDER, HIGHPASS_HZ, btype="highpass", analog=False, output="sos", fs=SR
    )
    return np.asarray(signal.sosfiltfilt(sos, audio), dtype=np.float32)


def segmentor_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Log-power spectrogram normalised exactly as the segmentor was trained: linear
    STFT, dB with an 80 dB floor, min-max scaled to [0, 1], DC bin dropped."""
    stft = librosa.stft(
        _highpass(audio), n_fft=N_FFT, hop_length=HOP, window="hann", center=True, pad_mode="reflect"
    )
    power = np.abs(stft) ** 2
    db = 10.0 * np.log10(np.maximum(power, 1e-10))
    db = np.maximum(db, db.max() - TOP_DB)
    spec = (db - db.min()) / max(db.max() - db.min(), 1e-9)
    spec = np.nan_to_num(spec[1:], nan=0.0, posinf=1.0, neginf=0.0)
    return spec.astype(np.float32)


def frame_probabilities(audio: np.ndarray) -> np.ndarray:
    """Per-frame P(syllable) from the ConvRNN segmentor, run over overlapping windows
    so that long uploads do not blow up the conv activations."""
    model = load_segmentation_model()
    spec = segmentor_spectrogram(audio)
    n_frames = spec.shape[1]

    if n_frames <= WINDOW_FRAMES:
        starts = [0]
    else:
        step = WINDOW_FRAMES - OVERLAP_FRAMES
        starts = list(range(0, n_frames - WINDOW_FRAMES + 1, step))
        if starts[-1] + WINDOW_FRAMES < n_frames:
            starts.append(n_frames - WINDOW_FRAMES)

    total = np.zeros(n_frames, dtype=np.float64)
    counts = np.zeros(n_frames, dtype=np.float64)
    with torch.no_grad():
        for start in starts:
            end = min(start + WINDOW_FRAMES, n_frames)
            window = torch.from_numpy(spec[:, start:end]).unsqueeze(0).unsqueeze(0)
            probs = torch.sigmoid(model(window)).squeeze(0).squeeze(-1).numpy()
            total[start:end] += probs
            counts[start:end] += 1.0

    return total / np.maximum(counts, 1.0)


def probabilities_to_spans(probs: np.ndarray) -> List[Tuple[float, float]]:
    """Contiguous runs above threshold → (onset_s, offset_s), as in predict_segments.py."""
    binary = (probs > SEG_THRESHOLD).astype(int)
    diff = np.diff(binary, prepend=0, append=0)
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]

    spans = []
    for on, off in zip(onsets, offsets):
        start_s = float(on * HOP / SR)
        end_s = float(off * HOP / SR)
        if (end_s - start_s) * 1000.0 >= MIN_SYLLABLE_MS:
            spans.append((start_s, end_s))
    return spans


def segment_audio(audio: np.ndarray) -> List[Tuple[float, float]]:
    if len(audio) < N_FFT * 4:
        return []
    return probabilities_to_spans(frame_probabilities(audio))


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe_audio(audio: np.ndarray) -> Dict[str, Any]:
    pipeline = load_pipeline()
    scaler = pipeline["scaler"]
    km_model = pipeline["km_model"]
    cluster_info = pipeline["cluster_info"]
    bpa_config = pipeline["bpa_config"]

    probs = frame_probabilities(audio) if len(audio) >= N_FFT * 4 else np.zeros(1)
    spans = probabilities_to_spans(probs)
    segments: List[Dict[str, Any]] = []
    scaled_features: List[np.ndarray] = []

    for idx, (start_s, end_s) in enumerate(spans):
        s = max(0, int((start_s - FEATURE_PAD_S) * SR))
        e = min(len(audio), int((end_s + FEATURE_PAD_S) * SR))
        clip = audio[s:e].astype(np.float32)

        vec, voiced_frac, f0_slope = extractor.extract_features_and_slope(clip, SR, "bird")
        feat_norm = scaler.transform(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        scaled_features.append(feat_norm[0])
        cluster_id = int(km_model.predict(feat_norm)[0])
        info = cluster_info.get(cluster_id, {})
        cv_label = info.get("cv_label")
        dur_ms = (end_s - start_s) * 1000.0

        if cv_label:
            label = simplify_short(make_bpa_label(cv_label, dur_ms, f0_slope, bpa_config), dur_ms, bpa_config)
        else:
            label = f"cluster_{cluster_id}"

        segments.append(
            {
                "index": idx + 1,
                "cluster_id": cluster_id,
                "cv_label": cv_label,
                "label": label,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_ms": round(dur_ms, 1),
                "voiced_frac": round(float(voiced_frac), 3),
                "f0_slope": round(float(f0_slope), 3),
            }
        )

    projection = load_projection()
    if scaled_features:
        coords = projection["pca"].transform(np.vstack(scaled_features)) / projection["scale"]
        for segment, point in zip(segments, np.round(coords, 4).tolist()):
            segment["pca"] = point

    return {
        "transcript": " ".join(item["label"] for item in segments),
        "segments": segments,
        "pca": {
            "reference": projection["reference"],
            "explained_variance": projection["explained_variance"],
            "feature_names": projection["feature_names"],
            "loadings": projection["loadings"],
        },
        "duration_s": round(len(audio) / SR, 3),
        "peak_confidence": round(float(probs.max()), 3),
        "threshold": SEG_THRESHOLD,
        "min_syllable_ms": MIN_SYLLABLE_MS,
        "spectrogram": build_spectrogram(audio, SR),
        "segmentation_model": "conv_rnn_pooled_all",
    }


def build_spectrogram(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Axis-free mel spectrogram spanning exactly [0, duration].

    Segment overlays and both axes are drawn by the client so they stay sharp and
    interactive at any zoom; the only thing the image has to guarantee is that its
    left edge is t=0 and its right edge is t=duration.
    """
    duration_s = len(audio) / sr
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=DISPLAY_HOP,
        n_mels=DISPLAY_N_MELS, fmin=DISPLAY_FMIN, fmax=DISPLAY_FMAX,
    )
    db = librosa.power_to_db(mel, ref=np.max)

    width_px = int(min(max(duration_s * DISPLAY_PX_PER_S, 640), DISPLAY_MAX_WIDTH_PX))
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, DISPLAY_HEIGHT_PX / dpi), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))  # fills the canvas: no margins, no ticks
    ax.set_axis_off()
    ax.imshow(
        db, origin="lower", aspect="auto", cmap=SPEC_CMAP,
        extent=(0.0, duration_s, 0.0, float(DISPLAY_N_MELS)),
        vmin=db.max() - TOP_DB, vmax=db.max(),
    )
    ax.set_xlim(0.0, duration_s)
    ax.set_ylim(0.0, float(DISPLAY_N_MELS))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0)
    plt.close(fig)

    # Mel bins are non-linear in Hz, so hand the client the fraction of image height
    # at which each labelled frequency sits.
    mel_hz = librosa.mel_frequencies(n_mels=DISPLAY_N_MELS, fmin=DISPLAY_FMIN, fmax=DISPLAY_FMAX)
    ticks = [
        {"hz": hz, "label": f"{hz // 1000}k" if hz >= 1000 else str(hz),
         "frac": float(np.argmin(np.abs(mel_hz - hz)) / DISPLAY_N_MELS)}
        for hz in FREQ_TICKS_HZ
        if DISPLAY_FMIN <= hz <= DISPLAY_FMAX
    ]

    return {
        "png": base64.b64encode(buf.getvalue()).decode("ascii"),
        "width_px": width_px,
        "height_px": DISPLAY_HEIGHT_PX,
        "native_px_per_s": round(width_px / duration_s, 2) if duration_s else DISPLAY_PX_PER_S,
        "freq_ticks": ticks,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/samples")
def list_samples() -> JSONResponse:
    with SAMPLE_INDEX_PATH.open("r", encoding="utf-8") as fh:
        items = json.load(fh)
    return JSONResponse(items[:3])


@app.post("/api/samples/{sample_id}")
async def load_sample(sample_id: str) -> JSONResponse:
    with SAMPLE_INDEX_PATH.open("r", encoding="utf-8") as fh:
        items = json.load(fh)
    item = next((entry for entry in items if entry["id"] == sample_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    audio_path = BASE_DIR / item["audio"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Sample audio not found")

    audio, _ = librosa.load(audio_path, sr=SR, mono=True, duration=MAX_AUDIO_S)
    payload = transcribe_audio(audio)
    payload["source"] = {
        "bird_name": item["bird_name"],
        "filename": item["filename"],
        "id": item["id"],
    }
    return JSONResponse(payload)


@app.post("/api/transcribe")
async def transcribe_upload(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please upload an audio file.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}:
        raise HTTPException(status_code=400, detail="Please upload a common audio format such as WAV, MP3, FLAC, or OGG.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio, _ = librosa.load(tmp_path, sr=SR, mono=True, duration=MAX_AUDIO_S)
        if audio.size < N_FFT * 4:
            raise HTTPException(status_code=400, detail="Recording is too short to segment.")
        return JSONResponse(transcribe_audio(audio))
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
