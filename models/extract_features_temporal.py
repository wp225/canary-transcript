"""
BPA Temporal Feature Extraction
================================
Extracts 26-dim temporal features for:
  - ALL bird syllables (no sampling)
  - TIMIT units: CV, CVC, VV, VC, V

Feature vector (26-dim):
  [ onset_mfcc_1..8 | nucleus_mfcc_1..8 | offset_mfcc_1..8 | voiced_frac | log_dur_ms ]

Each clip is split into three equal zones. Zones shorter than MIN_ZONE_LEN (512 samples)
are zero-padded so very short clips (trill elements ~16ms) are handled without dropping.

Also computes f0_slope (semitones/frame) as metadata — not part of the 26-dim vector
used for matching, but stored for BPA label generation.

Run:
    cd birdtranscript
    nohup python scripts/extract_features_temporal.py > logs/extract_temporal.log 2>&1 &

NOTE: Full bird extraction (~910k syllables) takes several hours.
      TIMIT extraction (~2k files, 5 unit types) takes ~30 minutes.
      Both are cache-aware — re-running skips already-completed files.
      If CV/CVC/VV/VC are cached but V is missing, only V is re-extracted.

Outputs (in data/features/):
    bird_features.pkl   — all bird syllables, 26-dim + f0_slope
    timit_cv.pkl        — TIMIT CV  units (consonant + vowel)
    timit_cvc.pkl       — TIMIT CVC units (consonant + vowel + consonant)
    timit_vv.pkl        — TIMIT VV  units (vowel + vowel)
    timit_vc.pkl        — TIMIT VC  units (vowel + consonant)
    timit_v.pkl         — TIMIT V   units (bare vowel)
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pickle
import logging
import time
import sys
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Relative to this file: scripts/ sits one level under the research repo root.
# The demo imports this module for extract_features_and_slope only, so importing
# must not create directories; main() makes OUT_DIR when a batch actually runs.
BASE        = Path(__file__).resolve().parents[1]
CORPUS_PATH = BASE / 'data/processed/my_corpus.csv'
TIMIT_ROOT  = BASE / 'data/timit'
OUT_DIR     = BASE / 'data/features'

# ── Config ────────────────────────────────────────────────────────────────────
BIRDS        = ['llb3', 'llb11', 'llb16']
SR_BIRD      = 44100
SR_TIMIT     = 16000

# MFCC
N_MFCC       = 9       # extract 9, skip C0 → use dims 1-8
N_MELS       = 40
FMAX_MEL     = 8000
HOP_LEN      = 64
N_FFT        = 512
MIN_ZONE_LEN = 512     # zero-pad zones shorter than this

# pyin — domain-specific fmin/fmax
PYIN_HOP        = int(SR_BIRD  * 0.01)   # 10ms hop (recomputed per domain below)
PYIN_FMIN_BIRD  = 500.0
PYIN_FMAX_BIRD  = 8000.0
PYIN_FMIN_TIMIT = 60.0
PYIN_FMAX_TIMIT = 400.0

# Filtering
MIN_DUR_BIRD_MS  = 15.0   # handles ~16ms trill elements
MIN_DUR_TIMIT_MS = 20.0   # minimum unit duration for TIMIT
MAX_GAP_MS       = 10.0   # max gap between adjacent phones to call them contiguous
C_ENERGY_GATE    = 1e-4   # RMS energy gate

# Feature layout
N_MFCC_DIMS   = N_MFCC - 1              # 8 dims per zone (C0 skipped)
FEATURE_DIM   = N_MFCC_DIMS * 3 + 2    # 24 MFCC + voiced_frac + log_dur_ms = 26
FEATURE_NAMES = (
    [f'mfcc_onset_{i}'   for i in range(1, 9)] +
    [f'mfcc_nucleus_{i}' for i in range(1, 9)] +
    [f'mfcc_offset_{i}'  for i in range(1, 9)] +
    ['voiced_frac', 'log_dur_ms']
)
assert len(FEATURE_NAMES) == FEATURE_DIM

# ── Phone sets ────────────────────────────────────────────────────────────────
PHONE_MAP = {
    'ax':'ah','ax-h':'ah','axr':'er','ix':'ih','hv':'hh',
    'el':'l','em':'m','en':'n','eng':'ng','nx':'n',
    'pcl':'p','tcl':'t','kcl':'k','bcl':'b','dcl':'d','gcl':'g',
    'h#':'sil','pau':'sil','epi':'sil','q':'t',
}
VOWELS = {
    'aa','ae','ah','ao','aw','ay','eh','er','ey',
    'ih','iy','ow','oy','uh','uw',
}
CONSONANTS = {
    'b','d','g','k','p','t',
    'dh','f','hh','s','sh','th','v','z','zh',
    'ch','jh','m','n','ng','l','r','w','y',
}


# ── Audio loaders ─────────────────────────────────────────────────────────────
def load_wav_timit(path):
    try:
        y, sr = sf.read(str(path), dtype='float32')
        return y, sr
    except Exception:
        with open(path, 'rb') as f:
            f.seek(1024)
            raw = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32)
            return raw / 32768.0, SR_TIMIT


def load_phn(phn_path):
    segs = []
    with open(phn_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            s, e, ph = int(parts[0]), int(parts[1]), parts[2].lower()
            ph = PHONE_MAP.get(ph, ph)
            if ph == 'sil':
                continue
            if ph not in CONSONANTS and ph not in VOWELS:
                continue
            segs.append((s, e, ph))
    return segs


def is_contiguous(e_prev, s_next, sr, max_gap_ms=MAX_GAP_MS):
    return (s_next - e_prev) <= int(max_gap_ms / 1000 * sr)


# ── Feature extraction ────────────────────────────────────────────────────────
def _zone_mfcc(clip, sr):
    """MFCC dims 1-8 (C0 skipped), averaged over time. Zero-pads short zones."""
    clip = clip.astype(np.float32)
    if len(clip) < MIN_ZONE_LEN:
        clip = np.pad(clip, (0, MIN_ZONE_LEN - len(clip)))
    mfccs = librosa.feature.mfcc(
        y=clip, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS,
        fmax=FMAX_MEL, n_fft=N_FFT, hop_length=HOP_LEN,
    )[1:]
    return mfccs.mean(axis=1).astype(np.float32)


def extract_features_and_slope(y, sr, domain):
    """
    Returns (feature_vec_26dim, voiced_frac, f0_slope).
    Single pyin call shared between voiced_frac and f0_slope.
    """
    assert domain in ('timit', 'bird')
    y = y.astype(np.float32)
    n = len(y)

    # 3-zone MFCC
    z_onset   = y[:n // 3]
    z_nucleus = y[n // 3 : 2 * n // 3]
    z_offset  = y[2 * n // 3:]

    # single pyin call
    fmin = PYIN_FMIN_BIRD  if domain == 'bird' else PYIN_FMIN_TIMIT
    fmax = PYIN_FMAX_BIRD  if domain == 'bird' else PYIN_FMAX_TIMIT
    hop  = int(sr * 0.01)
    try:
        f0, voiced_flag, _ = librosa.pyin(y, sr=sr, fmin=fmin, fmax=fmax,
                                           hop_length=hop)
        voiced_frac = float(np.nanmean(voiced_flag))
        if np.isnan(voiced_frac):
            voiced_frac = 0.0

        voiced_f0 = f0[voiced_flag > 0.5]
        if len(voiced_f0) >= 3:
            semitones = np.log2(voiced_f0 + 1e-9) * 12
            t = np.arange(len(semitones))
            f0_slope = float(np.polyfit(t, semitones, 1)[0])
        else:
            f0_slope = 0.0
    except Exception:
        voiced_frac = 0.0
        f0_slope    = 0.0

    log_dur = float(np.log1p(n / sr * 1000))

    vec = np.concatenate([
        _zone_mfcc(z_onset,   sr),
        _zone_mfcc(z_nucleus, sr),
        _zone_mfcc(z_offset,  sr),
        [voiced_frac, log_dur],
    ]).astype(np.float32)

    return vec, voiced_frac, f0_slope


# ── TIMIT helpers ─────────────────────────────────────────────────────────────
def _passes_gates(clip, dur_ms):
    if dur_ms < MIN_DUR_TIMIT_MS:
        return False
    if float(np.sqrt(np.mean(clip ** 2))) < C_ENERGY_GATE:
        return False
    return True


def _timit_meta(phones, wav_path, s_start, s_end, sr, y):
    clip   = y[s_start:s_end].astype(np.float32)
    dur_ms = (s_end - s_start) / sr * 1000.0
    label  = '+'.join(phones)
    return clip, dur_ms, label


def _timit_record(unit_type, phones, label, dur_ms, wav_path, s_start, s_end, vec, vf, slope):
    return {
        'unit_id':      f'{Path(wav_path).stem}_{unit_type}_{s_start}',
        'source':       'timit',
        'unit_type':    unit_type,
        'label':        label,
        'phones':       phones,
        'duration_ms':  dur_ms,
        'wav_path':     str(wav_path),
        'start_sample': s_start,
        'end_sample':   s_end,
        'features':     vec,
        'voiced_frac':  vf,
        'f0_slope':     slope,
    }


# ── TIMIT extraction ──────────────────────────────────────────────────────────
def extract_timit_all(wav_files):
    """Extract CV, CVC, VV, VC units from all TIMIT files."""
    cv_recs  = []
    cvc_recs = []
    vv_recs  = []
    vc_recs  = []
    v_recs   = []
    skipped  = defaultdict(int)

    for wav_path in tqdm(wav_files, desc='TIMIT', file=sys.stdout):
        phn_path = wav_path.with_suffix('.PHN')
        if not phn_path.exists():
            phn_path = wav_path.with_suffix('.phn')
        if not phn_path.exists():
            continue
        try:
            y, sr = load_wav_timit(wav_path)
        except Exception:
            skipped['load'] += 1
            continue

        segs = load_phn(phn_path)
        n    = len(segs)

        for i in range(n):
            s1, e1, ph1 = segs[i]

            # ── CV ────────────────────────────────────────────────────────────
            if i + 1 < n:
                s2, e2, ph2 = segs[i + 1]
                if (ph1 in CONSONANTS and ph2 in VOWELS
                        and is_contiguous(e1, s2, sr)):
                    clip, dur_ms, label = _timit_meta(
                        [ph1, ph2], wav_path, s1, e2, sr, y)
                    if _passes_gates(clip, dur_ms):
                        try:
                            vec, vf, slope = extract_features_and_slope(
                                clip, sr, 'timit')
                            cv_recs.append(_timit_record(
                                'CV', [ph1, ph2], label, dur_ms,
                                wav_path, s1, e2, vec, vf, slope))
                        except Exception:
                            skipped['cv_feat'] += 1
                    else:
                        skipped['cv_gate'] += 1

            # ── CVC ───────────────────────────────────────────────────────────
            if i + 2 < n:
                s2, e2, ph2 = segs[i + 1]
                s3, e3, ph3 = segs[i + 2]
                if (ph1 in CONSONANTS and ph2 in VOWELS and ph3 in CONSONANTS
                        and is_contiguous(e1, s2, sr)
                        and is_contiguous(e2, s3, sr)):
                    clip, dur_ms, label = _timit_meta(
                        [ph1, ph2, ph3], wav_path, s1, e3, sr, y)
                    if _passes_gates(clip, dur_ms):
                        try:
                            vec, vf, slope = extract_features_and_slope(
                                clip, sr, 'timit')
                            cvc_recs.append(_timit_record(
                                'CVC', [ph1, ph2, ph3], label, dur_ms,
                                wav_path, s1, e3, vec, vf, slope))
                        except Exception:
                            skipped['cvc_feat'] += 1
                    else:
                        skipped['cvc_gate'] += 1

            # ── VV ────────────────────────────────────────────────────────────
            if i + 1 < n:
                s2, e2, ph2 = segs[i + 1]
                if (ph1 in VOWELS and ph2 in VOWELS
                        and is_contiguous(e1, s2, sr)):
                    clip, dur_ms, label = _timit_meta(
                        [ph1, ph2], wav_path, s1, e2, sr, y)
                    if _passes_gates(clip, dur_ms):
                        try:
                            vec, vf, slope = extract_features_and_slope(
                                clip, sr, 'timit')
                            vv_recs.append(_timit_record(
                                'VV', [ph1, ph2], label, dur_ms,
                                wav_path, s1, e2, vec, vf, slope))
                        except Exception:
                            skipped['vv_feat'] += 1
                    else:
                        skipped['vv_gate'] += 1

            # ── VC ────────────────────────────────────────────────────────────
            if i + 1 < n:
                s2, e2, ph2 = segs[i + 1]
                if (ph1 in VOWELS and ph2 in CONSONANTS
                        and is_contiguous(e1, s2, sr)):
                    clip, dur_ms, label = _timit_meta(
                        [ph1, ph2], wav_path, s1, e2, sr, y)
                    if _passes_gates(clip, dur_ms):
                        try:
                            vec, vf, slope = extract_features_and_slope(
                                clip, sr, 'timit')
                            vc_recs.append(_timit_record(
                                'VC', [ph1, ph2], label, dur_ms,
                                wav_path, s1, e2, vec, vf, slope))
                        except Exception:
                            skipped['vc_feat'] += 1
                    else:
                        skipped['vc_gate'] += 1

            # ── V ─────────────────────────────────────────────────────────────
            if ph1 in VOWELS:
                clip, dur_ms, label = _timit_meta(
                    [ph1], wav_path, s1, e1, sr, y)
                if _passes_gates(clip, dur_ms):
                    try:
                        vec, vf, slope = extract_features_and_slope(
                            clip, sr, 'timit')
                        v_recs.append(_timit_record(
                            'V', [ph1], label, dur_ms,
                            wav_path, s1, e1, vec, vf, slope))
                    except Exception:
                        skipped['v_feat'] += 1
                else:
                    skipped['v_gate'] += 1

    log.info(f'TIMIT extraction complete:')
    log.info(f'  CV  : {len(cv_recs):>7,}  ({len(set(r["label"] for r in cv_recs))} types)')
    log.info(f'  CVC : {len(cvc_recs):>7,}  ({len(set(r["label"] for r in cvc_recs))} types)')
    log.info(f'  VV  : {len(vv_recs):>7,}  ({len(set(r["label"] for r in vv_recs))} types)')
    log.info(f'  VC  : {len(vc_recs):>7,}  ({len(set(r["label"] for r in vc_recs))} types)')
    log.info(f'  V   : {len(v_recs):>7,}  ({len(set(r["label"] for r in v_recs))} types)')
    log.info(f'  skipped: {dict(skipped)}')

    return cv_recs, cvc_recs, vv_recs, vc_recs, v_recs


# ── TIMIT V-only extraction (when CV/CVC/VV/VC are already cached) ────────────
def extract_timit_v_only(wav_files):
    """Extract only V (bare vowel) units. Avoids re-running the full 4-type pass."""
    v_recs  = []
    skipped = defaultdict(int)

    for wav_path in tqdm(wav_files, desc='TIMIT-V', file=sys.stdout):
        phn_path = wav_path.with_suffix('.PHN')
        if not phn_path.exists():
            phn_path = wav_path.with_suffix('.phn')
        if not phn_path.exists():
            continue
        try:
            y, sr = load_wav_timit(wav_path)
        except Exception:
            skipped['load'] += 1
            continue

        for s1, e1, ph1 in load_phn(phn_path):
            if ph1 in VOWELS:
                clip, dur_ms, label = _timit_meta([ph1], wav_path, s1, e1, sr, y)
                if _passes_gates(clip, dur_ms):
                    try:
                        vec, vf, slope = extract_features_and_slope(clip, sr, 'timit')
                        v_recs.append(_timit_record(
                            'V', [ph1], label, dur_ms,
                            wav_path, s1, e1, vec, vf, slope))
                    except Exception:
                        skipped['v_feat'] += 1
                else:
                    skipped['v_gate'] += 1

    log.info(f'V-only extraction complete:')
    log.info(f'  V   : {len(v_recs):>7,}  ({len(set(r["label"] for r in v_recs))} types)')
    log.info(f'  skipped: {dict(skipped)}')
    return v_recs


# ── Bird extraction ───────────────────────────────────────────────────────────
def extract_bird_all(corpus):
    log.info(f'Extracting bird features for {len(corpus):,} syllables...')
    records = []
    failed  = 0

    for audio_file, group in tqdm(
        corpus.groupby('audio_file'),
        desc='Bird',
        file=sys.stdout,
        total=corpus['audio_file'].nunique(),
    ):
        try:
            y_full, sr = librosa.load(audio_file, sr=SR_BIRD, mono=True)
        except Exception:
            failed += len(group)
            continue

        for _, row in group.iterrows():
            try:
                onset_s  = max(0.0, float(row['onset_s'])  - 0.01)
                offset_s = float(row['offset_s']) + 0.01
                s = max(0, int(onset_s  * sr))
                e = min(len(y_full), int(offset_s * sr))
                clip = y_full[s:e].astype(np.float32)
                dur_ms = float(row['duration_ms'])

                if len(clip) < int(sr * MIN_DUR_BIRD_MS / 1000):
                    failed += 1
                    continue

                vec, vf, slope = extract_features_and_slope(clip, sr, 'bird')

                records.append({
                    'unit_id':       f'{row["bird"]}_{row.name}',
                    'source':        'bird',
                    'unit_type':     'syllable',
                    'bird':          row['bird'],
                    'label':         row['label'],
                    'syllable_type': row.get('syllable_type', 'unknown'),
                    'audio_file':    audio_file,
                    'onset_s':       float(row['onset_s']),
                    'offset_s':      float(row['offset_s']),
                    'duration_ms':   dur_ms,
                    'features':      vec,
                    'voiced_frac':   vf,
                    'f0_slope':      slope,
                })

            except Exception:
                failed += 1

    log.info(f'Bird done — {len(records):,} records  failed={failed}')
    stype_counts = Counter(r['syllable_type'] for r in records)
    for stype, n in stype_counts.most_common():
        log.info(f'  {stype:15s}: {n:,}')
    return records


# ── Save helper ───────────────────────────────────────────────────────────────
def _save(path, records, extra=None):
    payload = {
        'records':       records,
        'feature_names': FEATURE_NAMES,
        'feature_dim':   FEATURE_DIM,
        'config': {
            'feature_dim':       FEATURE_DIM,
            'n_mfcc_dims':       N_MFCC_DIMS,
            'zones':             3,
            'min_zone_len':      MIN_ZONE_LEN,
            'min_dur_bird_ms':   MIN_DUR_BIRD_MS,
            'min_dur_timit_ms':  MIN_DUR_TIMIT_MS,
            'pyin_fmin_bird':    PYIN_FMIN_BIRD,
            'pyin_fmax_bird':    PYIN_FMAX_BIRD,
            'pyin_fmin_timit':   PYIN_FMIN_TIMIT,
            'pyin_fmax_timit':   PYIN_FMAX_TIMIT,
            'features':          'onset_mfcc1-8 + nucleus_mfcc1-8 + offset_mfcc1-8 + voiced_frac + log_dur_ms',
        },
    }
    if extra:
        payload.update(extra)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    log.info(f'  Saved {path.name}  ({path.stat().st_size/1e6:.1f} MB)  '
             f'{len(records):,} records')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log.info('=' * 60)
    log.info('BPA Temporal Feature Extraction')
    log.info('=' * 60)
    log.info(f'Feature dim     : {FEATURE_DIM}  (26)')
    log.info(f'Min bird dur    : {MIN_DUR_BIRD_MS}ms')
    log.info(f'Min TIMIT dur   : {MIN_DUR_TIMIT_MS}ms')
    log.info(f'Output dir      : {OUT_DIR}')

    # ── TIMIT ─────────────────────────────────────────────────────────────────
    timit_paths = {
        'CV':  OUT_DIR / 'timit_cv.pkl',
        'CVC': OUT_DIR / 'timit_cvc.pkl',
        'VV':  OUT_DIR / 'timit_vv.pkl',
        'VC':  OUT_DIR / 'timit_vc.pkl',
        'V':   OUT_DIR / 'timit_v.pkl',
    }
    base4_done = all(timit_paths[k].exists() for k in ['CV', 'CVC', 'VV', 'VC'])
    v_done     = timit_paths['V'].exists()

    if base4_done and v_done:
        log.info('\n[1/2] All TIMIT pkl files found — skipping extraction')
        for name, p in timit_paths.items():
            with open(p, 'rb') as f:
                d = pickle.load(f)
            n_types = len(set(r['label'] for r in d['records']))
            log.info(f'  {name:4s}: {len(d["records"]):>7,} records  {n_types} types')
    else:
        log.info('\n[1/2] Discovering TIMIT files...')
        timit_root = Path(TIMIT_ROOT)
        wav_files  = list({
            w.resolve()
            for w in list(timit_root.rglob('*.WAV')) +
                      list(timit_root.rglob('*.wav'))
        })
        wav_files = [
            w for w in wav_files
            if w.stem.upper().startswith('SI') or w.stem.upper().startswith('SX')
        ]
        log.info(f'Found {len(wav_files):,} TIMIT WAV files (SI+SX)')

        if not base4_done:
            log.info('Extracting TIMIT units (CV, CVC, VV, VC, V)...')
            cv_recs, cvc_recs, vv_recs, vc_recs, v_recs = extract_timit_all(wav_files)

            log.info('Saving TIMIT pkl files...')
            if not timit_paths['CV'].exists():
                _save(timit_paths['CV'],  cv_recs,  {'unit_type': 'CV'})
            if not timit_paths['CVC'].exists():
                _save(timit_paths['CVC'], cvc_recs, {'unit_type': 'CVC'})
            if not timit_paths['VV'].exists():
                _save(timit_paths['VV'],  vv_recs,  {'unit_type': 'VV'})
            if not timit_paths['VC'].exists():
                _save(timit_paths['VC'],  vc_recs,  {'unit_type': 'VC'})
            if not timit_paths['V'].exists():
                _save(timit_paths['V'],   v_recs,   {'unit_type': 'V'})
        else:
            # CV/CVC/VV/VC cached — only V is missing
            log.info('CV/CVC/VV/VC already cached — extracting V only...')
            v_recs = extract_timit_v_only(wav_files)
            _save(timit_paths['V'], v_recs, {'unit_type': 'V'})

    # ── Birds ─────────────────────────────────────────────────────────────────
    bird_path = OUT_DIR / 'bird_features.pkl'

    if bird_path.exists():
        log.info('\n[2/2] bird_features.pkl found — skipping extraction')
        with open(bird_path, 'rb') as f:
            d = pickle.load(f)
        log.info(f'  Loaded {len(d["records"]):,} bird records')
    else:
        log.info('\n[2/2] Loading bird corpus...')
        corpus = pd.read_csv(CORPUS_PATH)
        corpus = corpus[
            (corpus['bird'].isin(BIRDS)) &
            (corpus['duration_ms'] >= MIN_DUR_BIRD_MS)
        ].copy()

        log.info(f'  Total syllables: {len(corpus):,}')
        for bird, n in corpus['bird'].value_counts().items():
            log.info(f'    {bird}: {n:,}')
        log.info('  NOTE: extracting ALL syllables — this will take several hours')

        bird_records = extract_bird_all(corpus)

        log.info('Saving bird_features.pkl...')
        _save(bird_path, bird_records, {'birds': BIRDS})

    elapsed = time.time() - t0
    log.info(f'\nDone in {elapsed/60:.1f} minutes')
    log.info(f'Output files in {OUT_DIR}:')
    for p in sorted(OUT_DIR.glob('*.pkl')):
        log.info(f'  {p.name:30s}  {p.stat().st_size/1e6:.1f} MB')


if __name__ == '__main__':
    main()
