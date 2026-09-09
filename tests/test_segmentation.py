import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app


class SpectrogramInputTests(unittest.TestCase):
    def test_matches_shape_the_segmentor_was_trained_on(self):
        audio = np.random.randn(app.SR).astype(np.float32) * 0.1
        spec = app.segmentor_spectrogram(audio)

        self.assertEqual(spec.shape[0], app.FREQ_BINS)
        self.assertEqual(spec.shape[1], 1 + len(audio) // app.HOP)
        self.assertGreaterEqual(spec.min(), 0.0)
        self.assertLessEqual(spec.max(), 1.0)


class SpanTests(unittest.TestCase):
    def test_frame_runs_map_to_wall_clock_times(self):
        probs = np.zeros(1000)
        probs[100:300] = 0.9

        spans = app.probabilities_to_spans(probs)

        self.assertEqual(len(spans), 1)
        start_s, end_s = spans[0]
        self.assertAlmostEqual(start_s, 100 * app.HOP / app.SR)
        self.assertAlmostEqual(end_s, 300 * app.HOP / app.SR)

    def test_runs_under_the_5ms_floor_are_dropped(self):
        probs = np.zeros(1000)
        probs[100:103] = 0.9  # 3 frames ≈ 4.4 ms

        self.assertEqual(app.probabilities_to_spans(probs), [])

    def test_short_syllables_above_the_floor_are_kept(self):
        probs = np.zeros(1000)
        probs[100:105] = 0.9  # 5 frames ≈ 7.3 ms, shorter than a trill element

        self.assertEqual(len(app.probabilities_to_spans(probs)), 1)

    def test_separate_runs_stay_separate(self):
        probs = np.zeros(1000)
        probs[100:300] = 0.9
        probs[500:700] = 0.9

        self.assertEqual(len(app.probabilities_to_spans(probs)), 2)


class SpectrogramRenderTests(unittest.TestCase):
    def _tone(self, seconds):
        return np.sin(2 * np.pi * 440 * np.arange(int(seconds * app.SR)) / app.SR).astype(np.float32)

    def test_image_width_scales_with_duration(self):
        a = app.build_spectrogram(self._tone(4), app.SR)
        b = app.build_spectrogram(self._tone(8), app.SR)

        self.assertAlmostEqual(b["width_px"] / a["width_px"], 2.0, places=1)

    def test_reported_px_per_s_always_matches_the_image(self):
        # The client positions overlays by time, so width_px / duration must equal
        # the advertised px/s even for clips short enough to hit the width floor.
        for seconds in (0.5, 2.0, 4.0, 10.0):
            with self.subTest(seconds=seconds):
                spec = app.build_spectrogram(self._tone(seconds), app.SR)
                self.assertAlmostEqual(spec["width_px"] / seconds, spec["native_px_per_s"], places=1)

    def test_reports_frequency_ticks_as_height_fractions(self):
        audio = np.sin(2 * np.pi * 440 * np.arange(app.SR) / app.SR).astype(np.float32)

        ticks = app.build_spectrogram(audio, app.SR)["freq_ticks"]

        self.assertTrue(ticks)
        for tick in ticks:
            self.assertGreaterEqual(tick["frac"], 0.0)
            self.assertLessEqual(tick["frac"], 1.0)
        # mel axis is monotonic: higher frequency sits higher in the image
        self.assertEqual([t["frac"] for t in ticks], sorted(t["frac"] for t in ticks))


if __name__ == "__main__":
    unittest.main()
