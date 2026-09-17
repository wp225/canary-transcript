"""Browser check for the playback-speed slider and the Example sample labels.

Runs against the built site/ (build_site.py). Headless Chromium here has no audio
device: the <audio> element reports nothing seekable and its clock never advances,
so real playback cannot be timed. The browser owns "playbackRate makes the clock
run faster" anyway; what the page owns, and what this checks, is that the slider
sets the rate (and keeps it across sample swaps) and that the playhead, transcript
highlight and manifold all render from player.currentTime, so they follow the clock
at whatever rate it runs.
"""
import subprocess
from collections import Counter
import time
import unittest
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PORT = 8779


@unittest.skipUnless((SITE / "index.html").is_file(), "run build_site.py first")
class PlayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = subprocess.Popen(
            ["python", "-m", "http.server", str(PORT), "-d", str(SITE)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(args=["--mute-audio"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.srv.terminate()

    def setUp(self):
        self.page = self.browser.new_page()
        self.page.goto(f"http://127.0.0.1:{PORT}/")
        self.page.wait_for_selector("#results:not([hidden])")

    def tearDown(self):
        self.page.close()

    def set_speed(self, value):
        self.page.eval_on_selector(
            "#speed-range", f"el => {{ el.value = '{value}'; el.dispatchEvent(new Event('input')); }}"
        )

    def test_picker_names_examples_by_species(self):
        """The page claims the pipeline carries beyond canary, so the picker has to
        actually offer a finch recording and say which is which."""
        labels = self.page.eval_on_selector_all("#sample-picker button", "els => els.map(e => e.textContent)")
        self.assertEqual(labels, ["Canary 1", "Canary 2", "Canary 3", "Canary 4", "Finch 1", "Finch 2"])

    def test_finch_examples_transcribe_end_to_end(self):
        """The generalisation claim, checked rather than asserted."""
        self.page.click("#sample-picker button:nth-child(5)")     # Finch 1
        self.page.wait_for_timeout(1500)
        payload = self.page.evaluate("data")
        self.assertGreater(len(payload["segments"]), 5)
        self.assertEqual(payload["peak_confidence"], 1.0)
        self.assertTrue(all(s["label"] for s in payload["segments"]))

    def test_slider_sets_the_playback_rate(self):
        self.set_speed(2)
        self.assertEqual(self.page.evaluate("player.playbackRate"), 2)
        self.assertEqual(self.page.eval_on_selector("#speed-label", "e => e.textContent"), "2.00×")

        self.set_speed(0.5)
        self.assertEqual(self.page.evaluate("player.playbackRate"), 0.5)

    def test_rate_survives_switching_sample(self):
        self.set_speed(2)
        self.page.click("#sample-picker button:nth-child(3)")
        self.page.wait_for_timeout(1000)
        self.assertEqual(self.page.evaluate("player.playbackRate"), 2)

    def test_playhead_transcript_and_manifold_all_follow_the_clock(self):
        # Stand in for the audio clock, then run the same tick() the rAF loop runs.
        seen = self.page.evaluate("""() => {
          let now = 0, redraws = 0;
          Object.defineProperty(player, 'currentTime', {get: () => now, configurable: true});
          const inner = window.drawScatter;
          window.drawScatter = (...a) => { redraws += 1; return inner(...a); };
          const out = [];
          for (const t of data.segments.slice(0, 4).map(s => (s.start_s + s.end_s) / 2)) {
            now = t;
            tick();
            out.push([t, parseFloat(playhead.style.left), activeIndex, redraws]);
          }
          window.drawScatter = inner;
          delete player.currentTime;
          return out;
        }""")
        px_per_sec = self.page.evaluate("pxPerSec")

        for t, left, index, _ in seen:
            self.assertAlmostEqual(left, t * px_per_sec, places=3)  # playhead tracks the clock
            self.assertEqual(index, self.page.evaluate("indexAt(%r)" % t))  # right syllable lit

        indices = [row[2] for row in seen]
        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertGreater(seen[-1][3], seen[0][3])  # manifold repainted as it advanced

    def _brick_pixels(self):
        """Count red (selected-marker) pixels on the manifold canvas."""
        return self.page.evaluate("""() => {
          const c = document.getElementById('scatter');
          const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
          let n = 0;
          for (let i = 0; i < d.length; i += 4) {
            if (d[i] > 120 && d[i] < 200 && d[i + 1] < 110 && d[i + 2] < 110) n += 1;
          }
          return n;
        }""")

    def test_only_the_selected_syllable_is_marked(self):
        """One marker is red however common its label is -- a syllable used 17x must not
        light any more of the plot than one used once."""
        labels = self.page.evaluate("data.segments.map(s => s.label)")
        counts = Counter(labels)
        common, n_common = counts.most_common()[0]
        rare, n_rare = counts.most_common()[-1]
        self.assertGreater(n_common, n_rare)

        self.page.evaluate(f"setActive({labels.index(rare)})")
        self.page.wait_for_timeout(250)
        few = self._brick_pixels()

        self.page.evaluate(f"setActive({labels.index(common)})")
        self.page.wait_for_timeout(250)
        many = self._brick_pixels()

        self.assertGreater(few, 0)                       # the selection is always marked
        self.assertLess(abs(many - few) / max(few, 1), 0.5)   # and frequency changes nothing

    def test_legend_counts_match_the_transcript(self):
        labels = self.page.evaluate("data.segments.map(s => s.label)")
        transcript = self.page.evaluate("data.transcript").split()
        self.assertEqual(labels, transcript)
        self.assertGreater(len(set(labels)), 1)


    def test_zooming_in_never_stretches_the_page(self):
        """Regression: uploads add .has-results, which lays the masthead out as a flex
        row. #results and .result-card then inherited min-width:auto, so a zoomed-in
        spectrogram track (26000px on a 13s file at max zoom) became the page's minimum
        width instead of scrolling inside .spec-scroll, and Fit could not recover
        because fitZoom() measured the blown-out container."""
        self.page.evaluate("document.body.classList.add('has-results')")
        self.page.wait_for_timeout(200)
        self.page.evaluate("document.getElementById('zoom-fit').click()")
        self.page.wait_for_timeout(200)
        card_at_fit = self.page.evaluate(
            "document.querySelector('.result-card').getBoundingClientRect().width"
        )

        for _ in range(5):
            self.page.evaluate("document.getElementById('zoom-in').click()")
            self.page.wait_for_timeout(150)
            m = self.page.evaluate("""() => ({
              track: parseFloat(specTrack.style.width),
              card: document.querySelector('.result-card').getBoundingClientRect().width,
              doc: document.documentElement.scrollWidth,
              client: document.documentElement.clientWidth,
            })""")
            self.assertLessEqual(m["doc"], m["client"])      # page never scrolls sideways
            self.assertAlmostEqual(m["card"], card_at_fit, places=0)   # card never grows

        self.assertGreater(m["track"], m["card"] * 2)        # the track really did grow
        self.page.evaluate("document.getElementById('zoom-fit').click()")
        self.page.wait_for_timeout(250)
        self.assertLess(self.page.evaluate("pxPerSec"), 2000)          # Fit still recovers
        self.page.evaluate("document.body.classList.remove('has-results')")


if __name__ == "__main__":
    unittest.main()
