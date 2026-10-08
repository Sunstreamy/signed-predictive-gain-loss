"""Metric edge cases, using artificial flags only (no HAI scores)."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from evaluation import clean_mask, event_metrics, metrics, segments


class EvaluationTests(unittest.TestCase):
    def test_pre_alarm_is_not_new_detection(self):
        flags = np.array([False, True, True, True, True, False, True, True, False])
        onset = flags & ~np.r_[False, flags[:-1]]
        e = dict(start=12, end_exclusive=15)
        r = event_metrics(flags, onset, 10, e)
        self.assertTrue(r['any_hit']); self.assertTrue(r['pre_alarm']); self.assertFalse(r['detected'])
        self.assertIsNone(r['delay']); self.assertEqual(r['capped_delay'], 3)
        e = dict(start=15, end_exclusive=18)
        r = event_metrics(flags, onset, 10, e)
        self.assertTrue(r['detected']); self.assertEqual(r['delay'], 1)

    def test_mask_does_not_create_onset(self):
        flags = np.array([False, True, True, True, False])
        onset = flags & ~np.r_[False, flags[:-1]]
        selected = np.array([False, False, True, True, True])
        r = metrics(flags, onset, selected)
        self.assertEqual(r['alarm_points'], 2); self.assertEqual(r['episodes'], 0)
        self.assertEqual(r['longest_alarm_seconds'], 2)
        self.assertEqual(segments(flags, 20), [(21, 24)])

    def test_full_support_not_only_current_label(self):
        labels = np.zeros(120, int); labels[20:25] = 1
        clean = clean_mask(labels, 81, 81)
        direct = np.array([not labels[t-81:t+1].any() for t in range(81, 120)])
        np.testing.assert_array_equal(clean, direct)
        self.assertFalse(clean[105-81]); self.assertTrue(clean[106-81])


if __name__ == '__main__':
    unittest.main()
