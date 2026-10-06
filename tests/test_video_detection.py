from __future__ import annotations

import unittest

from video_detection import Detection, box_iou, class_aware_nms, quarter_regions


def detection(class_id: int, confidence: float, box: tuple[float, float, float, float]) -> Detection:
    return Detection(class_id, str(class_id), confidence, box, "test")


class GeometryTests(unittest.TestCase):
    def test_box_iou(self) -> None:
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(box_iou((0, 0, 2, 2), (3, 3, 4, 4)), 0.0)

    def test_nms_removes_same_class_overlap(self) -> None:
        items = [
            detection(0, 0.9, (0, 0, 10, 10)),
            detection(0, 0.8, (1, 1, 11, 11)),
            detection(1, 0.7, (1, 1, 11, 11)),
        ]
        kept = class_aware_nms(items, 0.5)
        self.assertEqual(
            [(item.class_id, item.confidence) for item in kept],
            [(0, 0.9), (1, 0.7)],
        )

    def test_quarters_cover_frame_with_overlap(self) -> None:
        self.assertEqual(
            quarter_regions(100, 80, 0.05),
            [(0, 0, 55, 44), (45, 0, 100, 44), (0, 36, 55, 80), (45, 36, 100, 80)],
        )


if __name__ == "__main__":
    unittest.main()
