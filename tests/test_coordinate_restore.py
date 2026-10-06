"""Verify crop-to-frame coordinates without requiring trained weights."""

from types import SimpleNamespace
import unittest

import numpy as np

from video_detection import QuarterDetector, TwoStageDetector


class Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def result(boxes, classes):
    return SimpleNamespace(
        names={0: "plate", 1: "car"},
        boxes=SimpleNamespace(
            xyxy=Values(boxes), cls=Values(classes), conf=Values([0.9] * len(boxes)),
        ),
    )


class Model:
    def __init__(self, results):
        self.results = results
        self.sources = []

    def predict(self, source, **kwargs):
        self.sources.append(source)
        return self.results


class CoordinateRestoreTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            image_size=640, conf=0.25, iou=0.7, device="cpu", margin_ratio=0.05,
            global_nms_iou=0.5, vehicle_class_id=1, plate_class_id=0,
            vehicle_crop_conf=0.25, draw_vehicles=False,
        )

    def test_bottom_right_quarter_offset(self):
        empty = result([], [])
        model = Model([empty, empty, empty, result([[1, 2, 11, 12]], [0])])
        detections = QuarterDetector(model, self.args).detect(np.zeros((80, 100, 3), dtype=np.uint8))
        self.assertEqual(detections[0].box, (46, 38, 56, 48))
        self.assertEqual(detections[0].source_region, "q3")

    def test_vehicle_crop_uses_actual_integer_origin(self):
        vehicles = Model([result([[50.2, 20.2, 120.1, 70.2]], [1])])
        plates = Model([result([[5, 4, 25, 14]], [0])])
        detections = TwoStageDetector(vehicles, plates, self.args).detect(
            np.zeros((100, 200, 3), dtype=np.uint8)
        )
        self.assertEqual(plates.sources[0][0].shape, (51, 71, 3))
        self.assertEqual(detections[0].box, (55, 24, 75, 34))
        self.assertEqual(detections[0].source_region, "vehicle_1")


if __name__ == "__main__":
    unittest.main()
