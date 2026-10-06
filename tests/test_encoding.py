from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from video_detection import encode_h264


class EncodingSafetyTests(unittest.TestCase):
    def test_failed_conversion_preserves_intermediate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "intermediate.mp4"
            destination = Path(directory) / "annotated.mp4"
            source.write_bytes(b"test intermediate")
            failure = subprocess.CompletedProcess([], 1, stdout="", stderr="encoder failed")
            with patch("video_detection.subprocess.run", return_value=failure):
                with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                    encode_h264(source, destination, "ffmpeg")
            self.assertEqual(source.read_bytes(), b"test intermediate")

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "intermediate.mp4"
            destination = Path(directory) / "annotated.mp4"
            destination.write_bytes(b"existing result")
            with patch("video_detection.subprocess.run") as converter:
                with self.assertRaises(FileExistsError):
                    encode_h264(source, destination, "ffmpeg")
                converter.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"existing result")


if __name__ == "__main__":
    unittest.main()
