"""YOLO video/webcam inference for full-frame, quarter, and two-stage detection."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("/mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset/original_video.mp4")
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs"
CSV_FIELDS = [
    "frame_index", "timestamp_seconds", "detection_index", "class_id",
    "class_name", "confidence", "xmin", "ymin", "xmax", "ymax",
    "stage", "source_region",
]


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    stage: str
    source_region: str = ""


class FrameDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv26 영상/웹캠 탐지 결과를 원본 FPS와 해상도의 MP4로 저장"
    )
    parser.add_argument("--model", type=Path, required=True, help="사용할 YOLO .pt 모델")
    parser.add_argument(
        "--source", default=str(DEFAULT_SOURCE),
        help=f"입력 영상 경로 또는 웹캠 번호(예: 0). 기본값: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--method", choices=("full", "quarter", "two-stage"), default="full",
        help="프레임 탐지 방식",
    )
    parser.add_argument(
        "--plate-model", type=Path,
        help="two-stage의 2단계 번호판 모델. 생략하면 --model을 재사용",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="0", help="예: 0, 1, cpu")
    parser.add_argument(
        "--codec", choices=("mp4v", "H264"), default="H264",
        help="최종 영상 코덱. 기본 H264는 OpenCV 저장 후 FFmpeg libx264로 변환",
    )
    parser.add_argument(
        "--fallback-fps", type=float, default=30.0,
        help="웹캠이 FPS를 보고하지 않을 때 사용할 값",
    )
    parser.add_argument("--max-frames", type=int, help="앞 N프레임만 처리(파이프라인 확인용)")
    parser.add_argument(
        "--infer-every", type=int, default=1,
        help="N프레임마다 추론. 건너뛴 프레임은 직전 bbox를 그려 저장",
    )
    parser.add_argument("--margin-ratio", type=float, default=0.05)
    parser.add_argument("--global-nms-iou", type=float, default=0.50)
    parser.add_argument("--vehicle-class-id", type=int, default=1)
    parser.add_argument("--plate-class-id", type=int, default=0)
    parser.add_argument("--vehicle-crop-conf", type=float, default=0.25)
    parser.add_argument(
        "--draw-vehicles", action="store_true",
        help="two-stage 결과 영상에 차량 bbox도 표시",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("conf", "iou", "global_nms_iou", "vehicle_crop_conf"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')}는 0~1 사이여야 합니다.")
    if not 0.0 <= args.margin_ratio < 0.5:
        raise ValueError("--margin-ratio는 0 이상 0.5 미만이어야 합니다.")
    for name in ("image_size", "infer_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')}는 1 이상이어야 합니다.")
    if args.fallback_fps <= 0:
        raise ValueError("--fallback-fps는 0보다 커야 합니다.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames는 1 이상이어야 합니다.")
    if not args.model.expanduser().is_file():
        raise FileNotFoundError(f"모델 파일이 없습니다: {args.model}")
    if args.plate_model is not None and not args.plate_model.expanduser().is_file():
        raise FileNotFoundError(f"번호판 모델 파일이 없습니다: {args.plate_model}")


def make_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"video_detection_{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(path, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def result_detections(result: Any, stage: str, source_region: str = "") -> list[Detection]:
    detections: list[Detection] = []
    if result.boxes is None:
        return detections
    names = normalize_names(result.names)
    for xyxy, confidence, class_value in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        class_id = int(class_value)
        detections.append(
            Detection(
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                confidence=float(confidence),
                box=tuple(float(value) for value in xyxy),
                stage=stage,
                source_region=source_region,
            )
        )
    return detections


def box_iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_nms(detections: list[Detection], threshold: float) -> list[Detection]:
    remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[Detection] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            candidate for candidate in remaining
            if candidate.class_id != current.class_id
            or box_iou(current.box, candidate.box) <= threshold
        ]
    return kept


def quarter_regions(width: int, height: int, margin_ratio: float) -> list[tuple[int, int, int, int]]:
    """Return TL, TR, BL, BR regions with overlap around the center lines."""
    center_x, center_y = width // 2, height // 2
    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))
    return [
        (0, 0, min(width, center_x + margin_x), min(height, center_y + margin_y)),
        (max(0, center_x - margin_x), 0, width, min(height, center_y + margin_y)),
        (0, max(0, center_y - margin_y), min(width, center_x + margin_x), height),
        (max(0, center_x - margin_x), max(0, center_y - margin_y), width, height),
    ]


class FullFrameDetector:
    def __init__(self, model: YOLO, args: argparse.Namespace) -> None:
        self.model = model
        self.args = args

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame, imgsz=self.args.image_size, conf=self.args.conf,
            iou=self.args.iou, device=self.args.device, verbose=False, save=False,
        )[0]
        return result_detections(result, "full_frame")


class QuarterDetector:
    def __init__(self, model: YOLO, args: argparse.Namespace) -> None:
        self.model = model
        self.args = args

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        regions = quarter_regions(width, height, self.args.margin_ratio)
        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in regions]
        results = self.model.predict(
            source=crops, imgsz=self.args.image_size, conf=self.args.conf,
            iou=self.args.iou, batch=4, device=self.args.device,
            verbose=False, save=False,
        )
        candidates: list[Detection] = []
        for region_index, (region, result) in enumerate(zip(regions, results)):
            offset_x, offset_y = region[0], region[1]
            for item in result_detections(result, "quarter", f"q{region_index}"):
                x1, y1, x2, y2 = item.box
                candidates.append(
                    Detection(
                        class_id=item.class_id, class_name=item.class_name,
                        confidence=item.confidence,
                        box=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                        stage=item.stage, source_region=item.source_region,
                    )
                )
        return class_aware_nms(candidates, self.args.global_nms_iou)


class TwoStageDetector:
    def __init__(self, vehicle_model: YOLO, plate_model: YOLO, args: argparse.Namespace) -> None:
        self.vehicle_model = vehicle_model
        self.plate_model = plate_model
        self.args = args

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        first_result = self.vehicle_model.predict(
            source=frame, imgsz=self.args.image_size,
            conf=min(self.args.conf, self.args.vehicle_crop_conf),
            iou=self.args.iou, device=self.args.device, verbose=False, save=False,
        )[0]
        vehicles = [
            item for item in result_detections(first_result, "vehicle_full_frame")
            if item.class_id == self.args.vehicle_class_id
            and item.confidence >= self.args.vehicle_crop_conf
        ]
        crops: list[np.ndarray] = []
        metadata: list[tuple[int, int, str]] = []
        for vehicle_index, vehicle in enumerate(vehicles, start=1):
            x1, y1, x2, y2 = vehicle.box
            left = max(0, int(math.floor(x1)))
            top = max(0, int(math.floor(y1)))
            right = min(width, int(math.ceil(x2)))
            bottom = min(height, int(math.ceil(y2)))
            if right > left and bottom > top:
                crops.append(frame[top:bottom, left:right])
                metadata.append((left, top, f"vehicle_{vehicle_index}"))
        plates: list[Detection] = []
        if crops:
            results = self.plate_model.predict(
                source=crops, imgsz=self.args.image_size, conf=self.args.conf,
                iou=self.args.iou, batch=len(crops), device=self.args.device,
                verbose=False, save=False,
            )
            for (offset_x, offset_y, region_name), result in zip(metadata, results):
                for item in result_detections(result, "plate_vehicle_crop", region_name):
                    if item.class_id != self.args.plate_class_id:
                        continue
                    x1, y1, x2, y2 = item.box
                    plates.append(
                        Detection(
                            class_id=item.class_id, class_name=item.class_name,
                            confidence=item.confidence,
                            box=(
                                max(0.0, min(width, x1 + offset_x)),
                                max(0.0, min(height, y1 + offset_y)),
                                max(0.0, min(width, x2 + offset_x)),
                                max(0.0, min(height, y2 + offset_y)),
                            ),
                            stage=item.stage, source_region=item.source_region,
                        )
                    )
        final_plates = class_aware_nms(plates, self.args.global_nms_iou)
        return (vehicles if self.args.draw_vehicles else []) + final_plates


def make_detector(args: argparse.Namespace) -> FrameDetector:
    model = YOLO(str(args.model.expanduser().resolve()))
    if args.method == "full":
        return FullFrameDetector(model, args)
    if args.method == "quarter":
        return QuarterDetector(model, args)
    plate_path = (args.plate_model or args.model).expanduser().resolve()
    plate_model = YOLO(str(plate_path))
    vehicle_names = normalize_names(model.names)
    plate_names = normalize_names(plate_model.names)
    if args.vehicle_class_id not in vehicle_names:
        raise ValueError(
            f"vehicle class {args.vehicle_class_id}가 1단계 모델에 없습니다: {vehicle_names}"
        )
    if args.plate_class_id not in plate_names:
        raise ValueError(
            f"plate class {args.plate_class_id}가 2단계 모델에 없습니다: {plate_names}"
        )
    return TwoStageDetector(model, plate_model, args)


def parse_source(source: str) -> tuple[int | str, str]:
    if source.isdecimal():
        return int(source), f"webcam{source}"
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"입력 영상이 없습니다: {path}")
    return str(path), path.stem


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.box)
        color = (0, 255, 0) if detection.stage.startswith("plate") else (255, 140, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            output, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1, cv2.LINE_AA,
        )
    return output


def detection_row(
    frame_index: int, fps: float, detection_index: int, detection: Detection,
) -> dict[str, Any]:
    x1, y1, x2, y2 = detection.box
    return {
        "frame_index": frame_index,
        "timestamp_seconds": f"{frame_index / fps:.6f}",
        "detection_index": detection_index,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": f"{detection.confidence:.6f}",
        "xmin": f"{x1:.3f}", "ymin": f"{y1:.3f}",
        "xmax": f"{x2:.3f}", "ymax": f"{y2:.3f}",
        "stage": detection.stage, "source_region": detection.source_region,
    }


def serializable_config(args: argparse.Namespace, source: int | str) -> dict[str, Any]:
    config = vars(args).copy()
    for key in ("model", "plate_model", "output_root"):
        value = config[key]
        config[key] = str(value.expanduser().resolve()) if value is not None else None
    config["resolved_source"] = source
    config["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return config


def require_h264_encoder() -> str:
    """Check the system encoder before starting expensive inference."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            "기본 H264 저장에는 FFmpeg가 필요합니다. ffmpeg를 설치하거나 "
            "--codec mp4v를 지정하세요(mp4v는 VS Code 미리보기 호환성이 낮음)."
        )
    result = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        capture_output=True, text=True, check=True,
    )
    if not any("libx264" in line.split() for line in result.stdout.splitlines()):
        raise RuntimeError("FFmpeg에 libx264 encoder가 없습니다. libx264 지원 FFmpeg를 설치하세요.")
    return executable


def encode_h264(source: Path, destination: Path, executable: str) -> None:
    """Finalize browser-compatible MP4; keep the intermediate if encoding fails."""
    if destination.exists():
        raise FileExistsError(f"변환 결과가 이미 있습니다: {destination}")
    result = subprocess.run(
        [
            executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-i", str(source), "-map", "0:v:0", "-c:v", "libx264",
            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(destination),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"H264 변환 실패. 중간 영상은 보존됩니다: {source}\n{result.stderr}"
        )
    # This is the intermediate created by this run, not an existing input video.
    source.unlink()


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    capture_source, source_name = parse_source(args.source)
    ffmpeg_path = require_h264_encoder() if args.codec == "H264" else None
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root.expanduser().resolve() / f"{source_name}_{args.method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = make_logger(output_dir / "video_detection.log")
    config = serializable_config(args, capture_source)
    config["writer_codec"] = "mp4v"
    config["final_video_encoder"] = "libx264" if ffmpeg_path else "opencv_mp4v"
    config["ffmpeg_executable"] = ffmpeg_path
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"[INFO] config={config}")
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"영상 소스를 열지 못했습니다: {args.source}")
    writer: cv2.VideoWriter | None = None
    started_at = time.perf_counter()
    frame_count = 0
    inference_frame_count = 0
    detection_count = 0
    interrupted = False
    fps = args.fallback_fps
    width = height = 0
    video_path = output_dir / "annotated.mp4"
    intermediate_path = output_dir / "annotated_mp4v.mp4" if ffmpeg_path else video_path
    try:
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError(f"첫 프레임을 읽지 못했습니다: {args.source}")
        height, width = first_frame.shape[:2]
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else args.fallback_fps
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(
            str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(
                "VideoWriter를 열지 못했습니다(writer_codec=mp4v). "
                "OpenCV/FFmpeg 저장 기능을 확인하세요."
            )
        logger.info(
            f"[INFO] width={width}, height={height}, fps={fps:.6f}, "
            f"reported_frames={reported_frames}, writer_codec=mp4v, final_codec={args.codec}"
        )
        detector = make_detector(args)
        last_detections: list[Detection] = []
        pending_frame: np.ndarray | None = first_frame
        with (output_dir / "detections.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            while pending_frame is not None:
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break
                frame = pending_frame
                pending_frame = None
                if frame.shape[:2] != (height, width):
                    raise ValueError(
                        f"프레임 해상도가 중간에 변경되었습니다: {frame.shape[1]}x{frame.shape[0]}"
                    )
                if frame_count % args.infer_every == 0:
                    last_detections = detector.detect(frame)
                    inference_frame_count += 1
                for detection_index, detection in enumerate(last_detections, start=1):
                    csv_writer.writerow(detection_row(frame_count, fps, detection_index, detection))
                detection_count += len(last_detections)
                writer.write(draw_detections(frame, last_detections))
                frame_count += 1
                if frame_count == 1 or frame_count % 100 == 0:
                    logger.info(
                        f"[LOG] frames={frame_count:,}, inference_frames={inference_frame_count:,}, "
                        f"current_detections={len(last_detections)}"
                    )
                ok, next_frame = capture.read()
                pending_frame = next_frame if ok and next_frame is not None else None
    except KeyboardInterrupt:
        interrupted = True
        logger.info("[INFO] 사용자 중단 요청을 받아 현재까지 결과를 마무리합니다.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_count == 0:
        raise RuntimeError(f"저장된 프레임이 없습니다. 실행 로그를 확인하세요: {output_dir}")
    inference_pipeline_seconds = time.perf_counter() - started_at
    encoding_seconds = 0.0
    if ffmpeg_path:
        logger.info("[INFO] H264 변환 중. 완료되면 annotated.mp4를 열 수 있습니다.")
        encoding_started_at = time.perf_counter()
        try:
            encode_h264(intermediate_path, video_path, ffmpeg_path)
        except Exception:
            logger.exception(f"[ERROR] 변환 실패. 중간 영상 위치: {intermediate_path}")
            raise
        encoding_seconds = time.perf_counter() - encoding_started_at
        logger.info(f"[INFO] H264 변환 완료: {video_path}")
    elapsed = time.perf_counter() - started_at
    summary = {
        "output_directory": str(output_dir),
        "video": str(output_dir / "annotated.mp4"),
        "frames_written": frame_count,
        "frames_inferred": inference_frame_count,
        "detection_rows": detection_count,
        "source_fps": fps,
        "width": width, "height": height,
        "video_codec": args.codec,
        "inference_pipeline_seconds": inference_pipeline_seconds,
        "encoding_seconds": encoding_seconds,
        "elapsed_seconds": elapsed,
        "processing_fps": frame_count / elapsed if elapsed > 0 else None,
        "interrupted": interrupted,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"[RESULT] {summary}")
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
