# YOLOv26 Video Detection

This independent project applies a trained license plate detection model to a video file or webcam. It saves an MP4 video with bounding boxes and a detection record for each frame. It does not depend on code in the parent `landing_pjt` project, so this folder can be published as a separate Git repository.

## Pipeline

```text
Video file in the shared data folder or webcam
  → Read frames
  → Select a detection method
      ├─ full: detect once on the entire frame
      ├─ quarter: split into four regions with margins → restore coordinates → global NMS
      └─ two-stage: detect vehicles → crop vehicles → detect license plates → restore coordinates → global NMS
  → Overlay bounding boxes and confidence scores
  → Save an intermediate MP4 with the original resolution and FPS through OpenCV
  → Convert it to the final H.264 MP4 through FFmpeg
  → Save detections.csv, configuration and summary JSON files, and logs
```

The recommended workflow is as follows.

1. Use a sample video that is 5 to 30 seconds long with `--max-frames` to check input, output, class IDs, and bounding boxes.
2. Run `full`, `quarter`, and `two-stage` on the same video.
3. Match the confidence threshold and global NMS IoU threshold to the conditions used for the existing image test evaluation.
4. Compare `processing_fps` and visual quality before processing the entire video.
5. Record false negatives for distant license plates, false positives caused by reflected light or signs, and bounding box jitter between frames separately.

Separate ground truth bounding boxes for every frame are required to calculate Precision, Recall, F1, and mIoU for a video. The current code records inference results and processing speed. It does not calculate quantitative detection performance without ground truth labels. Determine whether real time processing is possible only after confirming that the processing FPS is at least as high as the input FPS with the trained model and target video.


## Installation

Python 3.11 is recommended. When using a GPU, first install the PyTorch build that matches the CUDA version on the system, then install the remaining packages.

```bash
cd /home/hyejong/landing_pjt
conda activate yolo
python -m pip install -r 04_video_detection/requirements.txt
ffmpeg -version
```

The default H.264 output requires the `libx264` encoder in the system FFmpeg installation. It has been confirmed on the current server. If FFmpeg is unavailable in another environment, it can be installed in the Conda environment.

```bash
conda install -c conda-forge ffmpeg
```

## Input Video Location

Videos in the shared source folder are read directly without copying or preprocessing them. The default input is the following verified file.

```text
/mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset/original_video.mp4
```

The file above is used when `--source` is omitted. Specify a file path to use another video. The currently verified files are listed below.

| Video | Resolution | FPS | Frames | Duration |
| --- | --- | ---: | ---: | ---: |
| `original_video.mp4` | 1920×1080 | 30 | 322 | About 10.73 seconds |
| `data/sample.mp4` | 3840×2160 | 30 | 1,800 | About 60.01 seconds |

Both paths are relative to the shared folder shown above. No results are saved in the source data folder. The default result path is `04_video_detection/outputs/`. All commands below are run from the `landing_pjt` root. Each model uses the `best.pt` file in its existing training folder. Models and source videos are not included in Git.

```text
04_video_detection/
├── outputs/                # Run results are ignored by Git
├── tests/
├── video_detection.py
└── requirements.txt
```

## Usage

### 1. Baseline Detection on the Entire Frame

```bash
python 04_video_detection/video_detection.py \
  --method full \
  --model 01_baseline/runs/baseline_yolo26n_20260811-100138_50epoch_auto/weights/best.pt \
  --device 0 \
  --conf 0.25 \
  --iou 0.70
```

### 2. Detection with Four Regions

A default margin of 5 percent prevents license plates that cross the center boundaries from being cut off. Global NMS merges duplicate bounding boxes after their coordinates are restored.

```bash
python 04_video_detection/video_detection.py \
  --method quarter \
  --model 02_quarter/runs/quarter_yolo26n_20260903-152325/weights/best.pt \
  --margin-ratio 0.05 \
  --global-nms-iou 0.50 \
  --device 1
```

### 3. Vehicle Focused Two Stage Detection

The default classes are `plate=0` and `car=1`. One model with two classes is reused in both stages. Add `--plate-model` when the vehicle model and license plate model are different.

```bash
python 04_video_detection/video_detection.py \
  --method two-stage \
  --model 03_two_stage/runs/two_stage_yolo26n_20260916-162134/weights/best.pt \
  --draw-vehicles \
  --device 0
```

The following example uses different models.

```bash
python 04_video_detection/video_detection.py \
  --method two-stage \
  --model /absolute/path/to/vehicle_model.pt \
  --plate-model /absolute/path/to/plate_model.pt \
  --source /mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset/original_video.mp4
```

### 4. Webcam and Smoke Test

Run webcam 0 as shown below. Press `Ctrl+C` to stop.

```bash
python 04_video_detection/video_detection.py \
  --method full \
  --model 01_baseline/runs/baseline_yolo26n_20260820-142748_30epoch_auto/weights/best.pt \
  --source 0
```

Add the following option to inspect only the first 100 frames.

```bash
python 04_video_detection/video_detection.py \
  --method full \
  --model 01_baseline/runs/baseline_yolo26n_20260820-142748_30epoch_auto/weights/best.pt \
  --max-frames 100 \
  --device cpu
```

Run the following command to inspect only the first 100 frames of `data/sample.mp4`.

```bash
python 04_video_detection/video_detection.py \
  --method full \
  --model 01_baseline/runs/baseline_yolo26n_20260820-142748_30epoch_auto/weights/best.pt \
  --source /mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset/data/sample.mp4 \
  --max-frames 100 \
  --device 0
```

If the bounding boxes and output video are correct in the short test, remove `--max-frames 100` to process the entire video.

When processing is slower than the input FPS, every frame can still be preserved while inference runs only once every N frames. The most recent bounding boxes are displayed on the frames between inference runs.

```bash
python 04_video_detection/video_detection.py ... --infer-every 2
```

The default codec is `H264`. The `full`, `quarter`, and `two-stage` methods save an H.264 result with improved playback compatibility without any additional options. OpenCV first saves `annotated_mp4v.mp4`. After the writer closes, the system FFmpeg installation creates `annotated.mp4` with `libx264`. The final video uses `yuv420p` and `faststart`.

After a successful conversion, only the intermediate file created by the current run is removed. If conversion fails, the intermediate video and error log remain. Open the result video only after `[INFO] H264 conversion complete` or `[RESULT]` appears. An error is shown before inference when FFmpeg is unavailable. Explicitly setting `--codec mp4v` skips conversion, but the result might not play in VS Code.

## Output Structure

Each run creates a new folder with a timestamp, so existing results are not overwritten.

```text
04_video_detection/outputs/original_video_full_YYYYMMDD-HHMMSS/
├── annotated.mp4          # Default H.264 result video with bounding boxes
├── detections.csv         # Frame, time, class, confidence, and source coordinates
├── run_config.json        # Reproducible settings such as the model, thresholds, and device
├── run_summary.json       # Frame count, processing time, and processing FPS
└── video_detection.log
```

The output keeps the resolution and FPS read from the input. If a webcam does not report its FPS, the default value is 30 FPS. This value can be changed with `--fallback-fps`.

In `run_summary.json`, `encoding_seconds` is the time used for H.264 conversion. `inference_pipeline_seconds` is the time used before conversion for model loading, inference, drawing, and intermediate output. Both `elapsed_seconds` and `processing_fps` include the final conversion time.

When `--infer-every` is 2 or greater, the most recent detection results are also repeated in the CSV file. Use the default value of 1 for quantitative evaluation records. Each timestamp is a nominal time calculated as `frame_index / FPS`. It can differ from the actual capture time of a variable FPS video or webcam.

## Verification

```bash
(
  cd 04_video_detection
  python -m unittest discover -s tests -v
)
python 04_video_detection/video_detection.py --help
```

The verification environment uses Python 3.11.15, OpenCV 5.0.0, and Ultralytics 8.4.95 in the Conda `yolo` environment. MP4 output for all three modes was verified with a synthetic video and a local pretrained model. The result from full mode was confirmed to have a resolution of 320×240, an FPS of 10, and 3 frames. The synthetic video contains no vehicles, so coordinate restoration for the crop path in two stage detection is verified through a unit test that uses mock detection results. Detection quality from an actual license plate model and webcam input must be verified separately.

On October 6, 2026, the baseline training model shown above processed 3 frames on the CPU while reading `original_video.mp4` from the shared folder with `--source` omitted. The result MP4 was confirmed to have a resolution of 1920×1080, an FPS of 30, and 3 frames. All 5 coordinate and NMS tests also passed. This verifies the input path and output behavior. It is not a detection performance evaluation of the entire video.

After H.264 became the default output format, the actual two stage training model processed 3 frames from the same source video on the CPU. The final MP4 was confirmed to use the H.264 codec, `yuv420p`, a resolution of 1920×1080, an FPS of 30, and 3 frames. A total of 7 tests passed, including tests for preserving the intermediate file after a conversion failure and preventing existing results from being overwritten.

After a smoke test with an actual model and video, verify the following items.

* The resolution, FPS, and total frame count of the result MP4
* Whether license plate bounding boxes are restored to the correct source coordinates even at crop boundaries
* Whether NMS removes duplicate bounding boxes in `quarter` and `two-stage`
* Whether the model class IDs match the command arguments
* Whether `processing_fps` in `run_summary.json` meets the required FPS for real time processing
