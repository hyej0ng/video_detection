# YOLOv26 Video Detection

학습한 번호판 탐지 모델을 영상 파일 또는 웹캠에 적용하고, bbox가 그려진 MP4와 프레임별 탐지 기록을 저장하는 독립 프로젝트입니다. 상위 `landing_pjt` 코드에 의존하지 않으므로 이 폴더만 별도 Git 저장소로 공개할 수 있습니다.

## 파이프라인

```text
공용 데이터 폴더의 영상 파일 / 웹캠 
  → 프레임 읽기
  → 탐지 방식 선택
      ├─ full: 전체 프레임 1회 탐지
      ├─ quarter: 마진 포함 4분할 → 좌표 복원 → global NMS
      └─ two-stage: 차량 탐지 → 차량 crop → 번호판 탐지 → 좌표 복원 → global NMS
  → bbox와 confidence overlay
  → OpenCV로 원본 해상도·FPS의 중간 MP4 저장
  → FFmpeg로 H.264 최종 MP4 변환
  → detections.csv + 실행 설정/요약 JSON + 로그 저장
```

권장 작업 순서는 다음과 같습니다.

1. 5~30초 길이의 샘플 영상과 `--max-frames`로 입출력, class ID, bbox를 확인합니다.
2. 같은 영상에서 `full`, `quarter`, `two-stage`를 각각 실행합니다.
3. confidence와 global NMS IoU를 기존 이미지 test 평가 조건과 맞춥니다.
4. 처리 속도(`processing_fps`)와 육안 품질을 비교한 뒤 전체 영상을 실행합니다.
5. 원거리 번호판의 FN, 반사광/간판의 FP, 프레임 사이 bbox 흔들림을 따로 기록합니다.

영상에서 Precision/Recall/F1/mIoU를 계산하려면 별도의 프레임별 정답 bbox가 필요합니다. 현재 코드는 추론과 실행 속도를 기록하며, 정답 라벨 없이 정량 탐지 성능을 계산하지 않습니다. 처리 FPS가 입력 FPS 이상인지 실제 학습 모델과 대상 영상으로 확인한 후 실시간 가능 여부를 판단하세요.

> OpenCV `VideoWriter`는 입력 영상의 오디오를 복사하지 않습니다. 오디오가 필요하면 결과 영상과 원본 오디오를 FFmpeg로 다시 mux해야 합니다.

## 설치

Python 3.11 환경을 권장합니다. GPU 사용 시 먼저 장비의 CUDA에 맞는 PyTorch를 설치한 뒤 나머지 패키지를 설치하세요.

```bash
cd /home/hyejong/landing_pjt
conda activate yolo
python -m pip install -r 04_video_detection/requirements.txt
ffmpeg -version
```

기본 H.264 저장에는 시스템 FFmpeg의 `libx264` encoder가 필요합니다. 현재 서버에서는 확인되었습니다. 다른 환경에서 FFmpeg가 없다면 Conda 환경에 설치할 수 있습니다.

```bash
conda install -c conda-forge ffmpeg
```

## 입력 영상 위치

공용 원본 폴더의 영상을 복사하거나 전처리하지 않고 직접 읽습니다. 기본 입력은 실제 확인된 다음 파일입니다.

```text
/mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset/original_video.mp4
```

`--source`를 생략하면 위 영상을 사용합니다. 다른 영상을 사용할 때는 파일 경로를 직접 지정하세요. 현재 확인된 파일은 다음과 같습니다.

| 영상 | 해상도 | FPS | 프레임 수 | 길이 |
| --- | --- | ---: | ---: | ---: |
| `original_video.mp4` | 1920×1080 | 30 | 322 | 약 10.73초 |
| `data/sample.mp4` | 3840×2160 | 30 | 1,800 | 약 60.01초 |

두 경로는 위 공용 폴더 기준입니다. 원본에는 결과를 저장하지 않으며, 기본 결과 경로는 `04_video_detection/outputs/`입니다. 아래 실행 명령은 모두 `landing_pjt` 루트 기준입니다. 모델은 기존 학습 폴더의 `best.pt`를 사용하고, 모델과 원본 영상은 Git에 넣지 않습니다.

```text
04_video_detection/
├── outputs/                # 실행 결과는 .gitignore 처리됨
├── tests/
├── video_detection.py
└── requirements.txt
```

## 실행

### 1. Baseline 전체 프레임 탐지

```bash
python 04_video_detection/video_detection.py \
  --method full \
  --model 01_baseline/runs/baseline_yolo26n_20260811-100138_50epoch_auto/weights/best.pt \
  --device 0 \
  --conf 0.25 \
  --iou 0.70
```

### 2. 4분할 탐지

분할 중심선에 걸친 번호판이 잘리지 않도록 기본 5% 마진을 주고, 복원된 중복 bbox는 global NMS로 합칩니다.

```bash
python 04_video_detection/video_detection.py \
  --method quarter \
  --model 02_quarter/runs/quarter_yolo26n_20260903-152325/weights/best.pt \
  --margin-ratio 0.05 \
  --global-nms-iou 0.50 \
  --device 1
```

### 3. 차량 중심 2단계 탐지

기본 class는 `plate=0`, `car=1`입니다. 하나의 2-class 모델을 두 단계에서 재사용합니다. 차량 모델과 번호판 모델이 다르면 `--plate-model`을 추가합니다.

```bash
python 04_video_detection/video_detection.py \
  --method two-stage \
  --model 03_two_stage/runs/two_stage_yolo26n_20260916-162134/weights/best.pt \
  --draw-vehicles \
  --device 0
```

서로 다른 모델을 쓰는 예시는 다음과 같습니다.

```bash
python 04_video_detection/video_detection.py \
  --method two-stage \
  --model /absolute/path/to/vehicle_model.pt \
  --plate-model /absolute/path/to/plate_model.pt \
  --source /mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset/original_video.mp4
```

### 4. 웹캠과 smoke test

웹캠 0번은 아래처럼 실행합니다. 종료할 때는 `Ctrl+C`를 누릅니다.

```bash
python 04_video_detection/video_detection.py \
  --method full \
  --model 01_baseline/runs/baseline_yolo26n_20260820-142748_30epoch_auto/weights/best.pt \
  --source 0
```

처음 100프레임만 검사하려면 다음 옵션을 추가합니다.

```bash
python 04_video_detection/video_detection.py \
  --method full \
  --model 01_baseline/runs/baseline_yolo26n_20260820-142748_30epoch_auto/weights/best.pt \
  --max-frames 100 \
  --device cpu
```

`data/sample.mp4`를 선택해 처음 100프레임만 검사하려면 다음처럼 실행합니다.

```bash
python 04_video_detection/video_detection.py \
  --method full \
  --model 01_baseline/runs/baseline_yolo26n_20260820-142748_30epoch_auto/weights/best.pt \
  --source /mnt/hdd_10tb_sda/YOLO_Object_Detection_Dataset/data/sample.mp4 \
  --max-frames 100 \
  --device 0
```

짧은 테스트에서 bbox와 출력 영상이 정상이면 `--max-frames 100`을 제거하여 전체 영상을 처리합니다.

처리 속도가 원본 FPS보다 느리면 모든 프레임을 보존하되 N프레임마다 한 번만 추론할 수 있습니다. 중간 프레임에는 직전 bbox가 표시됩니다.

```bash
python 04_video_detection/video_detection.py ... --infer-every 2
```

기본 코덱은 `H264`입니다. `full`, `quarter`, `two-stage` 모두 추가 옵션 없이 재생 호환성을 높인 H.264 결과를 저장합니다. OpenCV로 `annotated_mp4v.mp4`를 먼저 저장하고, writer 종료 후 시스템 FFmpeg의 `libx264`로 `annotated.mp4`를 생성합니다. 최종 영상에는 `yuv420p`와 `faststart`를 적용합니다.

변환 성공 후 이번 실행에서 생성한 중간 파일만 정리합니다. 변환에 실패하면 중간 영상과 오류 로그가 남습니다. 결과 영상은 `[INFO] H264 변환 완료` 또는 `[RESULT]`가 출력된 뒤 여세요. FFmpeg가 없다면 추론 전에 오류를 표시합니다. 명시적으로 `--codec mp4v`를 지정하면 변환을 건너뛰지만 VS Code에서 재생되지 않을 수 있습니다.

## 결과 구조

실행마다 timestamp가 붙은 새 폴더가 생성되므로 기존 결과를 덮어쓰지 않습니다.

```text
04_video_detection/outputs/original_video_full_YYYYMMDD-HHMMSS/
├── annotated.mp4          # bbox가 표시된 H.264 결과 영상(기본)
├── detections.csv         # frame/time/class/confidence/원본 좌표
├── run_config.json        # 모델, threshold, device 등 재현 설정
├── run_summary.json       # 프레임 수, 처리 시간, processing FPS
└── video_detection.log
```

출력 해상도와 FPS는 입력에서 읽은 값을 유지합니다. 웹캠이 FPS를 보고하지 않으면 기본 30 FPS를 사용하며 `--fallback-fps`로 바꿀 수 있습니다.

`run_summary.json`의 `encoding_seconds`는 H.264 변환 시간이고, `inference_pipeline_seconds`는 그 이전 모델 로딩·추론·그리기·중간 저장 시간입니다. `elapsed_seconds`와 `processing_fps`에는 최종 변환 시간도 포함됩니다.

`--infer-every`가 2 이상이면 CSV에도 직전 탐지 결과가 반복 기록됩니다. 정량 평가용 기록에는 기본값 1을 사용하세요. timestamp는 `frame_index / FPS`로 계산한 명목 시간이며 가변 FPS 영상이나 웹캠의 실제 촬영 시각과 다를 수 있습니다.

## 검증

```bash
(
  cd 04_video_detection
  python -m unittest discover -s tests -v
)
python 04_video_detection/video_detection.py --help
```

검증 환경은 Conda `yolo`의 Python 3.11.15, OpenCV 5.0.0, Ultralytics 8.4.95입니다. 합성 영상과 로컬 사전학습 모델로 세 모드의 MP4 저장을 확인했고, full 모드 결과의 해상도 320×240, FPS 10, 프레임 수 3을 확인했습니다. 합성 영상에는 차량이 없어 2단계 crop 경로의 좌표 복원은 모의 탐지 결과를 사용하는 단위 테스트로 확인합니다. 실제 번호판 모델의 탐지 품질과 웹캠 입력은 별도로 검증해야 합니다.

2026-10-06에는 `--source`를 생략해 공용 폴더의 `original_video.mp4`를 읽고, 위 baseline 학습 모델을 CPU에서 3프레임 실행했습니다. 결과 MP4의 해상도 1920×1080, FPS 30, 프레임 수 3을 확인했으며 좌표/NMS 테스트 5개도 통과했습니다. 이는 입력 경로와 저장 동작 검증이며 전체 영상의 탐지 성능 평가 결과는 아닙니다.

H.264 기본 저장으로 변경한 뒤에는 실제 2단계 학습 모델로 같은 원본 영상의 3프레임을 CPU에서 처리했습니다. 최종 MP4의 H.264 코덱, `yuv420p`, 1920×1080 해상도, 30 FPS, 3프레임을 확인했습니다. 변환 실패 시 중간 파일 보존 및 기존 결과 덮어쓰기 방지 테스트를 포함한 총 7개 테스트가 통과했습니다.

실제 모델/영상 smoke test 후에는 다음을 확인합니다.

- 결과 MP4의 해상도, FPS, 전체 프레임 수
- 번호판 bbox가 crop 경계에서도 원본 위치에 정확히 복원되는지
- `quarter`와 `two-stage`에서 중복 bbox가 NMS로 제거되는지
- model class ID가 실행 인자와 일치하는지
- `run_summary.json`의 `processing_fps`가 실시간 요구 FPS 이상인지

## 이 폴더를 독립 Git 저장소로 올리기

이 폴더에는 이미 독립 `.git` 저장소가 초기화되어 있습니다. 상위 `landing_pjt/.gitignore`도 `/04_video_detection/`을 제외하므로 부모 저장소와 commit 이력이 섞이지 않습니다.

```bash
cd /home/hyejong/landing_pjt/04_video_detection
git status
git add .
git commit -m "feat: add YOLO video detection pipeline"
git branch -M main
```

GitHub에서 **빈 저장소**를 만든 다음 표시되는 URL을 연결합니다. README나 `.gitignore`를 GitHub에서 미리 생성하지 않으면 첫 push가 단순합니다.

```bash
git remote add origin https://github.com/<USER>/<REPOSITORY>.git
git push -u origin main
```

SSH를 사용한다면 remote만 다음 형식으로 바꿉니다.

```bash
git remote add origin git@github.com:<USER>/<REPOSITORY>.git
```

원본 영상에는 차량 번호, 얼굴, 위치 정보가 포함될 수 있습니다. 공개 저장소에는 영상·가중치·실행 결과를 올리지 말고, 꼭 필요한 예시만 사용 권한과 비식별 처리를 확인한 뒤 추가하세요.
