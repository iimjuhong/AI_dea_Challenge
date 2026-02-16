# 비디오 파일 테스트 모드

카메라 없이 폰으로 촬영한 영상 파일로 YOLO 검출 파이프라인을 테스트한다.

## 디렉토리 구조

```
test/
├── README.md          ← 이 문서
├── run_video.py       ← 테스트 실행 스크립트
└── sample.mp4         ← 여기에 영상 파일을 넣는다 (git 미추적)
```

영상 파일은 `test/` 폴더 안에 넣으면 된다. 파일명은 자유.
mp4, avi, mov 등 OpenCV가 읽을 수 있는 포맷이면 모두 가능.

## 실행 방법

### 기본 실행

```bash
python test/run_video.py --video test/sample.mp4 --model models/yolov8s.onnx
```

브라우저에서 `http://<IP>:5000` 접속하면 실시간 검출 결과를 볼 수 있다.

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--video` | (필수) | 비디오 파일 경로 |
| `--model` | `models/yolov8s.onnx` | YOLO 모델 경로 |
| `--port` | `5000` | 웹서버 포트 |
| `--display-width` | `640` | 스트리밍 해상도 너비 |
| `--display-height` | `480` | 스트리밍 해상도 높이 |
| `--roi-config` | `config/roi_config.json` | ROI 설정 파일 |
| `--conf-threshold` | `0.5` | 검출 신뢰도 임계값 |
| `--no-dynamodb` | off | DynamoDB 전송 비활성화 |
| `--no-loop` | off | 영상 반복 재생 비활성화 |
| `--no-fp16` | off | FP16 추론 비활성화 |

### 실행 예시

```bash
# 검출만 테스트 (DynamoDB 전송 안 함)
python test/run_video.py \
  --video test/sample.mp4 \
  --model models/yolov8s.onnx \
  --no-dynamodb

# ROI + 대기시간 추정 + DynamoDB 전송까지 풀 파이프라인
python test/run_video.py \
  --video test/sample.mp4 \
  --model models/yolov8s.onnx \
  --roi-config config/roi_config.json \
  --start-roi "입구" \
  --end-roi "카운터" \
  --port 5000

# 해상도 변경, 낮은 FPS로 실행
python test/run_video.py \
  --video test/sample.mp4 \
  --model models/yolov8s.onnx \
  --display-width 1280 --display-height 720 \
  --inference-fps 15

# 영상 한 번만 재생 후 종료
python test/run_video.py \
  --video test/sample.mp4 \
  --model models/yolov8s.onnx \
  --no-loop
```

## 파이프라인 구조

`main.py`(카메라 모드)와 동일한 3-스레드 파이프라인을 그대로 사용한다.
바뀌는 것은 소스(카메라 → 비디오 파일)뿐이다.

```
┌─────────────────────────────────────────────────────────┐
│  Thread 1: VideoFileManager (비디오 읽기)                │
│                                                         │
│  mp4 파일 → cv2.VideoCapture → 원본 FPS로 프레임 읽기    │
│           → display 해상도로 리사이즈                     │
│           → 영상 끝나면 처음부터 반복 재생                 │
│           → get_frame()으로 최신 프레임 전달              │
└──────────────────────┬──────────────────────────────────┘
                       │ get_frame()
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Thread 2: Inference Loop (검출 + 추적 + ROI)            │
│                                                         │
│  프레임 수신                                             │
│   → YOLOv8 검출 (사람 검출)                              │
│   → ByteTracker 추적 (track_id 부여)                     │
│   → ROI 영역별 인원 카운트                                │
│   → 체류 필터 (잠깐 들어왔다 나가는 사람 제외)              │
│   → WaitTimeEstimator 대기시간 예측                      │
│   → DynamoDB 주기적 전송 (10초 간격)                      │
│   → 바운딩박스/FPS 오버레이 그리기                         │
│   → JPEG 인코딩 → FrameBuffer에 저장                     │
└──────────────────────┬──────────────────────────────────┘
                       │ FrameBuffer.get()
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Thread 3: Flask 웹서버 (MJPEG 스트리밍)                  │
│                                                         │
│  /              → 웹 대시보드                             │
│  /video_feed    → MJPEG 스트림 (실시간 영상)              │
│  /api/stats     → 검출 통계 JSON                         │
│  /api/tracks    → 추적 결과 JSON                         │
│  /api/wait_time → 대기시간 예측 JSON                      │
│  /api/roi       → ROI 설정 CRUD                          │
└─────────────────────────────────────────────────────────┘
```

### 카메라 모드와의 차이

| | 카메라 모드 (`main.py`) | 비디오 모드 (`test/run_video.py`) |
|---|---|---|
| 소스 | `CameraManager` (CSI/USB) | `VideoFileManager` (파일) |
| Thread 2 (Inference) | 동일 | 동일 |
| Thread 3 (웹서버) | 동일 | 동일 |
| FPS | 카메라 설정값 | 영상 원본 FPS |
| 종료 조건 | 수동 (Ctrl+C) | 수동 또는 영상 끝 (`--no-loop`) |

`VideoFileManager`가 `CameraManager`와 동일한 인터페이스(`start()`, `stop()`, `get_frame()`, `is_running`)를 제공하므로, `app.py`의 inference loop는 수정 없이 그대로 동작한다.
