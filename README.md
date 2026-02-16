# 식당 대기시간 추정 시스템 (HY-eat)

NVIDIA Jetson Orin Super Nano 기반 실시간 식당 대기열 추적 및 대기시간 추정 시스템

> **최신 업데이트**: YOLOv8s 모델 업그레이드 (2026-02-16)

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [현재 구현 상태](#-현재-구현-상태)
3. [시스템 아키텍처](#-시스템-아키텍처)
4. [빠른 시작](#-빠른-시작)
5. [상세 설정 가이드](#-상세-설정-가이드)
6. [API 문서](#-api-문서)
7. [프로젝트 구조](#-프로젝트-구조)
8. [기술 스택](#-기술-스택)
9. [성능 지표](#-성능-지표)
10. [문제 해결](#-문제-해결)
11. [개발 로드맵](#-개발-로드맵)

---

## 프로젝트 개요

**HY-eat**는 한양대학교 학생식당의 실시간 대기시간을 측정하고 예측하는 AI 기반 시스템입니다.

### 핵심 목표

- 📊 **실시간 대기시간 측정**: 객체 추적 기반 정확한 대기시간 계산
- 🔮 **예측 알고리즘**: 머신러닝 기반 대기시간 예측
- ☁️ **클라우드 연동**: AWS DynamoDB를 통한 데이터 영속화 및 웹 서비스 제공
- ⚡ **엣지 컴퓨팅**: Jetson Nano에서 실시간 처리 (저지연)

### 하드웨어 사양

| 항목 | 사양 |
|------|------|
| **디바이스** | NVIDIA Jetson Orin Super Nano (16GB) |
| **카메라** | Arducam IMX219 (CSI, 1280×720 @ 30fps) |
| **OS** | Ubuntu 22.04 (JetPack 6.2.2) |
| **추론 엔진** | TensorRT 8.x (FP16) |

---

## ✨ 현재 구현 상태

### ✅ Phase 1-2: 카메라 스트리밍 + YOLO 검출 (완료)

<details>
<summary>세부 구현 내용 보기</summary>

- **실시간 카메라 스트리밍**
  - GStreamer 파이프라인 (`nvarguscamerasrc`)
  - 하드웨어 가속 영상 처리 (`nvvidconv`)
  - Zero-copy 메모리 관리
  
- **YOLO 객체 검출**
  - YOLOv8s TensorRT 최적화 (FP16)
  - 실시간 추론: 20-27 FPS
  - Person 클래스만 검출 (class_id=0)
  
- **웹 인터페이스**
  - Flask MJPEG 스트리밍
  - TurboJPEG 고속 인코딩 (2-3배 빠름)
  - 논블로킹 프레임 버퍼

</details>

---

### ✅ Phase 3: ROI 관리 시스템 (완료)

<details>
<summary>세부 구현 내용 보기</summary>

- **웹 기반 ROI 설정**
  - 마우스 클릭으로 다각형 영역 그리기
  - 좌클릭: 꼭짓점 추가
  - 우클릭: 완성
  
- **데이터 영속화**
  - JSON 형식으로 `config/roi_config.json`에 저장
  - 서버 재시작 시 자동 로드
  
- **실시간 카운팅**
  - Point-in-Polygon 알고리즘
  - 다중 ROI 동시 지원
  - 반투명 오버레이 시각화

</details>

---

### ✅ Phase 4: 객체 추적 + 칼만필터 (완료)

<details>
<summary>세부 구현 내용 보기</summary>

- **ByteTrack 다중 객체 추적**
  - 2단계 IoU 매칭 (고/저 신뢰도 분리)
  - 일시적 가림 대응 (occlusion handling)
  - 고유 track_id 부여 및 유지
  
- **칼만필터 안정화**
  - 7차원 상태 벡터: [x, y, w, h, vx, vy, vw]
  - 등속 모델 (constant velocity)
  - Bbox 지터링 제거
  
- **ROI 체류시간 필터**
  - 최소 체류 프레임 수 설정 (기본 30 프레임 ≈ 1초)
  - 오카운팅 방지 (잠깐 들어왔다 나가는 경우 제외)

</details>

---

### ✅ Phase 5: 대기시간 측정 및 예측 (완료)

<details>
<summary>세부 구현 내용 보기</summary>

- **ROI 이벤트 감지**
  - 상태 전이 기반 진입/퇴출 감지
  - track_id별 ROI 플로우 추적
  
- **대기시간 계산**
  - 진입 시각 → 퇴출 시각 차이 계산
  - 듀얼 모드 지원:
    - **단일 ROI**: start_roi 체류시간 = 대기시간
    - **플로우 모드**: start_roi 진입 → end_roi 진입
  
- **하이브리드 예측 알고리즘**
  - **EMA (Exponential Moving Average)**: 과거 트렌드 반영
  - **대기열 크기 보정**: 현재 대기 인원 수 고려
  - **IQR 이상치 필터링**: 비정상 샘플 자동 제거
  - **Stale 트랙 정리**: 300초 초과 시 자동 삭제

</details>

---

### ✅ Phase 6: AWS DynamoDB 연동 (완료) 🎉

<details>
<summary>세부 구현 내용 보기</summary>

- **비동기 배치 전송**
  - boto3 기반 DynamoDB 클라이언트
  - 최대 25개 아이템 배치 쓰기
  - 백그라운드 워커 스레드 (논블로킹)
  
- **데이터 변환 파이프라인**
  - Python snake_case → DynamoDB camelCase 자동 변환
  - PK/SK 자동 생성: `CORNER#{restaurant_id}#{corner_id}` / `{timestamp}`
  - ISO 8601 타임스탬프 (KST, +09:00)
  - TTL 자동 설정 (3일 후 자동 삭제)
  
- **전송 전략**
  - 주기적 전송: 10초마다 자동 전송
  - 값 변경 감지: 대기시간 변경 시 즉시 전송 (최소 2초 간격)
  - Exponential backoff 재시도
  - 전송 실패 시 큐에 재적재
  
- **모니터링**
  - `/api/dynamodb/stats` 엔드포인트
  - 전송 성공/실패 카운트
  - 대기 중인 아이템 수

**데이터 형식**:

**Jetson 전송 (snake_case)**:
```json
{
  "restaurant_id": "hanyang_plaza",
  "corner_id": "western",
  "queue_count": 15,
  "est_wait_time_min": 8,
  "timestamp": 1770349800000
}
```

**DynamoDB 저장 (camelCase)**:
```json
{
  "pk": "CORNER#hanyang_plaza#western",
  "sk": "1770349800000",
  "restaurantId": "hanyang_plaza",
  "cornerId": "western",
  "queueLen": 15,
  "estWaitTimeMin": 8,
  "dataType": "observed",
  "source": "jetson_nano",
  "timestampIso": "2026-02-15T17:00:00+09:00",
  "createdAtIso": "2026-02-15T17:00:01+09:00",
  "ttl": 1770609000
}
```

</details>

---

### ✅ 비디오 파일 테스트 모드 (완료)

<details>
<summary>세부 구현 내용 보기</summary>

- **카메라 없이 검출 파이프라인 테스트**
  - 폰으로 촬영한 영상 파일(mp4, avi, mov 등)로 전체 파이프라인 동작 확인
  - `VideoFileManager`: `CameraManager`와 동일 인터페이스 제공 (drop-in 교체)
  - 기존 `app.py` 수정 없이 검출/추적/ROI/DynamoDB 전부 동작

- **VideoFileManager 주요 기능**
  - 백그라운드 스레드에서 원본 FPS에 맞춰 프레임 읽기
  - 영상 끝나면 자동 루프 (무한 반복 재생)
  - `display_width/height`로 리사이즈
  - `--no-loop` 옵션으로 반복 재생 비활성화

- **사용법**
  ```bash
  python test/run_video.py --video test/sample.mp4 --model models/yolov8s.onnx
  # 브라우저: http://<IP>:5000
  ```

- **상세 문서**: [test/README.md](test/README.md)

</details>

---

### 🚧 Phase 7: 웹 대시보드 (예정)

- 실시간 통계 차트 (Chart.js)
- 시간대별 분석
- 코너별 대기 현황

---

## 🏗️ 시스템 아키텍처

### 전체 파이프라인 (End-to-End)

```
  [운영 모드: main.py]              [테스트 모드: test/run_video.py]

┌──────────────────────────┐     ┌──────────────────────────┐
│   CSI 카메라 (IMX219)     │     │   비디오 파일 (mp4 등)    │
│   1280x720 @ 30fps       │     │   원본 해상도 @ 원본 FPS  │
└───────────┬──────────────┘     └───────────┬──────────────┘
            ↓                                ↓
┌──────────────────────────┐     ┌──────────────────────────┐
│  GStreamer 파이프라인      │     │  cv2.VideoCapture        │
│  nvarguscamerasrc→BGR    │     │  → 리사이즈 → FPS pacing │
└───────────┬──────────────┘     └───────────┬──────────────┘
            ↓                                ↓
┌──────────────────────────┐     ┌──────────────────────────┐
│  CameraManager           │     │  VideoFileManager        │
│  (백그라운드 스레드)       │     │  (백그라운드 스레드)       │
│  - thread-safe 캡처      │     │  - thread-safe 읽기      │
│  - HW JPEG 인코딩        │     │  - 자동 루프 재생         │
└───────────┬──────────────┘     └───────────┬──────────────┘
            └──────────┬─────────────────────┘
                       ↓
              동일 인터페이스: start() / stop()
              get_frame() / is_running
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│         YOLOv8Detector (TensorRT FP16 추론)                  │
│  - ONNX → TensorRT 엔진 자동 변환 및 캐싱                     │
│  - ctypes CUDA 직접 호출 (PyCUDA 불필요)                     │
│  - NMS 후처리 (person만 검출)                                │
│  - FPS: 20-27                                               │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              ROIManager (영역 관리)                           │
│  - Point-in-Polygon 알고리즘                                 │
│  - ROI별 인원 카운팅                                         │
│  - JSON 설정 영구 저장                                       │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│         ByteTracker (다중 객체 추적)                          │
│  - 2단계 IoU 매칭 (고/저 신뢰도 분리)                         │
│  - 칼만필터 bbox 안정화                                       │
│  - track_id 부여 및 유지                                     │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│        WaitTimeEstimator (대기시간 측정/예측)                 │
│  - ROI 진입/퇴출 이벤트 감지                                  │
│  - 실제 대기시간 계산                                         │
│  - 하이브리드 예측 (EMA + 대기열 보정)                        │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
         ┌─────────────┴─────────────┐
         ↓                           ↓
┌──────────────────┐      ┌──────────────────────────┐
│  Flask 웹 서버    │      │  DynamoDBSender          │
│  (0.0.0.0:5000)  │      │  (백그라운드 워커)        │
│                  │      │  - 비동기 배치 전송       │
│  - MJPEG 스트림  │      │  - snake→camel 변환      │
│  - API 엔드포인트 │      │  - PK/SK 생성           │
│  - ROI CRUD      │      │  - TTL 설정             │
└────────┬─────────┘      └──────────┬───────────────┘
         ↓                           ↓
   브라우저 (웹 UI)       AWS DynamoDB (hyeat-waiting-data-dev)
                                     ↓
                            웹 대시보드 (Phase 7 예정)
```

---

### 3-Thread 아키텍처 (성능 최적화)

```
Thread 1 (프레임 소스):
  ├→ [운영] CameraManager._capture_loop
  │    └→ GStreamer에서 프레임 캡처
  └→ [테스트] VideoFileManager._read_loop
       └→ 비디오 파일에서 FPS 맞춰 읽기 (자동 루프)
  └→ _frame에 저장 (thread-safe)

Thread 2 (Inference):
  └→ _inference_loop
     └→ get_frame() → YOLO → ByteTracker → ROI
        └→ WaitTimeEstimator.update()
           └→ DynamoDBSender.send() (10초마다 or 변경 시)
              └→ JPEG 인코딩 → FrameBuffer.put()

Thread 3 (Network/Flask):
  └→ generate_frames()
     └→ FrameBuffer.get() → yield MJPEG
        └→ 느린 클라이언트는 프레임 스킵 (최신만 수신)

Thread 4 (DynamoDB Worker):
  └→ DynamoDBSender._worker_loop
     └→ 큐에서 배치 꺼내기 (최대 25개)
        └→ batch_write_item (재시도 포함)
```

**핵심 설계 원칙**:
- ✅ **비블로킹**: 각 스레드는 완전히 독립적
- ✅ **느린 소비자 무시**: 네트워크 지연이 inference에 영향 없음
- ✅ **Graceful Degradation**: DynamoDB 오류 시 시스템 계속 동작

### 주요 컴포넌트

#### 1. **CameraManager** (`src/core/camera.py`)
```python
- GStreamer 파이프라인 구성
- 백그라운드 스레드에서 프레임 캡처
- thread-safe 프레임 접근
- HW JPEG 인코딩 지원
```

#### 1-1. **VideoFileManager** (`src/core/video_source.py`)
```python
- CameraManager와 동일 인터페이스 (drop-in 교체)
- cv2.VideoCapture로 비디오 파일 읽기
- 원본 FPS에 맞춘 프레임 pacing
- 영상 끝 자동 루프 재생
```

#### 2. **YOLOv8Detector** (`src/core/detector.py`)
```python
- TensorRT 엔진 자동 빌드/로드
- ctypes를 통한 CUDA 메모리 관리
- FP16 추론 (Jetson 최적화)
- Letterbox 전처리 + NMS 후처리
```

#### 3. **ROIManager** (`src/core/roi_manager.py`)
```python
- 다각형 ROI 저장/불러오기
- Point-in-Polygon 알고리즘
- ROI별 검출 필터링 및 카운팅
- 반투명 오버레이 시각화
```

#### 4. **ByteTracker** (`src/core/tracker.py`)
```python
- KalmanBoxTracker: 7차원 칼만필터
- 2단계 연관 매칭 (고/저신뢰도)
- track_id 자동 부여
- ROIDwellFilter: 체류시간 필터링
```

#### 5. **WaitTimeEstimator** (`src/core/wait_time_estimator.py`)
```python
- 상태 전이 기반 이벤트 감지
- ROI 플로우 추적 (진입→퇴출)
- HybridPredictor: EMA + 대기열 보정
- IQR 이상치 필터링
```

#### 6. **Flask 웹 앱** (`src/web/app.py`)
```python
- MJPEG 스트림 제너레이터
- RESTful API 엔드포인트
- ROI 관리 API (CRUD)
- 대기시간 통계 API
```

---

## 🚀 빠른 시작

### 전제 조건

- ✅ NVIDIA Jetson Orin Super Nano (JetPack 6.2.2)
- ✅ Python 3.10+
- ✅ TensorRT 8.x
- ✅ GStreamer (NVIDIA 플러그인 포함)

### 기본 실행 (DynamoDB 없이)

```bash
# 1. 프로젝트 클론
cd /home/iimjuhong/projects/aidea

# 2. 의존성 설치
pip3 install -r requirements.txt

# 3. YOLO 모델 다운로드 (최초 1회)
bash scripts/download_model.sh

# 4. 프로그램 실행
python3 main.py

# 5. 웹 UI 접속
# http://localhost:5000
```

### 비디오 파일로 테스트 (카메라 없이)

```bash
# 1. test/ 폴더에 영상 파일 넣기
cp ~/Downloads/sample.mp4 test/

# 2. 비디오 모드로 실행
python3 test/run_video.py --video test/sample.mp4 --model models/yolov8s.onnx

# 3. 웹 UI 접속
# http://localhost:5000

# 4. DynamoDB 없이 검출만 테스트
python3 test/run_video.py --video test/sample.mp4 --model models/yolov8s.onnx --no-dynamodb
```

> 상세 옵션은 [test/README.md](test/README.md) 참조

### DynamoDB 연동 실행

```bash
# 1. AWS 자격증명 설정 (.env 파일 또는 환경 변수)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# 2. DynamoDB 설정 파일 수정
nano config/aws_config.json

# 3. 대기시간 추정 활성화 (start-roi 필수)
python3 main.py --start-roi "대기구역" --end-roi "카운터"

# 4. DynamoDB 전송 확인
curl http://localhost:5000/api/dynamodb/stats
```

---

## 📖 상세 설정 가이드

### 1️⃣ ROI (Region of Interest) 설정

대기시간 측정을 위해 ROI 영역을 설정해야 합니다.

#### 웹 UI에서 ROI 그리기

1. **웹 UI 접속**: `http://localhost:5000`
2. **ROI 이름 입력**: 예) "대기구역", "카운터"
3. **그리기 시작 클릭**
4. **마우스로 영역 지정**:
   - 좌클릭: 꼭짓점 추가 (최소 3개)
   - 우클릭: 완성
5. **저장 확인**: `config/roi_config.json` 자동 저장

#### ROI 설정 파일 (`config/roi_config.json`)

```json
{
  "rois": [
    {
      "name": "대기구역",
      "points": [[100, 200], [300, 200], [300, 400], [100, 400]],
      "color": [0, 255, 0]
    },
    {
      "name": "카운터",
      "points": [[400, 200], [600, 200], [600, 400], [400, 400]],
      "color": [255, 0, 0]
    }
  ]
}
```

---

### 2️⃣ AWS DynamoDB 설정

#### DynamoDB 테이블 생성

**AWS CLI 사용**:
```bash
aws dynamodb create-table \
  --table-name hyeat-waiting-data-dev \
  --attribute-definitions \
    AttributeName=pk,AttributeType=S \
    AttributeName=sk,AttributeType=S \
  --key-schema \
    AttributeName=pk,KeyType=HASH \
    AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region ap-northeast-2
```

**AWS 콘솔 사용**:
1. DynamoDB 콘솔 접속
2. "Create table" 클릭
3. 설정:
   - **Table name**: `hyeat-waiting-data-dev`
   - **Partition key**: `pk` (String)
   - **Sort key**: `sk` (String)
   - **Billing mode**: On-demand

#### TTL 설정 (자동 삭제)

```bash
aws dynamodb update-time-to-live \
  --table-name hyeat-waiting-data-dev \
  --time-to-live-specification \
    "Enabled=true, AttributeName=ttl" \
  --region ap-northeast-2
```

#### AWS 설정 파일 (`config/aws_config.json`)

```json
{
  "region": "ap-northeast-2",
  "table_name": "hyeat-waiting-data-dev",
  "restaurant_id": "hanyang_plaza",
  "corner_id": "western"
}
```

**필수 필드**:
- `region`: AWS 리전 (예: `ap-northeast-2`)
- `table_name`: DynamoDB 테이블 이름

**선택 필드**:
- `restaurant_id`: 식당 ID (기본값: "unknown")
- `corner_id`: 코너 ID (기본값: "unknown")

#### AWS 자격증명 설정

**방법 1: 환경 변수** (권장)
```bash
# ~/.bashrc 또는 ~/.profile에 추가
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
```

**방법 2: AWS CLI 설정**
```bash
aws configure
# AWS Access Key ID: [your-key]
# AWS Secret Access Key: [your-secret]
# Default region name: ap-northeast-2
# Default output format: json
```

**보안 주의사항**:
- ⚠️ **절대 코드에 하드코딩하지 마세요**
- ✅ 환경 변수 또는 AWS IAM Role 사용
- ✅ `.gitignore`에 자격증명 파일 추가

---

### 3️⃣ 대기시간 추정 설정

#### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--start-roi` | 대기 시작 ROI 이름 (필수) | None |
| `--end-roi` | 대기 종료 ROI 이름 (선택) | None |
| `--min-dwell` | 최소 체류 프레임 수 | 30 (≈1초@30fps) |

#### 실행 예시

**단일 ROI 모드** (체류시간 = 대기시간):
```bash
python3 main.py --start-roi "대기구역"
```

**플로우 모드** (진입→퇴출):
```bash
python3 main.py --start-roi "대기구역" --end-roi "카운터"
```

**DynamoDB 비활성화**:
```bash
python3 main.py --start-roi "대기구역" --no-dynamodb
```

---

### 4️⃣ 고급 옵션

```bash
python3 main.py \
  --capture-width 1280 \
  --capture-height 720 \
  --display-width 640 \
  --display-height 480 \
  --fps 30 \
  --conf-threshold 0.5 \
  --model models/yolov8s.onnx \
  --start-roi "대기구역" \
  --end-roi "카운터" \
  --min-dwell 30 \
  --aws-config config/aws_config.json
```

**전체 옵션 보기**:
```bash
python3 main.py --help
```


---

## 📡 API 문서

모든 API는 `http://localhost:5000` 또는 `http://[Jetson IP]:5000`에서 접근할 수 있습니다.

### 스트리밍 엔드포인트

#### `GET /video_feed`

MJPEG 실시간 비디오 스트리밍

**응답**: `multipart/x-mixed-replace; boundary=frame`

**사용 예시**:
```html
<img src="http://localhost:5000/video_feed" />
```

---

### 통계 API

#### `GET /api/stats`

실시간 검출 및 추적 통계

**응답 예시**:
```json
{
  "fps": 18.5,
  "person_count": 12,
  "detector_active": true,
  "tracker_active": true,
  "track_ids": [1, 2, 3, 5, 7],
  "roi_counts": {
    "대기구역": 8,
    "카운터": 4
  }
}
```

#### `GET /api/tracks`

현재 활성 추적 객체 목록

**응답 예시**:
```json
{
  "tracker_active": true,
  "tracks": [
    {
      "bbox": [120, 200, 180, 350],
      "confidence": 0.87,
      "track_id": 1
    },
    {
      "bbox": [250, 180, 310, 340],
      "confidence": 0.92,
      "track_id": 2
    }
  ]
}
```

---

### ROI 관리 API

#### `GET /api/roi`

모든 ROI 목록 조회

**응답 예시**:
```json
{
  "rois": [
    {
      "name": "대기구역",
      "points": [[100, 200], [300, 200], [300, 400], [100, 400]],
      "color": [0, 255, 0]
    }
  ]
}
```

#### `POST /api/roi`

새 ROI 추가

**요청 본문**:
```json
{
  "name": "대기구역",
  "points": [[100, 200], [300, 200], [300, 400], [100, 400]],
  "color": [0, 255, 0]
}
```

**응답**: `200 OK` (성공) / `409 Conflict` (이름 중복)

#### `PUT /api/roi/<name>`

기존 ROI 수정

**요청 본문**:
```json
{
  "points": [[110, 210], [310, 210], [310, 410], [110, 410]],
  "new_name": "대기구역_수정",
  "color": [0, 200, 0]
}
```

**응답**: `200 OK` (성공) / `404 Not Found` (없는 ROI)

#### `DELETE /api/roi/<name>`

ROI 삭제

**응답**: `200 OK` (성공) / `404 Not Found` (없는 ROI)

#### `GET /api/roi/stats`

ROI별 현재 인원 수

**응답 예시**:
```json
{
  "roi_counts": {
    "대기구역": 8,
    "카운터": 4
  }
}
```

---

### 대기시간 API

#### `GET /api/wait_time`

현재 대기시간 예측 및 대기열 정보

**응답 예시**:
```json
{
  "predicted_wait": 480.5,
  "current_queue": 8,
  "total_completed": 127,
  "active_waiters": {
    "1": 120.3,
    "2": 85.7,
    "5": 230.1
  },
  "statistics": {
    "total_samples": 127,
    "mean": 465.2,
    "min": 180.5,
    "max": 720.8,
    "recent_10_avg": 485.3
  }
}
```

**필드 설명**:
- `predicted_wait`: 예상 대기시간 (초)
- `current_queue`: 현재 대기 중인 인원 수
- `total_completed`: 완료된 대기시간 샘플 수
- `active_waiters`: track_id별 현재 대기 시간 (초)
- `statistics`: 통계 정보 (평균, 최소/최대, 최근 10개 평균)

---

### DynamoDB 관리 API

#### `GET /api/dynamodb/stats`

DynamoDB 전송 통계

**응답 예시**:
```json
{
  "sent": 1523,
  "errors": 2,
  "pending": 0
}
```

**필드 설명**:
- `sent`: 전송 성공 카운트
- `errors`: 전송 실패 카운트
- `pending`: 대기 중인 아이템 수

---

### 헬스체크 API

#### `GET /health`

시스템 상태 확인

**응답 예시**:
```json
{
  "status": "ok",
  "camera": true,
  "detector": true
}
```



## 📂 프로젝트 구조

```
aidea/
├── main.py                          # 메인 진입점 (카메라 모드)
├── config/
│   ├── aws_config.json             # AWS DynamoDB 설정
│   └── roi_config.json             # ROI 설정 저장
├── test/                            # 비디오 파일 테스트 🆕
│   ├── run_video.py                # 비디오 테스트 실행 스크립트
│   ├── README.md                   # 테스트 모드 사용법
│   └── (*.mp4)                     # 테스트 영상 파일 (git 미추적)
├── frontend/                        # 프론트엔드 타입 정의 (Phase 7)
│   └── src/
│       └── types/
│           └── hyeat.ts            # TypeScript 인터페이스
├── models/
│   ├── yolov8s.onnx                # YOLO 모델
│   └── yolov8s_fp16.engine         # TensorRT 엔진 캐시
├── src/
│   ├── cloud/                       # 클라우드 연동
│   │   └── dynamodb_sender.py      # DynamoDB 전송 모듈
│   ├── core/                       # 핵심 로직
│   │   ├── camera.py              # 카메라 관리자 (운영용)
│   │   ├── video_source.py        # 비디오 파일 관리자 (테스트용) 🆕
│   │   ├── detector.py            # YOLO 검출기
│   │   ├── roi_manager.py         # ROI 관리자
│   │   ├── tracker.py             # ByteTrack 추적기
│   │   └── wait_time_estimator.py # 대기시간 측정/예측
│   └── web/                       # 웹 인터페이스
│       ├── app.py                 # Flask 서버
│       └── templates/
│           └── index.html         # 웹 UI
├── scripts/
│   ├── download_model.sh          # 모델 다운로드
│   └── setup_env.sh               # 환경 설정
├── requirements.txt               # Python 의존성
├── TEST_DYNAMODB.md               # DynamoDB 테스트 가이드
└── docs/
    └── Phase5_대기시간_알고리즘_가이드.md  # 개발 가이드
```

---

## 🛠️ 기술 스택

### Core
- **Python 3.10**
- **OpenCV** (GStreamer 지원)
- **YOLOv8** (Ultralytics ONNX)
- **TensorRT** (FP16 추론)
- **Flask** (웹 서버)
- **boto3** (AWS SDK for Python)

### 하드웨어 최적화
- **GStreamer**: nvarguscamerasrc, nvvidconv
- **TensorRT**: FP16 엔진, CUDA 메모리 직접 관리
- **nvjpegenc**: 하드웨어 JPEG 인코딩
- **TurboJPEG**: NEON SIMD 가속 JPEG 인코딩

### 클라우드 (Phase 6) 🆕
- **AWS DynamoDB**: NoSQL 데이터베이스
- **boto3**: 비동기 배치 쓰기
- **IAM**: 자격증명 및 권한 관리

### 프론트엔드 (Phase 7 예정)
- **Chart.js**: 데이터 시각화
- **WebSocket**: 실시간 양방향 통신
- **TypeScript**: 타입 안전성

---

## 📊 성능 지표

### 현재 성능 (YOLOv8s 기준)
- **FPS**: 20-27 (YOLO + ROI + 추적 처리)
- **추론 시간**: ~37-50ms/frame
- **메모리 사용량**: ~2.5GB (TensorRT 엔진 포함)
- **지연 시간**: <100ms (카메라 → 웹 UI)

### 최적화 요소
- ✅ TensorRT FP16 모드 (2배 속도 향상)
- ✅ 엔진 캐싱 (재시작 시 즉시 로드)
- ✅ Zero-copy GStreamer 파이프라인
- ✅ 하드웨어 JPEG 인코딩

---

## 🎯 핵심 기능

### 1. 실시간 사람 검출
- YOLOv8s TensorRT 최적화
- Confidence threshold 필터링
- NMS를 통한 중복 제거

### 2. ROI 기반 영역 관리
- 웹 UI에서 직접 다각형 그리기
- 여러 영역 동시 관리
- 영역별 인원 수 실시간 카운팅

### 3. 웹 인터페이스
- MJPEG 실시간 스트리밍
- RESTful API (stats, ROI CRUD)
- 반응형 UI 디자인

---

## 🔧 설치 및 설정

### 필수 요구사항
- NVIDIA Jetson Orin Nano (JetPack 6.x)
- Python 3.10+
- TensorRT 8.x
- GStreamer 1.x

### 환경 설정
```bash
# 의존성 설치
pip3 install -r requirements.txt

# 모델 다운로드
bash scripts/download_model.sh
```


---

## 🔧 문제 해결

### 카메라 연결 오류

**증상**: `카메라 시작 실패. 종료합니다.`

**해결 방법**:
```bash
# 카메라 장치 확인
ls -l /dev/video*

# GStreamer 파이프라인 테스트
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! fakesink

# 권한 문제 해결
sudo usermod -aG video $USER
sudo reboot
```

---

### TensorRT 엔진 빌드 실패

**증상**: `TensorRT 엔진 빌드 실패`

**해결 방법**:
```bash
# TensorRT 버전 확인
dpkg -l | grep TensorRT

# CUDA 경로 확인
echo $LD_LIBRARY_PATH

# 기존 엔진 파일 삭제 후 재생성
rm models/yolov8s_fp16.engine
python3 main.py
```

---

### DynamoDB 연결 오류

**증상**: `DynamoDB 클라이언트 초기화 실패`

**원인 1**: AWS 자격증명 없음
```bash
# 환경 변수 확인
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# 설정 후 재실행
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

**원인 2**: 네트워크 연결 문제
```bash
# 인터넷 연결 확인
ping aws.amazon.com

# DynamoDB 엔드포인트 접근 테스트
curl https://dynamodb.ap-northeast-2.amazonaws.com
```

**원인 3**: 테이블 없음
```bash
# 테이블 존재 확인
aws dynamodb describe-table --table-name hyeat-waiting-data-dev --region ap-northeast-2

# 테이블 생성 (상세 설정 가이드 참조)
```

---

### DynamoDB 전송 모니터링

```bash
# 전송 통계 실시간 확인
watch -n 1 'curl -s http://localhost:5000/api/dynamodb/stats | python3 -m json.tool'

# 로그 확인
tail -f logs/aidea.log | grep DynamoDB
```

---

### 메모리 부족 오류

**증상**: Out of Memory (OOM)

**해결 방법**:
```bash
# 스왑 메모리 확인
free -h

# 스왑 메모리 추가 (8GB)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구 적용
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

### 성능 저하

**증상**: FPS가 10 이하로 떨어짐

**원인 및 해결**:

1. **과도한 ROI 수**:
   ```bash
   # ROI 개수 확인
   cat config/roi_config.json | jq '.rois | length'
   
   # 권장: 5개 이하
   ```

2. **높은 해상도**:
   ```bash
   # 해상도 낮추기
   python3 main.py --display-width 640 --display-height 480
   ```

3. **DynamoDB 전송 병목**:
   ```bash
   # DynamoDB 비활성화 테스트
   python3 main.py --no-dynamodb --start-roi "대기구역"
   ```

---

### ROI 설정이 저장되지 않음

**증상**: 재실행 시 ROI가 사라짐

**해결 방법**:
```bash
# config 디렉토리 권한 확인
ls -ld config/

# 권한 부여
chmod 755 config/
chmod 644 config/roi_config.json

# JSON 형식 검증
cat config/roi_config.json | python3 -m json.tool
```

---

## 📝 개발 로드맵

- [x] **Phase 1-2**: 카메라 스트리밍 + YOLO 검출 ✅
- [x] **Phase 3**: ROI 관리 시스템 ✅
- [x] **Phase 4**: 객체 추적 + 칼만필터 오차 보정 ✅
- [x] **Phase 5**: 대기시간 측정 및 예측 ✅
- [x] **Phase 6**: AWS DynamoDB 연동 ✅ 🎉
- [ ] **Phase 7**: 웹 대시보드 완성 🚧
  - [ ] Chart.js 실시간 차트
  - [ ] 시간대별 통계 분석
  - [ ] 코너별 대기 현황
  - [ ] 히스토리 데이터 시각화

---

## 💡 주요 특징

### 🚀 성능
- **20-27 FPS**: YOLOv8s TensorRT FP16 최적화
- **\u003c100ms 지연**: 카메라 → 웹 UI
- **~2GB 메모리**: 효율적인 메모리 관리
- **논블로킹**: 4-스레드 아키텍처로 완전 비동기 처리

### 🎯 정확도
- **ByteTrack**: 고성능 다중 객체 추적
- **칼만필터**: Bbox 안정화
- **하이브리드 예측**: EMA + 대기열 보정 + IQR 이상치 제거
- **ROI 체류 필터**: 오카운팅 방지

### ☁️ 클라우드 연동 (Phase 6) 🆕
- **비동기 배치 전송**: 최대 25개 아이템
- **자동 재시도**: Exponential backoff
- **데이터 변환**: snake_case ↔ camelCase
- **TTL 자동 설정**: 3일 후 자동 삭제

### 🛡️ 안정성
- **Graceful Degradation**: DynamoDB 오류 시에도 시스템 정상 동작
- **스레드 안전**: 모든 공유 자원은 Lock으로 보호
- **에러 복구**: 자동 재시도 및 큐 재적재
- **모니터링**: 전송 통계 API 제공

---

## 📄 라이선스

MIT License

---

## 📚 참고 문서

상세한 아키텍처 설계 및 개발 계획은 다음 문서 참조:
- `식당_대기시간_추정_시스템_설계서_v2.pdf`
- [빠른 실행 가이드](QUICKSTART.md)
- [폴더 구조 가이드](FOLDER_GUIDE.md)
- [비디오 파일 테스트 가이드](test/README.md)
- [DynamoDB 테스트 가이드](TEST_DYNAMODB.md)
- [3-Thread 아키텍처 가이드](docs/3-Thread_Architecture_Guide.md)
- [대기시간 알고리즘 가이드](docs/Phase5_대기시간_알고리즘_가이드.md)
