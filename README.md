# 식당 대기시간 추정 시스템

NVIDIA Jetson Orin Super Nano 기반 실시간 식당 대기열 추적 및 대기시간 추정 시스템

## 프로젝트 개요

- **목표**: 객체 추적 기반 실제 대기시간 측정 및 예측
- **디바이스**: NVIDIA Jetson Orin Super Nano (16GB)
- **카메라**: Arducam IMX219 (CSI)
- **OS**: Ubuntu 22.04 (JetPack 6.2.2)

---

## ✨ 현재 구현 상태

### ✅ Phase 1-2: 카메라 스트리밍 + YOLO 검출 (완료)
- **실시간 카메라 스트리밍**: GStreamer 파이프라인을 통한 CSI 카메라 영상 캡처
- **YOLO 객체 검출**: TensorRT 기반 YOLOv8 FP16 추론 (15-20 FPS)
- **사람 검출 및 시각화**: Bounding box 오버레이 및 신뢰도 표시
- **웹 인터페이스**: Flask 기반 MJPEG 스트리밍

### ✅ Phase 3: ROI 관리 시스템 (완료)
- **웹 기반 ROI 설정**: 브라우저에서 마우스로 다각형 영역 그리기
  - 좌클릭: 꼭짓점 추가
  - 우클릭: 완성
- **JSON 설정 저장/불러오기**: `config/roi_config.json`에 영구 저장
- **ROI별 인원 카운팅**: Point-in-Polygon 알고리즘으로 영역별 사람 수 실시간 표시
- **다중 ROI 지원**: 여러 영역 동시 관리 (입구, 대기 구역, 카운터 등)

### ✅ Phase 4: 객체 추적 + 칼만필터 (완료)
- **ByteTrack 기반 다중 객체 추적**: 2단계 연관 매칭 (고/저신뢰도 분리)
- **칼만필터 bbox 안정화**: 7차원 상태 벡터, 등속 모델
- **고유 ID 부여 및 유지**: track_id 단조 증가, 일시적 가림 대응
- **ROI 체류시간 필터**: 체류 프레임 수 추적으로 오카운팅 방지

### ✅ Phase 5: 대기시간 측정 및 예측 (완료)
- **ROI 진입/퇴출 이벤트 감지**: 상태 전이 기반 실시간 감지
- **실제 대기시간 계산**: track_id별 ROI 플로우 추적 (진입→퇴출)
- **하이브리드 예측 알고리즘**: EMA + 대기열 크기 보정 (IQR 이상치 제거)
- **듀얼 모드 지원**: 단일 ROI / 플로우 모드 선택 가능

### 🚧 Phase 6: 웹 대시보드 (예정)
- 실시간 통계 차트 (Chart.js)
- 시간대별 분석

### 🚧 Phase 7: DynamoDB 연동 (예정)
- AWS DynamoDB 데이터 영속화
- 비동기 배치 쓰기 (boto3/aioboto3)

---

## 🏗️ 시스템 아키텍처

### 전체 파이프라인
```
┌─────────────────────────────────────────────────────────────┐
│                    CSI 카메라 (IMX219)                       │
│                   1280x720 @ 30fps                          │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              GStreamer 파이프라인                            │
│  nvarguscamerasrc → nvvidconv → 640x480 → BGR               │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│           CameraManager (백그라운드 스레드)                   │
│  - 프레임 지속 캡처 (thread-safe)                             │
│  - HW JPEG 인코딩 지원 (nvjpegenc)                           │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│         YOLOv8Detector (TensorRT FP16 추론)                  │
│  - ONNX → TensorRT 엔진 자동 변환 및 캐싱                     │
│  - ctypes CUDA 직접 호출 (PyCUDA 불필요)                     │
│  - NMS 후처리 (person만 검출)                                │
│  - FPS: 15-20                                               │
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
┌─────────────────────────────────────────────────────────────┐
│              Flask 웹 서버 (0.0.0.0:5000)                    │
│  - MJPEG 스트리밍 (/video_feed)                             │
│  - 실시간 통계 API (/api/stats)                              │
│  - ROI CRUD API (/api/roi)                                 │
│  - 대기시간 예측 API (/api/wait_time)                        │
└─────────────────────────────────────────────────────────────┘
                       ↓
              브라우저 (웹 UI)
```

### 주요 컴포넌트

#### 1. **CameraManager** (`src/core/camera.py`)
```python
- GStreamer 파이프라인 구성
- 백그라운드 스레드에서 프레임 캡처
- thread-safe 프레임 접근
- HW JPEG 인코딩 지원
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

## 🚀 사용 방법

### 1. 프로그램 실행
```bash
cd /home/iimjuhong/projects/aidea
python3 main.py
```

**옵션:**
```bash
# 해상도 변경
python3 main.py --display-width 1280 --display-height 720

# FPS 변경
python3 main.py --fps 30

# 모델 경로 지정
python3 main.py --model models/yolov8n.onnx

# 신뢰도 임계값 조정
python3 main.py --conf-threshold 0.6

# 모든 옵션 보기
python3 main.py --help
```

### 2. 웹 UI 접속
브라우저에서 다음 주소로 접속:
- `http://localhost:5000` (같은 기기)
- `http://[Jetson IP]:5000` (네트워크 접속)

### 3. ROI 영역 설정
1. **ROI 이름 입력** (예: "대기구역")
2. **"그리기 시작" 클릭**
3. **좌클릭**으로 꼭짓점 추가 (최소 3개)
4. **우클릭**으로 완성
5. ROI 목록에서 영역별 실시간 인원 수 확인

### 4. 종료
터미널에서 `Ctrl + C`

---

## 📂 프로젝트 구조

```
aidea/
├── main.py                          # 메인 진입점
├── config/
│   └── roi_config.json             # ROI 설정 저장
├── models/
│   ├── yolov8n.onnx                # YOLO 모델
│   └── yolov8n_fp16.engine         # TensorRT 엔진 캐시
├── src/
│   ├── core/                       # 핵심 로직
│   │   ├── camera.py              # 카메라 관리자
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
└── requirements.txt               # Python 의존성
```

---

## 🛠️ 기술 스택

### Core
- **Python 3.10**
- **OpenCV** (GStreamer 지원)
- **YOLOv8** (Ultralytics ONNX)
- **TensorRT** (FP16 추론)
- **Flask** (웹 서버)

### 하드웨어 최적화
- **GStreamer**: nvarguscamerasrc, nvvidconv
- **TensorRT**: FP16 엔진, CUDA 메모리 직접 관리
- **nvjpegenc**: 하드웨어 JPEG 인코딩

### 추가 예정
- **boto3/aioboto3**: AWS DynamoDB 연동
- **Chart.js**: 데이터 시각화
- **WebSocket**: 실시간 양방향 통신

---

## 📊 성능 지표

### 현재 성능
- **FPS**: 15-20 (YOLO + ROI 처리)
- **추론 시간**: ~50-60ms/frame
- **메모리 사용량**: ~2GB (TensorRT 엔진 포함)
- **지연 시간**: <100ms (카메라 → 웹 UI)

### 최적화 요소
- ✅ TensorRT FP16 모드 (2배 속도 향상)
- ✅ 엔진 캐싱 (재시작 시 즉시 로드)
- ✅ Zero-copy GStreamer 파이프라인
- ✅ 하드웨어 JPEG 인코딩

---

## 🎯 핵심 기능

### 1. 실시간 사람 검출
- YOLOv8 TensorRT 최적화
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

## 📝 개발 로드맵

- [x] **Phase 1-2**: 카메라 스트리밍 + YOLO 검출
- [x] **Phase 3**: ROI 관리 시스템
- [x] **Phase 4**: 객체 추적 + 칼만필터 오차 보정
- [x] **Phase 5**: 대기시간 측정 및 예측
- [ ] **Phase 6**: 웹 대시보드 완성
- [ ] **Phase 7**: DynamoDB 연동

---

## 📄 라이선스

MIT License

---

## 📚 참고 문서

상세한 아키텍처 설계 및 개발 계획은 다음 문서 참조:
- `식당_대기시간_추정_시스템_설계서_v2.pdf`
- 향후 Phase 계획: 별도 문서 참조
