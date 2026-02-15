# 📁 프로젝트 폴더 구조 가이드

> 각 폴더와 파일의 역할을 한눈에 파악하세요!

---

## 🌳 전체 구조 한눈에 보기

```
aidea/
├── 📄 main.py                    # 프로그램 진입점 (여기서 시작!)
├── 📄 requirements.txt           # Python 패키지 의존성
├── 📄 QUICKSTART.md             # 빠른 실행 가이드
├── 📄 README.md                 # 전체 프로젝트 문서
│
├── 📂 src/                      # 핵심 소스 코드
│   ├── core/                    # 핵심 기능 (YOLO, 추적, ROI)
│   ├── cloud/                   # AWS DynamoDB 연동
│   ├── web/                     # Flask 웹 서버
│   ├── models/                  # 데이터 모델
│   └── utils/                   # 유틸리티 함수
│
├── 📂 config/                   # 설정 파일들
├── 📂 models/                   # YOLO 모델 파일
├── 📂 data/                     # 데이터 저장소 (스냅샷, 통계)
├── 📂 scripts/                  # 설치/실행 스크립트
├── 📂 tests/                    # 단위 테스트
├── 📂 docs/                     # 추가 문서
├── 📂 frontend/                 # 웹 프론트엔드 (React)
├── 📂 db/                       # 로컬 데이터베이스
├── 📂 logs/                     # 로그 파일
└── 📂 venv/                     # Python 가상환경 (직접 생성)
```

---

## 📂 주요 폴더 상세 설명

### 1. `src/` - 핵심 소스 코드

프로젝트의 모든 핵심 로직이 들어있는 메인 폴더입니다.

#### 📂 `src/core/` - 핵심 AI 기능
```
core/
├── camera.py              # 📷 카메라 관리 (GStreamer 파이프라인)
├── detector.py            # 🔍 YOLO 객체 검출 (TensorRT)
├── tracker.py             # 🎯 ByteTrack 다중 객체 추적 + 칼만필터
├── roi_manager.py         # 🗺️  ROI(관심영역) 관리
├── wait_time_estimator.py # ⏱️  대기시간 계산 및 예측
└── frame_buffer.py        # 🖼️  프레임 버퍼 (논블로킹)
```

**역할**:
- `camera.py`: CSI 카메라에서 영상 받아오기 (백그라운드 스레드)
- `detector.py`: YOLO로 사람 검출 (TensorRT 가속)
- `tracker.py`: 검출된 사람들 추적 (track_id 부여)
- `roi_manager.py`: "대기구역", "카운터" 같은 영역 관리
- `wait_time_estimator.py`: ROI 진입→퇴출 기반 대기시간 계산
- `frame_buffer.py`: 웹 스트리밍용 프레임 버퍼

#### 📂 `src/cloud/` - AWS 클라우드 연동
```
cloud/
├── __init__.py
└── dynamodb_sender.py    # ☁️  DynamoDB 비동기 전송
```

**역할**:
- DynamoDB로 대기시간 데이터 전송 (배치 처리, 재시도 로직)

#### 📂 `src/web/` - Flask 웹 서버
```
web/
├── app.py                # 🌐 Flask 앱 (API + MJPEG 스트림)
├── static/               # 정적 파일 (CSS, JS, 이미지)
└── templates/            # HTML 템플릿
```

**역할**:
- API 엔드포인트 제공 (`/api/stats`, `/api/wait_time` 등)
- MJPEG 비디오 스트리밍 (`/video_feed`)
- ROI 관리 웹 UI

#### 📂 `src/models/` - 데이터 모델
```
models/
└── .gitkeep
```

**역할**: 데이터 클래스 정의 (현재 비어있음)

#### 📂 `src/utils/` - 유틸리티
```
utils/
└── (빈 폴더)
```

**역할**: 공통 헬퍼 함수들

---

### 2. `config/` - 설정 파일

```
config/
├── aws_config.json       # ☁️  AWS DynamoDB 설정
└── roi_config.json       # 🗺️  ROI 영역 좌표 (웹 UI에서 자동 생성)
```

**역할**:
- `aws_config.json`: DynamoDB 테이블 이름, 리전, 식당/코너 ID
- `roi_config.json`: 웹 UI에서 그린 ROI 영역 좌표 저장

**예시**:
```json
// aws_config.json
{
  "region": "ap-northeast-2",
  "table_name": "hyeat-waiting-data-dev",
  "restaurant_id": "hanyang_plaza",
  "corner_id": "western"
}
```

---

### 3. `models/` - AI 모델 파일

```
models/
└── yolov8n.onnx          # 🤖 YOLO 모델 (ONNX 형식)
    yolov8n.engine        # ⚡ TensorRT 엔진 (자동 생성)
```

**역할**:
- YOLO 모델 저장소
- `.onnx`: 원본 모델
- `.engine`: TensorRT로 최적화된 모델 (첫 실행 시 자동 생성)

---

### 4. `data/` - 데이터 저장소

```
data/
├── snapshots/            # 📸 스냅샷 이미지 (Phase 7 예정)
└── statistics/           # 📊 통계 데이터 (Phase 7 예정)
```

**역할**:
- 현재는 비어있음
- 향후 로컬 데이터 백업용

---

### 5. `scripts/` - 스크립트

```
scripts/
├── download_model.sh     # 📥 YOLO 모델 다운로드
└── setup_env.sh          # ⚙️  환경 설정
```

**역할**:
- 자동화 스크립트 모음
- 모델 다운로드, 환경 설정 등

**사용 예시**:
```bash
bash scripts/download_model.sh
```

---

### 6. `tests/` - 테스트 코드

```
tests/
├── test_dynamodb_sender.py   # ☁️  DynamoDB 전송 테스트
└── test_*.py                  # 기타 단위 테스트
```

**역할**:
- 코드 품질 검증
- 버그 방지

**실행 방법**:
```bash
pytest tests/
```

---

### 7. `docs/` - 문서

```
docs/
└── Phase5_대기시간_알고리즘_가이드.md
```

**역할**:
- 추가 기술 문서
- 알고리즘 상세 설명

---

### 8. `frontend/` - 웹 프론트엔드

```
frontend/
└── src/
    └── types/
        └── hyeat.ts      # TypeScript 타입 정의
```

**역할**:
- React 웹 대시보드 (Phase 7 예정)
- TypeScript 인터페이스 정의

---

### 9. `db/` - 로컬 데이터베이스

```
db/
└── .gitkeep
```

**역할**:
- SQLite 등 로컬 DB 파일 저장 (Phase 7 예정)

---

### 10. `logs/` - 로그 파일

```
logs/
└── (자동 생성)
```

**역할**:
- 프로그램 실행 로그 자동 저장

---

## 🔥 자주 수정하는 파일들

### 💻 코드 수정 시:

| 파일 | 수정하는 경우 |
|------|--------------|
| `main.py` | 프로그램 전체 흐름 변경 |
| `src/core/detector.py` | YOLO 모델 교체, 검출 임계값 조정 |
| `src/core/wait_time_estimator.py` | 대기시간 알고리즘 변경 |
| `src/cloud/dynamodb_sender.py` | AWS 전송 로직 수정 |
| `src/web/app.py` | API 추가, 웹 UI 수정 |

### ⚙️ 설정 변경 시:

| 파일 | 수정하는 경우 |
|------|--------------|
| `config/aws_config.json` | DynamoDB 테이블 변경, 코너 ID 변경 |
| `config/roi_config.json` | ROI 영역 수동 수정 (보통 웹 UI 사용) |
| `requirements.txt` | 새 패키지 추가 |

---

## 🚫 건드리지 말아야 할 것들

- `venv/` - 가상환경 (삭제 후 재생성 가능)
- `__pycache__/` - Python 캐시 (자동 생성)
- `.pytest_cache/` - pytest 캐시
- `models/yolov8n.engine` - TensorRT 엔진 (자동 생성)
- `.git/` - Git 버전 관리

---

## 🎯 코드 찾기 팁

### "카메라가 안 켜져요" 🔧
→ `src/core/camera.py` 확인

### "YOLO 검출이 이상해요" 🔧
→ `src/core/detector.py` 확인

### "ROI 설정이 안 돼요" 🔧
→ `src/core/roi_manager.py` + `config/roi_config.json` 확인

### "대기시간 계산이 이상해요" 🔧
→ `src/core/wait_time_estimator.py` 확인

### "AWS 전송이 안 돼요" 🔧
→ `src/cloud/dynamodb_sender.py` + `config/aws_config.json` 확인

### "웹 UI가 안 떠요" 🔧
→ `src/web/app.py` 확인

---

## 📚 더 알아보기

- **빠른 실행**: [QUICKSTART.md](QUICKSTART.md)
- **전체 문서**: [README.md](README.md)
- **알고리즘 상세**: [docs/Phase5_대기시간_알고리즘_가이드.md](docs/Phase5_대기시간_알고리즘_가이드.md)

---

## 💡 폴더 구조 원칙

1. **`src/`**: 모든 Python 소스 코드
2. **`config/`**: 설정 파일 (JSON)
3. **`models/`**: AI 모델 파일
4. **`data/`**: 데이터 저장소
5. **`docs/`**: 문서
6. **루트**: 진입점(`main.py`), 의존성(`requirements.txt`), 가이드 문서

이렇게 분리하면 코드 찾기 쉽고, 유지보수가 편합니다! 🎉
