# 식당 대기시간 추정 시스템

NVIDIA Jetson Orin Super Nano 기반 실시간 식당 대기열 추적 및 대기시간 추정 시스템

## 프로젝트 개요

- **목표**: 객체 추적 기반 실제 대기시간 측정 및 예측
- **디바이스**: NVIDIA Jetson Orin Super Nano (16GB)
- **카메라**: Arducam IMX219 (CSI)
- **OS**: Ubuntu 22.04 (JetPack 6.2.2)

## 핵심 기능

1. **실시간 카메라 스트리밍** - CSI 카메라 영상을 웹 브라우저로 전송
2. **ROI 기반 영역 관리** - 대기 줄 영역 설정 및 관리
3. **사람 검출** - YOLOv8로 ROI 내 사람 검출
4. **객체 추적** - DeepSORT/ByteTrack으로 개인별 ID 부여 및 추적
5. **실제 대기시간 측정** - ROI 진입/퇴출 시간 기록
6. **대기시간 예측** - 실제 측정 데이터 기반 예상 대기시간 계산
7. **웹 대시보드** - 실시간 영상 + 대기시간 + 통계
8. **비동기 DB 저장** - 로컬 SQLite + 원격 DB 동기화

## 기술 스택

### Core
- **Python 3.10**
- **OpenCV** (GStreamer 지원)
- **YOLOv8** (Ultralytics)
- **DeepSORT** 또는 **ByteTrack** (객체 추적)
- **Flask** (웹 스트리밍)

### Database
- **SQLite** (로컬 버퍼)
- **PostgreSQL/MySQL** (원격 DB, 옵션)

## 시스템 아키텍처

```
카메라 → YOLO 검출 → 객체 추적 → ROI 분석 → 대기시간 계산
                                              ↓
                                        웹 스트리밍
                                              ↓
                                    비동기 DB 큐 → SQLite → 원격 DB
```

## 프로젝트 구조

```
aidea/
├── config/              # 설정 파일
├── src/                 # 소스 코드
│   ├── core/           # 핵심 비즈니스 로직
│   │   ├── camera.py           # 카메라 관리
│   │   ├── detector.py         # YOLO 검출
│   │   ├── tracker.py          # 객체 추적 (DeepSORT)
│   │   ├── roi_manager.py      # ROI 관리
│   │   ├── queue_analyzer.py   # 대기열 분석
│   │   ├── db_manager.py       # DB 비동기 처리
│   │   └── data_logger.py      # 데이터 로깅
│   ├── web/            # 웹 인터페이스
│   ├── utils/          # 유틸리티 모듈
│   └── models/         # 데이터 모델
├── models/             # AI 모델 파일
├── db/                 # SQLite 데이터베이스
├── logs/               # 로그 파일
├── data/               # 데이터 저장소
├── tests/              # 테스트 코드
└── scripts/            # 유틸리티 스크립트
```

## 작동 원리

### 1. 객체 추적
- YOLO로 사람 검출
- DeepSORT로 각 사람에게 고유 ID 부여
- 프레임마다 ID 유지 추적

### 2. 대기시간 측정
- ROI 진입 시간 기록 (ID별)
- ROI 퇴출 시간 기록
- 실제 대기시간 = 퇴출 - 진입

### 3. 대기시간 예측
- 최근 N명의 평균 대기시간 계산
- 현재 대기 인원 × 평균 대기시간
- 시간대별 가중치 적용

### 4. 비동기 DB 처리
- 실시간 처리는 메모리에서
- 백그라운드 큐로 DB 저장
- Latency 없음

## 개발 단계

- [ ] **Phase 1**: 카메라 스트리밍
- [ ] **Phase 2**: YOLO 객체 검출
- [ ] **Phase 3**: ROI 관리 시스템
- [ ] **Phase 4**: 객체 추적 (DeepSORT)
- [ ] **Phase 5**: 대기시간 측정 및 예측
- [ ] **Phase 6**: 웹 대시보드
- [ ] **Phase 7**: DB 연동 (로컬 + 원격)

## 성능 목표

- **FPS**: 15-20 (YOLO + Tracking)
- **정확도**: 대기시간 ±10% 이내
- **DB Latency**: 0 (비동기 처리)

## 장점

✅ **실제 데이터 기반** - 가정 없이 측정  
✅ **자동 보정** - 시간대별 서빙 속도 변화 반영  
✅ **높은 정확도** - 현재 대기줄 속도로 예측  
✅ **낮은 성능 부담** - Depth Map보다 가벼움  

## 라이선스

MIT License

## 참고 문서

상세한 아키텍처 설계는 별도 PDF 문서 참조
