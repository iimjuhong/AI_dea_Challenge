# 식당 대기시간 추정 시스템

NVIDIA Jetson Orin Super Nano 기반 실시간 식당 혼잡도 분석 및 대기시간 추정 시스템

## 프로젝트 개요

- **목표**: ROI 기반 대기 줄 자동 분석 및 대기시간 추정
- **디바이스**: NVIDIA Jetson Orin Super Nano (16GB)
- **카메라**: Arducam IMX219 (CSI)
- **OS**: Ubuntu 22.04 (JetPack 6.2.2)

## 핵심 기능

1. **실시간 카메라 스트리밍** - CSI 카메라 영상을 웹 브라우저로 전송
2. **ROI 기반 영역 관리** - 대기 줄 영역 설정 및 관리
3. **사람 검출** - YOLOv8로 ROI 내 사람 검출
4. **Depth Map 생성** - MiDaS로 거리 정보 추정
5. **대기열 분석** - 깊이 정보 기반 대기 순서 파악
6. **대기시간 추정** - 평균 서빙 속도 기반 대기시간 계산
7. **웹 대시보드** - 실시간 영상 + 대기시간 + 통계
8. **데이터 로깅** - 시간대별 대기 인원 및 대기시간 기록

## 기술 스택

- **Python 3.10**
- **OpenCV** (GStreamer 지원)
- **YOLOv8** (Ultralytics)
- **MiDaS** (Depth Estimation)
- **Flask** (웹 스트리밍)

## 프로젝트 구조

```
aidea/
├── config/              # 설정 파일
├── src/                 # 소스 코드
│   ├── core/           # 핵심 비즈니스 로직
│   ├── web/            # 웹 인터페이스
│   ├── utils/          # 유틸리티 모듈
│   └── models/         # 데이터 모델
├── models/             # AI 모델 파일
├── logs/               # 로그 파일
├── data/               # 데이터 저장소
├── tests/              # 테스트 코드
└── scripts/            # 유틸리티 스크립트
```

## 개발 단계

- [ ] Phase 1: 카메라 스트리밍
- [ ] Phase 2: YOLO 객체 인식
- [ ] Phase 3: ROI 관리 시스템
- [ ] Phase 4: Depth Estimation
- [ ] Phase 5: 대기열 분석 및 대기시간 추정
- [ ] Phase 6: 웹 대시보드 구축
- [ ] Phase 7: 데이터 로깅 및 분석

## 참고 문서

상세한 아키텍처 설계는 별도 PDF 문서 참조

## 라이선스

MIT License
