import argparse
import logging
import os
import signal
import sys

from src.core.camera import CameraManager
from src.core.detector import YOLOv8Detector
from src.core.roi_manager import ROIManager
from src.core.tracker import ByteTracker
from src.web.app import app, init_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='식당 대기시간 추정 시스템')
    parser.add_argument('--host', default='0.0.0.0', help='서버 바인드 주소')
    parser.add_argument('--port', type=int, default=5000, help='서버 포트')
    parser.add_argument('--capture-width', type=int, default=1280,
                        help='카메라 캡처 해상도 너비')
    parser.add_argument('--capture-height', type=int, default=720,
                        help='카메라 캡처 해상도 높이')
    parser.add_argument('--display-width', type=int, default=640,
                        help='스트리밍 표시 해상도 너비')
    parser.add_argument('--display-height', type=int, default=480,
                        help='스트리밍 표시 해상도 높이')
    parser.add_argument('--fps', type=int, default=30, help='카메라 FPS')
    parser.add_argument('--sensor-id', type=int, default=0, help='CSI 센서 ID')
    parser.add_argument('--flip', type=int, default=0,
                        help='영상 회전 (0=없음, 2=180도)')
    parser.add_argument('--model', type=str, default='models/yolov8n.onnx',
                        help='YOLOv8 ONNX 모델 경로')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='검출 신뢰도 임계값')
    parser.add_argument('--no-fp16', action='store_true',
                        help='FP16 추론 비활성화 (FP32 사용)')
    parser.add_argument('--roi-config', type=str, default='config/roi_config.json',
                        help='ROI 설정 파일 경로')
    parser.add_argument('--inference-fps', type=int, default=None,
                        help='Inference FPS 제한 (기본값: 카메라 FPS와 동일)')
    parser.add_argument('--max-age', type=int, default=30,
                        help='추적기: 미매칭 트랙 유지 프레임 수')
    parser.add_argument('--min-hits', type=int, default=3,
                        help='추적기: 출력 최소 매칭 횟수')
    parser.add_argument('--min-dwell', type=int, default=30,
                        help='ROI 최소 체류 프레임 수 (기본 30 ≈ 1초@30fps)')
    return parser.parse_args()


def main():
    args = parse_args()

    camera = CameraManager(
        sensor_id=args.sensor_id,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        display_width=args.display_width,
        display_height=args.display_height,
        fps=args.fps,
        flip_method=args.flip,
    )

    # 검출기 초기화 (모델 파일이 존재하는 경우)
    detector = None
    if args.model and os.path.exists(args.model):
        detector = YOLOv8Detector(
            model_path=args.model,
            conf_threshold=args.conf_threshold,
            fp16=not args.no_fp16,
            target_classes=[0],  # person만 검출
        )
        if not detector.initialize():
            logger.warning("검출기 초기화 실패. 검출 없이 스트리밍만 진행합니다.")
            detector = None

    # 추적기 초기화
    tracker = ByteTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        high_thresh=args.conf_threshold,
    )
    logger.info(f"ByteTracker 초기화 (max_age={args.max_age}, min_hits={args.min_hits})")

    # ROI 매니저 초기화
    roi_mgr = ROIManager(config_path=args.roi_config)
    roi_mgr.load()

    def shutdown(signum, frame):
        logger.info("종료 시그널 수신, 정리 중...")
        camera.stop()
        if detector is not None:
            detector.destroy()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if not camera.start():
        logger.error("카메라 시작 실패. 종료합니다.")
        sys.exit(1)

    init_app(camera, detector, roi_mgr, tracker,
             inference_fps=args.inference_fps or args.fps,
             min_dwell_frames=args.min_dwell)

    logger.info(f"서버 시작: http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        camera.stop()
        if detector is not None:
            detector.destroy()


if __name__ == '__main__':
    main()
