"""비디오 파일로 YOLO 검출 파이프라인 테스트

카메라 없이 녹화 영상으로 검출/추적/ROI/DynamoDB 전체 파이프라인을 실행.
웹 브라우저에서 실시간 결과 확인 가능.

사용법:
    python test/run_video.py --video test/sample.mp4 --model models/yolov8s.onnx --port 5000
"""
import argparse
import logging
import os
import signal
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.video_source import VideoFileManager
from src.core.detector import YOLOv8Detector
from src.core.roi_manager import ROIManager
from src.core.tracker import ByteTracker
from src.core.wait_time_estimator import WaitTimeEstimator
from src.web.app import app, init_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='비디오 파일로 검출 파이프라인 테스트'
    )
    parser.add_argument('--video', type=str, required=True,
                        help='테스트용 비디오 파일 경로')
    parser.add_argument('--host', default='0.0.0.0', help='서버 바인드 주소')
    parser.add_argument('--port', type=int, default=5000, help='서버 포트')
    parser.add_argument('--display-width', type=int, default=640,
                        help='스트리밍 표시 해상도 너비')
    parser.add_argument('--display-height', type=int, default=480,
                        help='스트리밍 표시 해상도 높이')
    parser.add_argument('--model', type=str, default='models/yolov8s.onnx',
                        help='YOLOv8 ONNX 모델 경로')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='검출 신뢰도 임계값')
    parser.add_argument('--no-fp16', action='store_true',
                        help='FP16 추론 비활성화 (FP32 사용)')
    parser.add_argument('--roi-config', type=str, default='config/roi_config.json',
                        help='ROI 설정 파일 경로')
    parser.add_argument('--inference-fps', type=int, default=None,
                        help='Inference FPS 제한 (기본값: 비디오 FPS)')
    parser.add_argument('--max-age', type=int, default=30,
                        help='추적기: 미매칭 트랙 유지 프레임 수')
    parser.add_argument('--min-hits', type=int, default=3,
                        help='추적기: 출력 최소 매칭 횟수')
    parser.add_argument('--min-dwell', type=int, default=30,
                        help='ROI 최소 체류 프레임 수')
    parser.add_argument('--aws-config', type=str, default='config/aws_config.json',
                        help='AWS DynamoDB 설정 파일 경로')
    parser.add_argument('--no-dynamodb', action='store_true',
                        help='DynamoDB 전송 비활성화')
    parser.add_argument('--start-roi', type=str, default=None,
                        help='대기시간 측정 시작 ROI 이름')
    parser.add_argument('--end-roi', type=str, default=None,
                        help='대기시간 측정 종료 ROI 이름')
    parser.add_argument('--no-loop', action='store_true',
                        help='비디오 반복 재생 비활성화')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        logger.error(f"비디오 파일을 찾을 수 없습니다: {args.video}")
        sys.exit(1)

    # 비디오 소스 초기화 (CameraManager 대체)
    video_src = VideoFileManager(
        video_path=args.video,
        display_width=args.display_width,
        display_height=args.display_height,
        loop=not args.no_loop,
    )

    # 검출기 초기화
    detector = None
    if args.model and os.path.exists(args.model):
        detector = YOLOv8Detector(
            model_path=args.model,
            conf_threshold=args.conf_threshold,
            fp16=not args.no_fp16,
            target_classes=[0],
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

    # WaitTimeEstimator 초기화
    wait_estimator = None
    if args.start_roi:
        wait_estimator = WaitTimeEstimator(
            start_roi=args.start_roi,
            end_roi=args.end_roi,
            predictor_type='hybrid',
        )
        logger.info(f"WaitTimeEstimator 초기화: start={args.start_roi}, end={args.end_roi}")

    # DynamoDB 전송기 초기화
    dynamodb_sender = None
    if not args.no_dynamodb and os.path.exists(args.aws_config):
        try:
            from src.cloud.dynamodb_sender import DynamoDBSender
            dynamodb_sender = DynamoDBSender(config_path=args.aws_config)
            dynamodb_sender.start()
            logger.info("DynamoDB 전송기 초기화 완료")
        except Exception as e:
            logger.warning(f"DynamoDB 전송기 초기화 실패 (전송 없이 계속): {e}")

    def shutdown(signum, frame):
        logger.info("종료 시그널 수신, 정리 중...")
        if dynamodb_sender is not None:
            dynamodb_sender.stop()
        video_src.stop()
        if detector is not None:
            detector.destroy()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if not video_src.start():
        logger.error("비디오 소스 시작 실패. 종료합니다.")
        sys.exit(1)

    # 비디오 FPS를 inference FPS 기본값으로 사용
    inference_fps = args.inference_fps or int(video_src._fps)

    init_app(video_src, detector, roi_mgr, tracker,
             inference_fps=inference_fps,
             min_dwell_frames=args.min_dwell,
             wait_estimator=wait_estimator,
             dynamodb_sender=dynamodb_sender)

    logger.info(f"서버 시작: http://{args.host}:{args.port}")
    logger.info(f"비디오: {args.video} (loop={'on' if not args.no_loop else 'off'})")
    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port,
              threads=8,
              channel_timeout=120,
              recv_bytes=65536,
              send_bytes=262144,
              log_socket_errors=False)
    except ImportError:
        logger.warning("waitress 미설치 — Flask 개발 서버 사용 (pip install waitress 권장)")
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        if dynamodb_sender is not None:
            dynamodb_sender.stop()
        video_src.stop()
        if detector is not None:
            detector.destroy()


if __name__ == '__main__':
    main()
