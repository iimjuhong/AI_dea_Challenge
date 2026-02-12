import cv2
import threading
import logging
import time
import subprocess

logger = logging.getLogger(__name__)


class CameraManager:
    """CSI 카메라(Arducam IMX219) 관리 - GStreamer 파이프라인 사용

    Jetson 최적화:
    - 캡처 해상도와 스트리밍 해상도 분리
    - nvjpegenc 하드웨어 JPEG 인코딩 지원
    - get_jpeg_frame()으로 CPU imencode 우회 가능
    """

    def __init__(self, sensor_id=0, capture_width=1280, capture_height=720,
                 display_width=640, display_height=480, fps=30,
                 flip_method=0, use_hw_encode=True):
        self.sensor_id = sensor_id
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.display_width = display_width
        self.display_height = display_height
        self.fps = fps
        self.flip_method = flip_method
        self.use_hw_encode = use_hw_encode

        self._cap = None
        self._frame = None
        self._jpeg_buf = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._hw_encode_available = False
        if use_hw_encode:
            self._hw_encode_available = self._check_nvjpegenc()

    @staticmethod
    def _check_nvjpegenc():
        """nvjpegenc (HW JPEG 인코더) 사용 가능 여부 확인"""
        try:
            result = subprocess.run(
                ['gst-inspect-1.0', 'nvjpegenc'],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _build_gstreamer_pipeline(self):
        """Jetson용 nvarguscamerasrc GStreamer 파이프라인 생성

        캡처는 고해상도, 출력은 display 해상도로 스케일링
        """
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), "
            f"width=(int){self.capture_width}, height=(int){self.capture_height}, "
            f"framerate=(fraction){self.fps}/1, format=(string)NV12 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width=(int){self.display_width}, height=(int){self.display_height}, "
            f"format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink drop=1"
        )

    def _build_jpeg_pipeline(self):
        """HW JPEG 인코딩 GStreamer 파이프라인 (nvjpegenc 활용)"""
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), "
            f"width=(int){self.capture_width}, height=(int){self.capture_height}, "
            f"framerate=(fraction){self.fps}/1, format=(string)NV12 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.display_width}, "
            f"height=(int){self.display_height}, format=(string)I420 ! "
            f"nvjpegenc quality=80 ! "
            f"appsink drop=1"
        )

    def start(self):
        """카메라 캡처 시작"""
        if self._running:
            logger.warning("카메라가 이미 실행 중입니다")
            return True

        pipeline = self._build_gstreamer_pipeline()
        logger.info(f"GStreamer 파이프라인: {pipeline}")

        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            logger.error("CSI 카메라를 열 수 없습니다. USB 카메라로 폴백합니다.")
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                logger.error("카메라를 열 수 없습니다")
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("카메라 캡처 시작")

        if self._hw_encode_available:
            logger.info("nvjpegenc HW JPEG 인코딩 사용 가능")
        else:
            logger.info("SW JPEG 인코딩 사용 (nvjpegenc 미지원)")

        return True

    def _capture_loop(self):
        """백그라운드 프레임 캡처 루프"""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("프레임 읽기 실패")
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def get_frame(self):
        """최신 프레임 반환 (thread-safe, BGR numpy array)"""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_jpeg_frame(self, quality=80):
        """최신 프레임을 JPEG 바이트로 반환

        MJPEG 스트리밍 시 cv2.imencode 대신 직접 사용 가능.
        향후 nvjpegenc 파이프라인 연동 시 zero-copy 전환 가능.

        Returns:
            bytes or None: JPEG 인코딩된 프레임 바이트
        """
        frame = self.get_frame()
        if frame is None:
            return None

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            return None
        return buf.tobytes()

    def stop(self):
        """카메라 캡처 중지 및 리소스 해제"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame = None
        logger.info("카메라 중지 완료")

    @property
    def is_running(self):
        return self._running

    def __del__(self):
        self.stop()
