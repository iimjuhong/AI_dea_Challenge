import cv2
import threading
import logging
import time

logger = logging.getLogger(__name__)


class VideoFileManager:
    """비디오 파일 소스 — CameraManager와 동일 인터페이스

    카메라 없이 녹화 영상으로 검출 파이프라인을 테스트할 때 사용.
    CameraManager를 대체(drop-in)할 수 있도록 start(), stop(),
    get_frame(), is_running 인터페이스를 동일하게 제공.
    """

    def __init__(self, video_path, display_width=640, display_height=480,
                 loop=True):
        self.video_path = video_path
        self.display_width = display_width
        self.display_height = display_height
        self.loop = loop

        self._cap = None
        self._fps = 30.0
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """비디오 파일 열기 및 프레임 읽기 스레드 시작"""
        if self._running:
            logger.warning("VideoFileManager가 이미 실행 중입니다")
            return True

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없습니다: {self.video_path}")
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"비디오 열기: {self.video_path} "
            f"({width}x{height} @ {self._fps:.1f}fps, {total}프레임)"
        )

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("비디오 프레임 읽기 스레드 시작")
        return True

    def _read_loop(self):
        """백그라운드 프레임 읽기 루프 (원본 FPS에 맞춰 pacing)"""
        interval = 1.0 / self._fps

        while self._running:
            t0 = time.monotonic()

            ret, frame = self._cap.read()
            if not ret:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.info("비디오 끝 — 처음부터 반복 재생")
                    continue
                else:
                    logger.info("비디오 끝 — 재생 종료")
                    self._running = False
                    break

            # display 해상도로 리사이즈
            if (frame.shape[1] != self.display_width or
                    frame.shape[0] != self.display_height):
                frame = cv2.resize(
                    frame,
                    (self.display_width, self.display_height),
                )

            with self._lock:
                self._frame = frame

            # 원본 FPS에 맞춰 sleep
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame(self):
        """최신 프레임 반환 (thread-safe, CameraManager와 동일 시맨틱)

        프레임 소유권이 호출자에게 이전됩니다.
        다음 프레임이 준비될 때까지 None을 반환합니다.
        """
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame
            self._frame = None
            return frame

    def stop(self):
        """비디오 읽기 중지 및 리소스 해제"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame = None
        logger.info("비디오 소스 중지 완료")

    @property
    def is_running(self):
        return self._running

    def __del__(self):
        self.stop()
