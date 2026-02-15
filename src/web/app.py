import threading
import time

import cv2
import logging
import numpy as np
from flask import Flask, Response, render_template, jsonify, request

logger = logging.getLogger(__name__)

# TurboJPEG: cv2.imencode 대비 2-3x 빠른 JPEG 인코딩 (libjpeg-turbo NEON SIMD)
try:
    from turbojpeg import TurboJPEG
    _turbojpeg = TurboJPEG()
    logger.info("TurboJPEG 고속 JPEG 인코딩 활성화")
except (ImportError, RuntimeError, OSError):
    _turbojpeg = None

app = Flask(__name__)

# ============================================================
#  아키텍처: 3-Thread 파이프라인
#
#  Thread 1 (Camera)    : CameraManager._capture_loop (기존)
#      └→ 최신 프레임을 _frame에 저장, get_frame()으로 폴링
#
#  Thread 2 (Inference)  : _inference_loop
#      └→ get_frame() → YOLO → ROI → JPEG → FrameBuffer.put()
#      └→ 네트워크와 완전 독립, 절대 블로킹 안 됨
#
#  Thread 3 (Network/Flask) : generate_frames()
#      └→ FrameBuffer.get() → yield MJPEG
#      └→ 느린 클라이언트는 프레임 스킵 (항상 최신만 수신)
#      └→ 와이파이가 끊겨도 Inference 스레드에 영향 없음
# ============================================================

# main.py에서 주입
camera = None
detector = None
roi_manager = None
tracker = None
_dwell_filter = None
_wait_estimator = None
_dynamodb_sender = None

_latest_roi_counts = {}
_latest_tracked = []  # 최신 추적 결과 (API용)
_latest_wait_result = {}  # 최신 대기시간 결과
_inference_thread = None

_TARGET_FPS = 30
_DYNAMODB_INTERVAL = 10.0  # DynamoDB 전송 주기 (초)


# --- FrameBuffer: Inference → Network 사이 논블로킹 버퍼 ---

class FrameBuffer:
    """스레드 안전 프레임 버퍼 (최신 1프레임만 유지, 오래된 프레임 자동 폐기)

    - put(): 항상 즉시 반환 (논블로킹). 이전 프레임은 덮어쓰기 (드롭).
    - get(): 새 프레임 도착까지 대기. 타임아웃 시 None 반환.
    - 여러 소비자(MJPEG 클라이언트)가 동시에 읽어도 안전.
    - 느린 소비자는 중간 프레임을 건너뛰고 항상 최신 프레임을 받음.
    """

    def __init__(self):
        self._jpeg = None
        self._frame_id = 0
        self._cond = threading.Condition()

    def put(self, jpeg_bytes):
        """새 프레임 저장. 논블로킹, 이전 프레임 덮어쓰기."""
        with self._cond:
            self._jpeg = jpeg_bytes
            self._frame_id += 1
            self._cond.notify_all()

    def get(self, prev_id=0, timeout=1.0):
        """새 프레임 대기 후 반환.

        Args:
            prev_id: 마지막으로 받은 frame_id (중복 방지)
            timeout: 최대 대기 시간 (초)

        Returns:
            (jpeg_bytes, frame_id) — 새 프레임이 있으면
            (None, prev_id)        — 타임아웃 시
        """
        with self._cond:
            if self._frame_id == prev_id:
                self._cond.wait(timeout=timeout)
            if self._frame_id != prev_id:
                return self._jpeg, self._frame_id
            return None, prev_id


_frame_buf = FrameBuffer()


# --- init ---

def init_app(camera_manager, detector_instance=None, roi_manager_instance=None,
             tracker_instance=None, inference_fps=None, min_dwell_frames=30,
             wait_estimator=None, dynamodb_sender=None):
    """Flask 앱에 카메라/검출기/ROI 매니저/추적기 연결 후 Inference 스레드 시작"""
    global camera, detector, roi_manager, tracker, _dwell_filter
    global _wait_estimator, _dynamodb_sender
    global _inference_thread, _TARGET_FPS
    camera = camera_manager
    detector = detector_instance
    roi_manager = roi_manager_instance
    tracker = tracker_instance
    _wait_estimator = wait_estimator
    _dynamodb_sender = dynamodb_sender

    if tracker is not None:
        from src.core.tracker import ROIDwellFilter
        _dwell_filter = ROIDwellFilter(min_dwell_frames=min_dwell_frames)
        logger.info(f"ROI 체류 필터 활성화 (min_dwell_frames={min_dwell_frames})")

    if inference_fps is not None:
        _TARGET_FPS = inference_fps
        logger.info(f"Inference FPS 설정: {_TARGET_FPS}")

    if _inference_thread is None:
        _inference_thread = threading.Thread(target=_inference_loop, daemon=True)
        _inference_thread.start()
        logger.info(f"Inference 스레드 시작 (target={_TARGET_FPS}fps)")


# --- Thread 2: Inference Loop ---

def _inference_loop():
    """[Inference 스레드] 카메라 → 검출 → ROI → JPEG → FrameBuffer

    - 네트워크와 완전 분리: put()은 항상 논블로킹
    - 예외 발생 시 스레드가 죽지 않고 복구
    - Queue가 꽉 차는 개념 없음 (최신 1프레임 덮어쓰기)
    """
    global _latest_roi_counts, _latest_wait_result
    target_interval = 1.0 / _TARGET_FPS
    error_count = 0

    # 성능 측정 (100프레임마다 로깅)
    _perf_n = 0
    _perf_detect = 0.0
    _perf_jpeg = 0.0
    _perf_total = 0.0
    _PERF_LOG_INTERVAL = 100

    # DynamoDB 전송 상태
    _last_dynamo_send = 0.0
    _last_sent_wait_min = -1

    while True:
        try:
            t0 = time.monotonic()

            # 카메라 대기
            if camera is None or not camera.is_running:
                time.sleep(0.1)
                continue

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # --- 검출 + 추적 (실패 시 원본 프레임으로 계속) ---
            detections = []
            tracked = []
            if detector is not None and detector.is_ready:
                try:
                    detections = detector.detect(frame)

                    # 추적기 적용: detections → tracked (track_id 포함)
                    if tracker is not None and detections:
                        tracked = tracker.update(detections)
                    elif tracker is not None:
                        tracked = tracker.update([])
                    else:
                        tracked = detections

                    _latest_tracked[:] = tracked
                    frame = _annotate_frame(frame, tracked)
                    frame = _overlay_fps(frame, detector.get_fps())
                except Exception as e:
                    logger.warning(f"검출/추적 오류 (스킵): {e}")

            # --- ROI (tracked 사용, 실패 시 오버레이 없이 계속) ---
            if roi_manager is not None:
                try:
                    if tracked:
                        if _dwell_filter is not None:
                            # 체류 시간 필터: 잠깐 들어왔다 나가는 사람 제외
                            roi_dets = roi_manager.filter_detections_by_roi(tracked)
                            _latest_roi_counts = _dwell_filter.update(roi_dets)
                        else:
                            _latest_roi_counts = roi_manager.count_per_roi(tracked)
                    elif detector is not None and detector.is_ready:
                        rois = roi_manager.get_all_rois()
                        _latest_roi_counts = {r["name"]: 0 for r in rois}
                    frame = roi_manager.draw_rois(frame)
                except Exception as e:
                    logger.warning(f"ROI 오류 (스킵): {e}")

            # --- WaitTimeEstimator 업데이트 ---
            if _wait_estimator is not None and roi_manager is not None:
                try:
                    if tracked:
                        roi_dets = roi_manager.filter_detections_by_roi(tracked)
                    else:
                        roi_dets = {}
                    wait_result = _wait_estimator.update(roi_dets)
                    _latest_wait_result = wait_result
                except Exception as e:
                    logger.warning(f"WaitTimeEstimator 오류 (스킵): {e}")

            # --- DynamoDB 주기적 전송 ---
            if _dynamodb_sender is not None and _wait_estimator is not None:
                try:
                    now_mono = time.monotonic()
                    predicted_min = int(_latest_wait_result.get('predicted_wait', 0) / 60)
                    queue_count = _latest_wait_result.get('current_queue', 0)
                    time_elapsed = now_mono - _last_dynamo_send
                    value_changed = predicted_min != _last_sent_wait_min

                    if time_elapsed >= _DYNAMODB_INTERVAL or (value_changed and time_elapsed >= 2.0):
                        import time as _time_mod
                        timestamp_ms = int(_time_mod.time() * 1000)
                        _dynamodb_sender.send({
                            'restaurant_id': _dynamodb_sender._restaurant_id,
                            'corner_id': _dynamodb_sender._corner_id,
                            'queue_count': queue_count,
                            'est_wait_time_min': predicted_min,
                            'timestamp': timestamp_ms,
                        })
                        _last_dynamo_send = now_mono
                        _last_sent_wait_min = predicted_min
                except Exception as e:
                    logger.warning(f"DynamoDB 전송 준비 오류 (스킵): {e}")

            # --- JPEG 인코딩 → FrameBuffer (논블로킹) ---
            t_jpeg = time.monotonic()
            if _turbojpeg is not None:
                jpeg_bytes = _turbojpeg.encode(frame, quality=80)
                _frame_buf.put(jpeg_bytes)
                error_count = 0
            else:
                ret, buffer = cv2.imencode('.jpg', frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    _frame_buf.put(buffer.tobytes())
                    error_count = 0
            t_jpeg_end = time.monotonic()

            # 성능 측정 로깅
            elapsed = t_jpeg_end - t0
            _perf_n += 1
            _perf_jpeg += (t_jpeg_end - t_jpeg)
            _perf_total += elapsed
            if _perf_n >= _PERF_LOG_INTERVAL:
                avg_jpeg = _perf_jpeg / _perf_n * 1000
                avg_total = _perf_total / _perf_n * 1000
                enc = "TurboJPEG" if _turbojpeg else "cv2.imencode"
                logger.info(
                    f"[Perf/{_perf_n}f] jpeg({enc})={avg_jpeg:.1f}ms "
                    f"total={avg_total:.1f}ms ({1000/avg_total:.1f}fps)")
                _perf_n = 0
                _perf_jpeg = 0.0
                _perf_total = 0.0

            # FPS 조절 — 최소 1ms sleep으로 GIL 양보 보장
            sleep_time = max(target_interval - elapsed, 0.001)
            time.sleep(sleep_time)

        except Exception as e:
            error_count += 1
            logger.error(f"Inference 오류 ({error_count}회): {e}", exc_info=True)
            time.sleep(min(error_count * 0.5, 5.0))


# --- Thread 3: Network Sender (Flask MJPEG generators) ---

def generate_frames():
    """[Network 스레드] FrameBuffer에서 최신 JPEG를 읽어 MJPEG 스트림 전송

    - Inference 스레드와 완전 독립
    - 느린 클라이언트는 프레임을 건너뜀 (frame_id 추적)
    - 와이파이 끊김 → 이 generator만 멈춤, Inference는 계속 동작
    - 클라이언트 끊김 시 안전하게 종료
    """
    last_id = 0
    frame_timeout = max(1.0 / _TARGET_FPS * 3, 0.5)  # 3프레임 대기, 최소 0.5초
    empty_count = 0
    MAX_EMPTY = 60  # 60회 연속 빈 프레임이면 스트림 종료 (클라이언트 버퍼 절약)

    try:
        while True:
            jpeg, last_id = _frame_buf.get(prev_id=last_id, timeout=frame_timeout)
            if jpeg is None:
                empty_count += 1
                if empty_count >= MAX_EMPTY:
                    logger.warning("MJPEG: 프레임 없음 타임아웃, 스트림 종료")
                    return
                continue
            empty_count = 0

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            )
    except GeneratorExit:
        return
    except Exception as e:
        logger.debug(f"MJPEG 스트림 종료: {e}")
        return


# --- Drawing helpers ---

def _track_color(track_id):
    """HSV 기반 트랙별 고유 색상 생성 (BGR 반환)"""
    hue = (track_id * 37) % 180  # 골든 앵글 근사로 색상 분산
    hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def _annotate_frame(frame, tracked):
    """프레임에 추적 결과 오버레이 (track_id + 고유 색상)"""
    for det in tracked:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        track_id = det.get('track_id')

        if track_id is not None:
            color = _track_color(track_id)
            label = f"#{track_id} {conf:.2f}"
        else:
            color = (0, 255, 0)
            label = f"person {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def _overlay_fps(frame, fps):
    """프레임 좌상단에 FPS 오버레이"""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stats')
def api_stats():
    tracker_active = tracker is not None
    track_ids = tracker.active_track_ids if tracker_active else []

    if detector is not None and detector.is_ready:
        return jsonify({
            'fps': detector.get_fps(),
            'person_count': detector.get_detection_count(),
            'detector_active': True,
            'tracker_active': tracker_active,
            'track_ids': track_ids,
            'roi_counts': _latest_roi_counts,
        })
    return jsonify({
        'fps': 0,
        'person_count': 0,
        'detector_active': False,
        'tracker_active': tracker_active,
        'track_ids': track_ids,
        'roi_counts': _latest_roi_counts,
    })


@app.route('/api/tracks')
def api_tracks():
    """현재 활성 추적 결과 반환"""
    return jsonify({
        'tracks': list(_latest_tracked),
        'tracker_active': tracker is not None,
    })


@app.route('/api/wait_time')
def api_wait_time():
    """현재 대기시간 예측 및 대기열 정보"""
    if _wait_estimator is None:
        return jsonify({'error': 'WaitTimeEstimator not initialized'}), 503

    result = dict(_latest_wait_result)
    # active_waiters는 JSON 직렬화 위해 정리
    result.pop('events', None)
    result.pop('completed', None)
    active = result.pop('active_waiters', {})
    result['active_waiters'] = {str(k): round(v, 1) for k, v in active.items()}
    result['predicted_wait'] = round(result.get('predicted_wait', 0), 1)

    stats = _wait_estimator.get_statistics()
    if stats:
        result['statistics'] = stats

    return jsonify(result)


@app.route('/api/dynamodb/stats')
def api_dynamodb_stats():
    """DynamoDB 전송 통계"""
    if _dynamodb_sender is None:
        return jsonify({'error': 'DynamoDB sender not initialized'}), 503
    return jsonify(_dynamodb_sender.stats)


# --- ROI API ---

@app.route('/api/roi', methods=['GET'])
def api_roi_list():
    if roi_manager is None:
        return jsonify({'error': 'ROI manager not initialized'}), 503
    return jsonify({'rois': roi_manager.get_all_rois()})


@app.route('/api/roi', methods=['POST'])
def api_roi_add():
    if roi_manager is None:
        return jsonify({'error': 'ROI manager not initialized'}), 503

    data = request.get_json(silent=True)
    if not data or 'name' not in data or 'points' not in data:
        return jsonify({'error': 'name과 points 필드가 필요합니다'}), 400

    name = data['name'].strip()
    points = data['points']
    color = data.get('color')

    if not name:
        return jsonify({'error': '이름이 비어있습니다'}), 400
    if not isinstance(points, list) or len(points) < 3:
        return jsonify({'error': '최소 3개의 꼭짓점이 필요합니다'}), 400

    if roi_manager.add_roi(name, points, color):
        return jsonify({'ok': True, 'roi': roi_manager.get_roi(name)})
    return jsonify({'error': f'ROI 추가 실패 (이름 중복: {name})'}), 409


@app.route('/api/roi/stats', methods=['GET'])
def api_roi_stats():
    return jsonify({'roi_counts': _latest_roi_counts})


@app.route('/api/roi/<name>', methods=['PUT'])
def api_roi_update(name):
    if roi_manager is None:
        return jsonify({'error': 'ROI manager not initialized'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'JSON body가 필요합니다'}), 400

    if roi_manager.update_roi(
        name,
        points=data.get('points'),
        color=data.get('color'),
        new_name=data.get('new_name'),
    ):
        updated_name = data.get('new_name', name)
        return jsonify({'ok': True, 'roi': roi_manager.get_roi(updated_name)})
    return jsonify({'error': f'ROI를 찾을 수 없습니다: {name}'}), 404


@app.route('/api/roi/<name>', methods=['DELETE'])
def api_roi_delete(name):
    if roi_manager is None:
        return jsonify({'error': 'ROI manager not initialized'}), 503

    if roi_manager.remove_roi(name):
        return jsonify({'ok': True})
    return jsonify({'error': f'ROI를 찾을 수 없습니다: {name}'}), 404


@app.route('/health')
def health():
    running = camera is not None and camera.is_running
    det_ready = detector is not None and detector.is_ready
    return jsonify({
        'status': 'ok' if running else 'error',
        'camera': running,
        'detector': det_ready,
    })
