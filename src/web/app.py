import cv2
import logging
from flask import Flask, Response, render_template, jsonify

logger = logging.getLogger(__name__)

app = Flask(__name__)

# main.py에서 주입
camera = None
detector = None


def init_app(camera_manager, detector_instance=None):
    """Flask 앱에 카메라 매니저 및 검출기 연결"""
    global camera, detector
    camera = camera_manager
    detector = detector_instance


def _annotate_frame(frame, detections):
    """프레임에 검출 결과 오버레이"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cls_id = det['class_id']

        color = (0, 255, 0) if cls_id == 0 else (255, 128, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"id:{cls_id} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def _overlay_fps(frame, fps):
    """프레임 좌상단에 FPS 오버레이"""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


def generate_frames():
    """MJPEG 스트리밍용 프레임 제너레이터"""
    while True:
        if camera is None:
            continue

        frame = camera.get_frame()
        if frame is None:
            continue

        # 검출기가 활성화된 경우 추론 및 오버레이
        if detector is not None and detector.is_ready:
            detections = detector.detect(frame)
            frame = _annotate_frame(frame, detections)
            frame = _overlay_fps(frame, detector.get_fps())

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG 스트리밍 엔드포인트"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stats')
def api_stats():
    """실시간 통계 API: FPS, 검출된 사람 수"""
    if detector is not None and detector.is_ready:
        return jsonify({
            'fps': detector.get_fps(),
            'person_count': detector.get_detection_count(),
            'detector_active': True,
        })
    return jsonify({
        'fps': 0,
        'person_count': 0,
        'detector_active': False,
    })


@app.route('/health')
def health():
    """헬스체크"""
    running = camera is not None and camera.is_running
    det_ready = detector is not None and detector.is_ready
    return jsonify({
        'status': 'ok' if running else 'error',
        'camera': running,
        'detector': det_ready,
    })
