import cv2
import logging
from flask import Flask, Response, render_template, jsonify, request

logger = logging.getLogger(__name__)

app = Flask(__name__)

# main.py에서 주입
camera = None
detector = None
roi_manager = None

# ROI별 인원 수 캐시 (generate_frames에서 갱신)
_latest_roi_counts = {}


def init_app(camera_manager, detector_instance=None, roi_manager_instance=None):
    """Flask 앱에 카메라 매니저, 검출기, ROI 매니저 연결"""
    global camera, detector, roi_manager
    camera = camera_manager
    detector = detector_instance
    roi_manager = roi_manager_instance


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
    global _latest_roi_counts

    while True:
        if camera is None:
            continue

        frame = camera.get_frame()
        if frame is None:
            continue

        # 검출기가 활성화된 경우 추론 및 오버레이
        detections = []
        if detector is not None and detector.is_ready:
            detections = detector.detect(frame)
            frame = _annotate_frame(frame, detections)
            frame = _overlay_fps(frame, detector.get_fps())

        # ROI 오버레이 및 인원 수 갱신
        if roi_manager is not None:
            if detections:
                _latest_roi_counts = roi_manager.count_per_roi(detections)
            elif detector is not None and detector.is_ready:
                # 검출기 활성이나 검출 없음 → 모든 ROI 0명
                rois = roi_manager.get_all_rois()
                _latest_roi_counts = {r["name"]: 0 for r in rois}
            frame = roi_manager.draw_rois(frame)

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
    """실시간 통계 API: FPS, 검출된 사람 수, ROI별 인원 수"""
    if detector is not None and detector.is_ready:
        return jsonify({
            'fps': detector.get_fps(),
            'person_count': detector.get_detection_count(),
            'detector_active': True,
            'roi_counts': _latest_roi_counts,
        })
    return jsonify({
        'fps': 0,
        'person_count': 0,
        'detector_active': False,
        'roi_counts': _latest_roi_counts,
    })


# --- ROI API ---

@app.route('/api/roi', methods=['GET'])
def api_roi_list():
    """전체 ROI 목록 반환"""
    if roi_manager is None:
        return jsonify({'error': 'ROI manager not initialized'}), 503
    return jsonify({'rois': roi_manager.get_all_rois()})


@app.route('/api/roi', methods=['POST'])
def api_roi_add():
    """ROI 추가: {name, points, color?}"""
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
    """ROI별 인원 수 반환"""
    return jsonify({'roi_counts': _latest_roi_counts})


@app.route('/api/roi/<name>', methods=['PUT'])
def api_roi_update(name):
    """ROI 수정"""
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
    """ROI 삭제"""
    if roi_manager is None:
        return jsonify({'error': 'ROI manager not initialized'}), 503

    if roi_manager.remove_roi(name):
        return jsonify({'ok': True})
    return jsonify({'error': f'ROI를 찾을 수 없습니다: {name}'}), 404


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
