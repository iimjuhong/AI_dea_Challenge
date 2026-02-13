"""ByteTrack 기반 다중 객체 추적기 + 칼만필터

Phase 4: 프레임 간 동일 인물 추적, bbox 안정화, 일시적 가림 대응.
- KalmanBoxTracker: cv2.KalmanFilter 래핑 (7차원 상태, 4차원 측정)
- ByteTracker: 2단계 연관 (고/저신뢰도 분리 매칭)
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
#  KalmanBoxTracker
# ============================================================

class KalmanBoxTracker:
    """칼만필터 기반 단일 바운딩박스 추적기

    상태 벡터 (7D): [cx, cy, s, r, dx, dy, ds]
      cx, cy = 중심 좌표
      s      = 면적 (w * h)
      r      = 종횡비 (w / h)  — 상수로 가정
      dx, dy, ds = 속도

    측정 벡터 (4D): [cx, cy, s, r]
    """

    _next_id = 1  # 클래스 변수: ID 단조 증가 (절대 재사용 안 함)

    def __init__(self, bbox, confidence=0.0):
        """
        Args:
            bbox: [x1, y1, x2, y2] (정수 또는 실수)
            confidence: 검출 신뢰도
        """
        self.kf = cv2.KalmanFilter(7, 4)

        # 전이행렬 F: 등속 모델
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        self.kf.transitionMatrix[0, 4] = 1.0  # cx += dx
        self.kf.transitionMatrix[1, 5] = 1.0  # cy += dy
        self.kf.transitionMatrix[2, 6] = 1.0  # s  += ds

        # 측정행렬 H: 위치만 관측
        self.kf.measurementMatrix = np.zeros((4, 7), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.measurementMatrix[2, 2] = 1.0
        self.kf.measurementMatrix[3, 3] = 1.0

        # 프로세스 노이즈 Q
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[4, 4] = 1e-2   # dx
        self.kf.processNoiseCov[5, 5] = 1e-2   # dy
        self.kf.processNoiseCov[6, 6] = 5e-3   # ds (면적 변화는 느림)

        # 측정 노이즈 R
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        # 초기 오차 공분산 P
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        self.kf.errorCovPost[4, 4] = 10.0
        self.kf.errorCovPost[5, 5] = 10.0
        self.kf.errorCovPost[6, 6] = 10.0

        # 초기 상태
        z = self._bbox_to_z(bbox)
        self.kf.statePost = np.zeros((7, 1), dtype=np.float32)
        self.kf.statePost[:4] = z.reshape(4, 1)

        # 트랙 속성
        self.track_id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.hits = 1           # 총 매칭 횟수
        self.age = 0            # 생성 후 총 프레임 수
        self.time_since_update = 0  # 마지막 매칭 후 경과 프레임
        self.confidence = confidence

    def predict(self):
        """칼만 예측 단계. 예측된 bbox 반환.

        Returns:
            [x1, y1, x2, y2] numpy array
        """
        # 면적이 음수가 되지 않도록 보정
        if self.kf.statePost[2, 0] + self.kf.statePost[6, 0] <= 0:
            self.kf.statePost[6, 0] = 0.0

        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._z_to_bbox(self.kf.statePre[:4].flatten())

    def update(self, bbox, confidence=0.0):
        """칼만 보정 단계.

        Args:
            bbox: [x1, y1, x2, y2]
            confidence: 검출 신뢰도
        """
        z = self._bbox_to_z(bbox).reshape(4, 1).astype(np.float32)
        self.kf.correct(z)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = confidence

    def get_state(self):
        """현재 상태에서 bbox 반환.

        Returns:
            [x1, y1, x2, y2] (int list)
        """
        return self._z_to_bbox(self.kf.statePost[:4].flatten()).astype(int).tolist()

    @staticmethod
    def _bbox_to_z(bbox):
        """[x1, y1, x2, y2] → [cx, cy, s, r]"""
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h        # 면적
        r = w / h if h > 0 else 1.0  # 종횡비
        return np.array([cx, cy, s, r], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z):
        """[cx, cy, s, r] → [x1, y1, x2, y2]"""
        cx, cy, s, r = z[0], z[1], z[2], z[3]
        s = max(s, 1.0)
        r = max(r, 0.01)
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 1.0
        return np.array([
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ], dtype=np.float32)


# ============================================================
#  IoU 배치 계산
# ============================================================

def _iou_batch(bboxes_a, bboxes_b):
    """두 bbox 배열 사이의 IoU 행렬 계산 (numpy 벡터화)

    Args:
        bboxes_a: (N, 4) array — [x1, y1, x2, y2]
        bboxes_b: (M, 4) array — [x1, y1, x2, y2]

    Returns:
        (N, M) IoU 행렬
    """
    a = np.array(bboxes_a, dtype=np.float32)
    b = np.array(bboxes_b, dtype=np.float32)

    if a.size == 0 or b.size == 0:
        return np.empty((len(a), len(b)), dtype=np.float32)

    # 브로드캐스팅: (N,1,4) vs (1,M,4)
    a = a[:, np.newaxis, :]  # (N, 1, 4)
    b = b[np.newaxis, :, :]  # (1, M, 4)

    # 교차 영역
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    # 합집합 영역
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union_area = area_a + area_b - inter_area

    return inter_area / np.maximum(union_area, 1e-6)


# ============================================================
#  그리디 매칭
# ============================================================

def _greedy_assignment(iou_matrix, iou_threshold=0.3):
    """IoU 행렬 기반 그리디 매칭 (scipy 불필요)

    IoU 내림차순으로 정렬 후 탐욕적 할당.

    Args:
        iou_matrix: (N, M) IoU 행렬
        iou_threshold: 최소 IoU 임계값

    Returns:
        matches: list of (row, col) 튜플
        unmatched_rows: list of int
        unmatched_cols: list of int
    """
    if iou_matrix.size == 0:
        return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

    n_rows, n_cols = iou_matrix.shape

    # 모든 (row, col, iou) 쌍을 IoU 내림차순 정렬
    flat_indices = np.argsort(-iou_matrix.ravel())
    rows = flat_indices // n_cols
    cols = flat_indices % n_cols
    ious = iou_matrix.ravel()[flat_indices]

    matched_rows = set()
    matched_cols = set()
    matches = []

    for r, c, iou in zip(rows, cols, ious):
        if iou < iou_threshold:
            break
        if r in matched_rows or c in matched_cols:
            continue
        matches.append((int(r), int(c)))
        matched_rows.add(r)
        matched_cols.add(c)

    unmatched_rows = [i for i in range(n_rows) if i not in matched_rows]
    unmatched_cols = [j for j in range(n_cols) if j not in matched_cols]

    return matches, unmatched_rows, unmatched_cols


# ============================================================
#  ByteTracker
# ============================================================

class ByteTracker:
    """ByteTrack 기반 다중 객체 추적기

    2단계 연관:
      1) 고신뢰도 검출 (≥ high_thresh) ↔ 전체 트랙 IoU 매칭
      2) 저신뢰도 검출 (≥ low_thresh) ↔ 1단계 미매칭 트랙 IoU 매칭

    미매칭 고신뢰도 검출 → 새 트랙 생성
    time_since_update > max_age → 트랙 제거
    hits >= min_hits → 출력에 포함
    """

    def __init__(self, max_age=30, min_hits=3,
                 high_thresh=0.5, low_thresh=0.1,
                 iou_threshold=0.3):
        """
        Args:
            max_age: 미매칭 후 트랙 유지 프레임 수
            min_hits: 출력 최소 매칭 횟수
            high_thresh: 고신뢰도 검출 임계값
            low_thresh: 저신뢰도 검출 임계값
            iou_threshold: IoU 매칭 임계값
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_threshold = iou_threshold

        self._tracks = []  # list of KalmanBoxTracker

    def update(self, detections):
        """프레임 단위 추적 업데이트

        Args:
            detections: list of {'bbox': [x1,y1,x2,y2], 'confidence': float, 'class_id': int}

        Returns:
            list of dict: [{'bbox', 'confidence', 'class_id', 'track_id', 'age', 'hits'}, ...]
        """
        # --- 1. 칼만 예측 (모든 기존 트랙) ---
        for trk in self._tracks:
            trk.predict()

        # --- 2. 검출 분류: 고/저 신뢰도 ---
        dets_high = []  # (index, det)
        dets_low = []

        for i, det in enumerate(detections):
            conf = det['confidence']
            if conf >= self.high_thresh:
                dets_high.append((i, det))
            elif conf >= self.low_thresh:
                dets_low.append((i, det))

        # --- 3. 1단계: 고신뢰도 ↔ 전체 트랙 매칭 ---
        unmatched_tracks_idx = list(range(len(self._tracks)))

        if dets_high and self._tracks:
            det_bboxes = np.array([d['bbox'] for _, d in dets_high], dtype=np.float32)
            trk_bboxes = np.array([t.get_state() for t in self._tracks], dtype=np.float32)

            iou_mat = _iou_batch(det_bboxes, trk_bboxes)
            matches_1, unmatched_dets_high_idx, unmatched_tracks_idx = \
                _greedy_assignment(iou_mat, self.iou_threshold)

            for d_idx, t_idx in matches_1:
                _, det = dets_high[d_idx]
                self._tracks[t_idx].update(det['bbox'], det['confidence'])
        else:
            unmatched_dets_high_idx = list(range(len(dets_high)))

        # --- 4. 2단계: 저신뢰도 ↔ 1단계 미매칭 트랙 매칭 ---
        remaining_tracks = [self._tracks[i] for i in unmatched_tracks_idx]

        if dets_low and remaining_tracks:
            det_bboxes = np.array([d['bbox'] for _, d in dets_low], dtype=np.float32)
            trk_bboxes = np.array([t.get_state() for t in remaining_tracks], dtype=np.float32)

            iou_mat = _iou_batch(det_bboxes, trk_bboxes)
            matches_2, _, still_unmatched_trk_idx = \
                _greedy_assignment(iou_mat, self.iou_threshold)

            for d_idx, t_idx in matches_2:
                _, det = dets_low[d_idx]
                remaining_tracks[t_idx].update(det['bbox'], det['confidence'])

        # --- 5. 미매칭 고신뢰도 검출 → 새 트랙 생성 ---
        for d_idx in unmatched_dets_high_idx:
            _, det = dets_high[d_idx]
            new_trk = KalmanBoxTracker(det['bbox'], det['confidence'])
            self._tracks.append(new_trk)

        # --- 6. 오래된 트랙 제거 ---
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # --- 7. 출력: 활성 트랙 (hits >= min_hits 또는 아직 초기) ---
        results = []
        for trk in self._tracks:
            if trk.time_since_update > 0:
                continue  # 이번 프레임에 매칭 안 된 트랙은 출력에서 제외
            if trk.hits < self.min_hits:
                continue  # 충분히 확인되지 않은 트랙 제외

            bbox = trk.get_state()
            results.append({
                'bbox': bbox,
                'confidence': trk.confidence,
                'class_id': 0,  # person
                'track_id': trk.track_id,
                'age': trk.age,
                'hits': trk.hits,
            })

        return results

    @property
    def active_track_count(self):
        """현재 활성 트랙 수"""
        return len(self._tracks)

    @property
    def active_track_ids(self):
        """현재 활성 트랙 ID 목록"""
        return [t.track_id for t in self._tracks if t.time_since_update == 0]


# ============================================================
#  ROI 체류 시간 필터
# ============================================================

class ROIDwellFilter:
    """ROI 체류 시간 기반 카운팅 필터

    잠깐 ROI에 들어왔다 나가는 사람을 인원수에서 제외한다.
    track_id별로 각 ROI 안에 연속으로 머문 프레임 수를 추적하고,
    min_dwell_frames 이상 체류한 트랙만 카운팅에 포함한다.

    ROI를 벗어나면 해당 카운터는 즉시 리셋된다.
    """

    def __init__(self, min_dwell_frames=30):
        """
        Args:
            min_dwell_frames: ROI 내 최소 체류 프레임 수 (기본 30 ≈ 1초@30fps)
        """
        self.min_dwell_frames = min_dwell_frames
        # {(track_id, roi_name): 연속 체류 프레임 수}
        self._dwell = {}

    def update(self, roi_detections):
        """프레임 단위 체류 시간 업데이트 및 필터링된 카운트 반환

        Args:
            roi_detections: {roi_name: [tracked_det, ...]}
                roi_manager.filter_detections_by_roi() 결과

        Returns:
            {roi_name: filtered_count} — min_dwell_frames 이상 체류한 인원만 카운트
        """
        # 이번 프레임에서 ROI 안에 있는 (track_id, roi_name) 쌍 수집
        current_pairs = set()
        for roi_name, dets in roi_detections.items():
            for det in dets:
                track_id = det.get('track_id')
                if track_id is None:
                    continue
                key = (track_id, roi_name)
                current_pairs.add(key)
                self._dwell[key] = self._dwell.get(key, 0) + 1

        # ROI를 벗어난 쌍은 카운터 리셋 (즉시 삭제)
        expired = [k for k in self._dwell if k not in current_pairs]
        for k in expired:
            del self._dwell[k]

        # min_dwell_frames 이상 체류한 트랙만 카운팅
        counts = {}
        for roi_name in roi_detections:
            count = 0
            for det in roi_detections[roi_name]:
                track_id = det.get('track_id')
                if track_id is None:
                    continue
                key = (track_id, roi_name)
                if self._dwell.get(key, 0) >= self.min_dwell_frames:
                    count += 1
            counts[roi_name] = count

        return counts
