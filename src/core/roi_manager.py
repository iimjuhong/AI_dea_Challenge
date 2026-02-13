import json
import logging
import os
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 8-color palette (BGR)
_COLOR_PALETTE = [
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 0, 255),      # red
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 165, 255),    # orange
    (203, 192, 255),  # pink
]


class ROIManager:
    """ROI(Region of Interest) 관리자

    다각형 ROI를 생성/수정/삭제하고, 검출 결과를 ROI별로 분류한다.
    좌표는 640x480 디스플레이 해상도 기준이다.
    """

    def __init__(self, config_path="config/roi_config.json"):
        self._config_path = config_path
        self._rois = []  # list of {"name": str, "points": [[x,y],...], "color": [B,G,R]}
        self._lock = threading.Lock()
        self._overlay = None  # draw_rois() 오버레이 버퍼 재사용

    def load(self) -> bool:
        """JSON 파일에서 ROI 설정을 로드한다."""
        if not os.path.exists(self._config_path):
            logger.info(f"ROI 설정 파일 없음 (첫 실행): {self._config_path}")
            return True
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with self._lock:
                self._rois = data.get("rois", [])
            logger.info(f"ROI {len(self._rois)}개 로드 완료: {self._config_path}")
            return True
        except Exception as e:
            logger.error(f"ROI 설정 로드 실패: {e}")
            return False

    def save(self) -> bool:
        """현재 ROI 설정을 JSON 파일로 저장한다."""
        try:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with self._lock:
                data = {"rois": list(self._rois)}
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"ROI 설정 저장 완료: {self._config_path}")
            return True
        except Exception as e:
            logger.error(f"ROI 설정 저장 실패: {e}")
            return False

    def add_roi(self, name, points, color=None) -> bool:
        """새 ROI를 추가한다.

        Args:
            name: ROI 이름 (고유)
            points: 다각형 꼭짓점 좌표 [[x,y], ...]
            color: BGR 색상 [B,G,R] (None이면 팔레트에서 자동 선택)
        """
        with self._lock:
            if any(r["name"] == name for r in self._rois):
                logger.warning(f"ROI 이름 중복: {name}")
                return False
            if color is None:
                color = list(_COLOR_PALETTE[len(self._rois) % len(_COLOR_PALETTE)])
            self._rois.append({
                "name": name,
                "points": points,
                "color": color,
            })
        self.save()
        logger.info(f"ROI 추가: {name} ({len(points)}개 꼭짓점)")
        return True

    def remove_roi(self, name) -> bool:
        """ROI를 삭제한다."""
        with self._lock:
            before = len(self._rois)
            self._rois = [r for r in self._rois if r["name"] != name]
            if len(self._rois) == before:
                return False
        self.save()
        logger.info(f"ROI 삭제: {name}")
        return True

    def update_roi(self, name, points=None, color=None, new_name=None) -> bool:
        """기존 ROI를 수정한다."""
        with self._lock:
            roi = next((r for r in self._rois if r["name"] == name), None)
            if roi is None:
                return False
            if new_name is not None:
                if any(r["name"] == new_name for r in self._rois if r is not roi):
                    return False
                roi["name"] = new_name
            if points is not None:
                roi["points"] = points
            if color is not None:
                roi["color"] = color
        self.save()
        logger.info(f"ROI 수정: {name}")
        return True

    def get_all_rois(self):
        """전체 ROI 목록을 반환한다."""
        with self._lock:
            return [dict(r) for r in self._rois]

    def get_roi(self, name):
        """이름으로 ROI를 찾아 반환한다."""
        with self._lock:
            for r in self._rois:
                if r["name"] == name:
                    return dict(r)
        return None

    def count_per_roi(self, detections):
        """각 ROI에 포함된 검출 수를 반환한다.

        cv2.pointPolygonTest (C++) 사용으로 GIL 해제 + 성능 향상.

        Args:
            detections: list of {'bbox': [x1,y1,x2,y2], ...}

        Returns:
            dict: {roi_name: count}
        """
        with self._lock:
            rois = list(self._rois)
        counts = {}
        for roi in rois:
            contour = np.array(roi["points"], dtype=np.float32).reshape(-1, 1, 2)
            count = 0
            for det in detections:
                pt = self._get_bottom_center(det["bbox"])
                # cv2.pointPolygonTest: C++에서 실행되므로 GIL 해제
                if cv2.pointPolygonTest(contour, pt, False) >= 0:
                    count += 1
            counts[roi["name"]] = count
        return counts

    def filter_detections_by_roi(self, detections):
        """ROI별로 검출 결과를 분류한다.

        Returns:
            dict: {roi_name: [detection, ...]}
        """
        with self._lock:
            rois = list(self._rois)
        result = {}
        for roi in rois:
            contour = np.array(roi["points"], dtype=np.float32).reshape(-1, 1, 2)
            matched = []
            for det in detections:
                pt = self._get_bottom_center(det["bbox"])
                if cv2.pointPolygonTest(contour, pt, False) >= 0:
                    matched.append(det)
            result[roi["name"]] = matched
        return result

    def draw_rois(self, frame, alpha=0.3):
        """프레임에 모든 ROI를 반투명 다각형으로 오버레이한다.

        Args:
            frame: BGR numpy array
            alpha: 투명도 (0.0 ~ 1.0)

        Returns:
            오버레이된 프레임
        """
        with self._lock:
            rois = list(self._rois)

        if not rois:
            return frame

        # 오버레이 버퍼 재사용 (매 프레임 할당 방지)
        if self._overlay is None or self._overlay.shape != frame.shape:
            self._overlay = np.empty_like(frame)
        np.copyto(self._overlay, frame)

        for roi in rois:
            pts = np.array(roi["points"], dtype=np.int32)
            color = tuple(roi["color"])

            # 반투명 채우기
            cv2.fillPoly(self._overlay, [pts], color)

            # 테두리 (불투명)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            # ROI 이름 라벨
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            label = roi["name"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 6),
                          (cx + tw // 2 + 4, cy + 4), color, -1)
            cv2.putText(frame, label, (cx - tw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.addWeighted(self._overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    @staticmethod
    def _point_in_polygon(px, py, polygon):
        """Ray-casting 알고리즘으로 점이 다각형 내부에 있는지 판별한다."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _get_bottom_center(bbox):
        """bbox [x1, y1, x2, y2]의 하단 중심점을 반환한다.

        Returns:
            (float, float) — cv2.pointPolygonTest 호환 좌표
        """
        x1, y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y2))
