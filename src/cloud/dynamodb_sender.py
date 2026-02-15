"""DynamoDB 데이터 전송 모듈

Phase 6: Jetson Orin Nano에서 측정한 대기시간 데이터를 AWS DynamoDB로 전송.

기능:
  - snake_case 입력 → camelCase DynamoDB 아이템 변환
  - PK/SK 자동 생성
  - ISO 8601 타임스탬프 (KST, +09:00)
  - TTL 자동 계산 (30일)
  - 배치 쓰기 (최대 25개 아이템)
  - Exponential backoff 재시도

보안:
  - AWS 자격증명은 환경 변수 사용 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
  - 코드에 하드코딩 금지
"""

import json
import logging
import os
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))
_TTL_DAYS = 30
_MAX_BATCH_SIZE = 25
_MAX_RETRIES = 3
_BASE_DELAY = 0.5  # 초


class DynamoDBSender:
    """DynamoDB 비동기 배치 전송 클라이언트

    사용법:
        sender = DynamoDBSender(config_path='config/aws_config.json')
        sender.start()

        # 데이터 전송 (논블로킹, 내부 큐에 적재)
        sender.send({
            'restaurant_id': 'hanyang_plaza',
            'corner_id': 'korean',
            'queue_count': 15,
            'est_wait_time_min': 8,
            'timestamp': 1770349800000,
        })

        # 종료 시
        sender.stop()
    """

    def __init__(self, config_path='config/aws_config.json'):
        self._config = self._load_config(config_path)
        self._table_name = self._config['table_name']
        self._restaurant_id = self._config.get('restaurant_id', 'unknown')
        self._corner_id = self._config.get('corner_id', 'unknown')
        self._region = self._config.get('region', 'ap-northeast-2')

        self._queue = deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._client = None

        # 통계
        self._sent_count = 0
        self._error_count = 0

    @staticmethod
    def _load_config(config_path):
        """설정 파일 로드"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"AWS 설정 파일 없음: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        required = ['region', 'table_name']
        for key in required:
            if key not in config:
                raise ValueError(f"설정 파일에 '{key}' 키 누락: {config_path}")
        return config

    def _init_client(self):
        """boto3 DynamoDB 클라이언트 초기화 (lazy)"""
        if self._client is not None:
            return
        try:
            import boto3
            self._client = boto3.resource(
                'dynamodb',
                region_name=self._region,
            )
            logger.info(
                f"DynamoDB 클라이언트 초기화 완료 "
                f"(region={self._region}, table={self._table_name})"
            )
        except Exception as e:
            logger.error(f"DynamoDB 클라이언트 초기화 실패: {e}")
            raise

    def start(self):
        """백그라운드 전송 워커 시작"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("DynamoDB 워커가 이미 실행 중")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name='dynamodb-sender'
        )
        self._worker_thread.start()
        logger.info("DynamoDB 전송 워커 시작")

    def stop(self):
        """워커 종료 (남은 큐 플러시 후 종료)"""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=10)
            logger.info(
                f"DynamoDB 워커 종료 (전송={self._sent_count}, 오류={self._error_count})"
            )

    def send(self, data):
        """데이터를 전송 큐에 추가 (논블로킹)

        Args:
            data: dict with keys:
                - restaurant_id (str)
                - corner_id (str)
                - queue_count (int)
                - est_wait_time_min (float)
                - timestamp (int): epoch milliseconds
        """
        item = self._transform(data)
        with self._lock:
            self._queue.append(item)

    def send_batch(self, data_list):
        """여러 데이터를 한 번에 큐에 추가

        Args:
            data_list: list of dicts (send()와 동일한 형식)
        """
        items = [self._transform(d) for d in data_list]
        with self._lock:
            self._queue.extend(items)

    @property
    def pending_count(self):
        """전송 대기 중인 아이템 수"""
        with self._lock:
            return len(self._queue)

    @property
    def stats(self):
        """전송 통계"""
        return {
            'sent': self._sent_count,
            'errors': self._error_count,
            'pending': self.pending_count,
        }

    # ----------------------------------------------------------
    #  데이터 변환
    # ----------------------------------------------------------

    def _transform(self, data):
        """snake_case 입력 → camelCase DynamoDB 아이템 변환

        Args:
            data: {
                'restaurant_id': str,
                'corner_id': str,
                'queue_count': int,
                'est_wait_time_min': float,
                'timestamp': int (epoch ms),
            }

        Returns:
            DynamoDB 아이템 dict (camelCase)
        """
        restaurant_id = data.get('restaurant_id', self._restaurant_id)
        corner_id = data.get('corner_id', self._corner_id)
        timestamp_ms = data['timestamp']
        queue_count = data['queue_count']
        est_wait_time_min = data['est_wait_time_min']

        # epoch ms → datetime (KST)
        timestamp_sec = timestamp_ms / 1000.0
        dt_kst = datetime.fromtimestamp(timestamp_sec, tz=_KST)
        timestamp_iso = dt_kst.isoformat()

        # 현재 시각 (DB 저장 시간)
        now_kst = datetime.now(tz=_KST)
        created_at_iso = now_kst.isoformat()

        # TTL: 원본 타임스탬프 + 30일 (epoch 초 단위)
        ttl = int(timestamp_sec) + (_TTL_DAYS * 24 * 60 * 60)

        return {
            'pk': f"CORNER#{restaurant_id}#{corner_id}",
            'sk': str(timestamp_ms),
            'restaurantId': restaurant_id,
            'cornerId': corner_id,
            'queueLen': int(queue_count),
            'estWaitTimeMin': int(est_wait_time_min),
            'dataType': 'observed',
            'source': 'jetson_nano',
            'timestampIso': timestamp_iso,
            'createdAtIso': created_at_iso,
            'ttl': ttl,
        }

    # ----------------------------------------------------------
    #  백그라운드 워커
    # ----------------------------------------------------------

    def _worker_loop(self):
        """배치 전송 워커 루프"""
        try:
            self._init_client()
        except Exception:
            logger.error("DynamoDB 클라이언트 초기화 실패, 워커 종료")
            return

        while not self._stop_event.is_set():
            batch = self._drain_queue(_MAX_BATCH_SIZE)
            if not batch:
                self._stop_event.wait(timeout=1.0)
                continue

            self._write_batch(batch)

        # 종료 시 남은 큐 플러시
        while True:
            batch = self._drain_queue(_MAX_BATCH_SIZE)
            if not batch:
                break
            self._write_batch(batch)

    def _drain_queue(self, max_count):
        """큐에서 최대 max_count개 아이템 꺼내기"""
        with self._lock:
            count = min(len(self._queue), max_count)
            if count == 0:
                return []
            return [self._queue.popleft() for _ in range(count)]

    def _write_batch(self, items):
        """DynamoDB batch_write_item 실행 (재시도 포함)

        Args:
            items: list of DynamoDB item dicts
        """
        table = self._client.Table(self._table_name)

        for attempt in range(_MAX_RETRIES):
            try:
                with table.batch_writer() as batch:
                    for item in items:
                        batch.put_item(Item=item)

                self._sent_count += len(items)
                logger.debug(f"DynamoDB 전송 완료: {len(items)}건")
                return

            except Exception as e:
                self._error_count += 1
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    f"DynamoDB 전송 실패 (시도 {attempt + 1}/{_MAX_RETRIES}): {e}. "
                    f"{delay:.1f}초 후 재시도"
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)

        # 모든 재시도 실패 → 아이템을 다시 큐에 넣기
        logger.error(
            f"DynamoDB 전송 최종 실패: {len(items)}건. 큐에 재적재."
        )
        with self._lock:
            self._queue.extendleft(reversed(items))
