"""DynamoDBSender 단위 테스트

boto3를 mock하여 실제 AWS 연결 없이 테스트.
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from src.cloud.dynamodb_sender import DynamoDBSender


def _make_config(tmpdir, **overrides):
    """테스트용 설정 파일 생성"""
    config = {
        'region': 'ap-northeast-2',
        'table_name': 'test-table',
        'restaurant_id': 'hanyang_plaza',
        'corner_id': 'korean',
    }
    config.update(overrides)
    path = os.path.join(tmpdir, 'aws_config.json')
    with open(path, 'w') as f:
        json.dump(config, f)
    return path


def _sample_data(timestamp_ms=1770349800000, queue_count=15, est_wait_time_min=8):
    """테스트용 입력 데이터 생성"""
    return {
        'restaurant_id': 'hanyang_plaza',
        'corner_id': 'korean',
        'queue_count': queue_count,
        'est_wait_time_min': est_wait_time_min,
        'timestamp': timestamp_ms,
    }


class TestDynamoDBSenderConfig(unittest.TestCase):
    """설정 파일 로드 테스트"""

    def test_load_valid_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_config(tmpdir)
            sender = DynamoDBSender(config_path=path)
            self.assertEqual(sender._table_name, 'test-table')
            self.assertEqual(sender._region, 'ap-northeast-2')
            self.assertEqual(sender._restaurant_id, 'hanyang_plaza')

    def test_missing_config_file(self):
        with self.assertRaises(FileNotFoundError):
            DynamoDBSender(config_path='/nonexistent/path.json')

    def test_missing_required_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'bad.json')
            with open(path, 'w') as f:
                json.dump({'region': 'us-east-1'}, f)  # table_name 누락
            with self.assertRaises(ValueError):
                DynamoDBSender(config_path=path)


class TestTransform(unittest.TestCase):
    """snake_case → camelCase 변환 테스트"""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_config(tmpdir)
            self.sender = DynamoDBSender(config_path=path)
        # tmpdir은 삭제되지만 sender는 이미 config 로드 완료

    def test_pk_format(self):
        item = self.sender._transform(_sample_data())
        self.assertEqual(item['pk'], 'CORNER#hanyang_plaza#korean')

    def test_sk_is_string_timestamp(self):
        data = _sample_data(timestamp_ms=1770349800000)
        item = self.sender._transform(data)
        self.assertEqual(item['sk'], '1770349800000')
        self.assertIsInstance(item['sk'], str)

    def test_camel_case_keys(self):
        item = self.sender._transform(_sample_data())
        expected_keys = {
            'pk', 'sk', 'restaurantId', 'cornerId', 'queueLen',
            'estWaitTimeMin', 'dataType', 'source',
            'timestampIso', 'createdAtIso', 'ttl',
        }
        self.assertEqual(set(item.keys()), expected_keys)

    def test_queue_len_value(self):
        item = self.sender._transform(_sample_data(queue_count=15))
        self.assertEqual(item['queueLen'], 15)
        self.assertIsInstance(item['queueLen'], int)

    def test_est_wait_time_min(self):
        item = self.sender._transform(_sample_data(est_wait_time_min=8))
        self.assertEqual(item['estWaitTimeMin'], 8)

    def test_data_type_observed(self):
        item = self.sender._transform(_sample_data())
        self.assertEqual(item['dataType'], 'observed')

    def test_source_jetson(self):
        item = self.sender._transform(_sample_data())
        self.assertEqual(item['source'], 'jetson_nano')

    def test_timestamp_iso_kst(self):
        item = self.sender._transform(_sample_data(timestamp_ms=1770349800000))
        self.assertIn('+09:00', item['timestampIso'])

    def test_created_at_iso_kst(self):
        item = self.sender._transform(_sample_data())
        self.assertIn('+09:00', item['createdAtIso'])

    def test_ttl_30_days(self):
        data = _sample_data(timestamp_ms=1770349800000)
        item = self.sender._transform(data)
        expected_ttl = 1770349800 + (30 * 24 * 60 * 60)
        self.assertEqual(item['ttl'], expected_ttl)

    def test_default_restaurant_corner_from_config(self):
        """restaurant_id, corner_id 미지정 시 config 값 사용"""
        data = {
            'queue_count': 5,
            'est_wait_time_min': 3,
            'timestamp': 1770349800000,
        }
        item = self.sender._transform(data)
        self.assertEqual(item['restaurantId'], 'hanyang_plaza')
        self.assertEqual(item['cornerId'], 'korean')


class TestSendQueue(unittest.TestCase):
    """전송 큐 관리 테스트"""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_config(tmpdir)
            self.sender = DynamoDBSender(config_path=path)

    def test_send_adds_to_queue(self):
        self.assertEqual(self.sender.pending_count, 0)
        self.sender.send(_sample_data())
        self.assertEqual(self.sender.pending_count, 1)

    def test_send_batch(self):
        data_list = [_sample_data(timestamp_ms=i) for i in range(5)]
        self.sender.send_batch(data_list)
        self.assertEqual(self.sender.pending_count, 5)

    def test_stats_initial(self):
        stats = self.sender.stats
        self.assertEqual(stats['sent'], 0)
        self.assertEqual(stats['errors'], 0)
        self.assertEqual(stats['pending'], 0)


class TestBatchWrite(unittest.TestCase):
    """배치 쓰기 테스트 (boto3 mock)"""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_config(tmpdir)
            self.sender = DynamoDBSender(config_path=path)

    @patch('src.cloud.dynamodb_sender.DynamoDBSender._init_client')
    def test_write_batch_success(self, mock_init):
        """정상 배치 쓰기"""
        mock_table = MagicMock()
        mock_batch_writer = MagicMock()
        mock_table.batch_writer.return_value.__enter__ = MagicMock(
            return_value=mock_batch_writer
        )
        mock_table.batch_writer.return_value.__exit__ = MagicMock(return_value=False)

        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        self.sender._client = mock_resource

        items = [self.sender._transform(_sample_data(timestamp_ms=i)) for i in range(3)]
        self.sender._write_batch(items)

        self.assertEqual(self.sender._sent_count, 3)
        self.assertEqual(mock_batch_writer.put_item.call_count, 3)

    @patch('src.cloud.dynamodb_sender.DynamoDBSender._init_client')
    def test_write_batch_retry_on_failure(self, mock_init):
        """실패 시 재시도"""
        mock_table = MagicMock()
        # 첫 2번 실패, 3번째 성공
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            mock_ctx = MagicMock()
            if call_count <= 2:
                mock_ctx.__enter__ = MagicMock(side_effect=Exception("Throttled"))
            else:
                mock_ctx.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.__exit__ = MagicMock(return_value=False)
            return mock_ctx

        mock_table.batch_writer.side_effect = side_effect
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        self.sender._client = mock_resource

        items = [self.sender._transform(_sample_data())]
        self.sender._write_batch(items)

        self.assertEqual(self.sender._sent_count, 1)
        self.assertEqual(self.sender._error_count, 2)


class TestWorkerLifecycle(unittest.TestCase):
    """워커 스레드 생명주기 테스트"""

    @patch('src.cloud.dynamodb_sender.DynamoDBSender._init_client')
    @patch('src.cloud.dynamodb_sender.DynamoDBSender._write_batch')
    def test_start_stop(self, mock_write, mock_init):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_config(tmpdir)
            sender = DynamoDBSender(config_path=path)

        sender.start()
        self.assertTrue(sender._worker_thread.is_alive())

        sender.send(_sample_data())
        time.sleep(0.5)  # 워커가 처리할 시간

        sender.stop()
        self.assertFalse(sender._worker_thread.is_alive())

    @patch('src.cloud.dynamodb_sender.DynamoDBSender._init_client')
    @patch('src.cloud.dynamodb_sender.DynamoDBSender._write_batch')
    def test_flush_on_stop(self, mock_write, mock_init):
        """종료 시 남은 큐 플러시"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_config(tmpdir)
            sender = DynamoDBSender(config_path=path)

        # 워커 시작 전에 데이터 적재
        for i in range(3):
            sender.send(_sample_data(timestamp_ms=i))

        sender.start()
        time.sleep(1.0)
        sender.stop()

        # _write_batch가 호출되었는지 확인
        self.assertTrue(mock_write.called)


class TestDrainQueue(unittest.TestCase):
    """큐 드레인 테스트"""

    def test_drain_respects_max_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_config(tmpdir)
            sender = DynamoDBSender(config_path=path)

        for i in range(30):
            sender.send(_sample_data(timestamp_ms=i))

        batch = sender._drain_queue(25)
        self.assertEqual(len(batch), 25)
        self.assertEqual(sender.pending_count, 5)

    def test_drain_empty_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_config(tmpdir)
            sender = DynamoDBSender(config_path=path)

        batch = sender._drain_queue(25)
        self.assertEqual(len(batch), 0)


if __name__ == '__main__':
    unittest.main()
