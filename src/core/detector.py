import logging
import os
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# TensorRT는 Jetson 시스템 패키지로만 제공됨
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT를 찾을 수 없습니다. 검출기 비활성화.")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 - CUDA 컨텍스트 초기화에 필요
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    if TRT_AVAILABLE:
        logger.warning("PyCUDA를 찾을 수 없습니다. TensorRT 추론 불가.")


class YOLOv8Detector:
    """TensorRT 10.3 기반 YOLOv8 객체 검출기

    기능:
    - ONNX → TensorRT 엔진 변환 및 캐싱
    - FP16 추론 (Jetson Orin Nano GPU 활용)
    - NMS 후처리 포함
    """

    # COCO 클래스 중 사람(person) 인덱스
    PERSON_CLASS_ID = 0

    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.45,
                 input_size=640, fp16=True, target_classes=None):
        """
        Args:
            model_path: ONNX 모델 파일 경로
            conf_threshold: 검출 신뢰도 임계값
            nms_threshold: NMS IoU 임계값
            input_size: 모델 입력 크기 (정사각형)
            fp16: FP16 추론 사용 여부
            target_classes: 검출할 클래스 ID 리스트 (None이면 전체)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.fp16 = fp16
        self.target_classes = target_classes

        self.engine = None
        self.context = None
        self._input_buf = None
        self._output_buf = None
        self._d_input = None
        self._d_output = None
        self._stream = None
        self._ready = False

    def initialize(self):
        """엔진 로드 또는 빌드 후 추론 준비"""
        if not TRT_AVAILABLE or not CUDA_AVAILABLE:
            logger.error("TensorRT 또는 PyCUDA가 없어 검출기를 초기화할 수 없습니다")
            return False

        if not os.path.exists(self.model_path):
            logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            return False

        engine_path = self._get_engine_path()

        if os.path.exists(engine_path):
            logger.info(f"캐시된 TensorRT 엔진 로드: {engine_path}")
            self.engine = self._load_engine(engine_path)
        else:
            logger.info(f"ONNX → TensorRT 엔진 변환 시작: {self.model_path}")
            self.engine = self._build_engine(self.model_path, engine_path)

        if self.engine is None:
            logger.error("TensorRT 엔진 생성 실패")
            return False

        self.context = self.engine.create_execution_context()
        self._allocate_buffers()
        self._ready = True
        logger.info("YOLOv8 검출기 초기화 완료")
        return True

    def _get_engine_path(self):
        """ONNX 파일명 기반으로 엔진 캐시 경로 생성"""
        base = os.path.splitext(self.model_path)[0]
        precision = "fp16" if self.fp16 else "fp32"
        return f"{base}_{precision}.engine"

    def _build_engine(self, onnx_path, engine_path):
        """ONNX 모델에서 TensorRT 엔진 빌드"""
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX 파싱 오류: {parser.get_error(i)}")
                return None

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB

        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 모드 활성화")

        start = time.time()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            return None
        elapsed = time.time() - start
        logger.info(f"엔진 빌드 완료 ({elapsed:.1f}초)")

        # 엔진 캐시 저장
        with open(engine_path, 'wb') as f:
            f.write(serialized)
        logger.info(f"엔진 캐시 저장: {engine_path}")

        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(serialized)

    def _load_engine(self, engine_path):
        """캐시된 TensorRT 엔진 로드"""
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        """입출력 GPU/CPU 버퍼 할당"""
        self._stream = cuda.Stream()

        # 입력 텐서
        input_name = self.engine.get_tensor_name(0)
        input_shape = self.engine.get_tensor_shape(input_name)
        input_size = int(np.prod(input_shape)) * np.float32().itemsize
        self._input_buf = np.zeros(input_shape, dtype=np.float32)
        self._d_input = cuda.mem_alloc(input_size)

        # 출력 텐서
        output_name = self.engine.get_tensor_name(1)
        output_shape = self.engine.get_tensor_shape(output_name)
        output_size = int(np.prod(output_shape)) * np.float32().itemsize
        self._output_buf = np.zeros(output_shape, dtype=np.float32)
        self._d_output = cuda.mem_alloc(output_size)

        # I/O 텐서 주소 설정
        self.context.set_tensor_address(input_name, int(self._d_input))
        self.context.set_tensor_address(output_name, int(self._d_output))

        self._input_name = input_name
        self._output_name = output_name

        logger.info(f"입력: {input_name} {input_shape}, 출력: {output_name} {output_shape}")

    def detect(self, frame):
        """프레임에서 객체 검출

        Args:
            frame: BGR numpy array (OpenCV 프레임)

        Returns:
            list of dict: [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class_id': int}, ...]
        """
        if not self._ready:
            return []

        img_h, img_w = frame.shape[:2]
        input_tensor = self._preprocess(frame)

        # GPU로 입력 전송
        np.copyto(self._input_buf, input_tensor)
        cuda.memcpy_htod_async(self._d_input, self._input_buf, self._stream)

        # 추론
        self.context.execute_async_v3(stream_handle=self._stream.handle)

        # 결과 수신
        cuda.memcpy_dtoh_async(self._output_buf, self._d_output, self._stream)
        self._stream.synchronize()

        # 후처리
        detections = self._postprocess(self._output_buf, img_w, img_h)
        return detections

    def _preprocess(self, frame):
        """YOLOv8 입력 전처리: letterbox resize + normalize"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Letterbox resize
        h, w = img.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dy, dx = (self.input_size - new_h) // 2, (self.input_size - new_w) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = img

        # HWC → CHW, normalize to [0, 1]
        tensor = canvas.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)
        return np.ascontiguousarray(tensor)

    def _postprocess(self, output, img_w, img_h):
        """YOLOv8 출력 후처리: 좌표 변환 + NMS

        YOLOv8 출력 형태: [1, 84, 8400] (cx, cy, w, h, class_scores...)
        """
        # [1, 84, 8400] → [8400, 84]
        preds = output[0].T

        # 클래스 스코어 및 ID
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # 신뢰도 필터
        mask = confidences > self.conf_threshold
        if self.target_classes is not None:
            class_mask = np.isin(class_ids, self.target_classes)
            mask = mask & class_mask

        preds = preds[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(preds) == 0:
            return []

        # cx, cy, w, h → x1, y1, x2, y2
        boxes = preds[:, :4].copy()
        boxes[:, 0] = preds[:, 0] - preds[:, 2] / 2  # x1
        boxes[:, 1] = preds[:, 1] - preds[:, 3] / 2  # y1
        boxes[:, 2] = preds[:, 0] + preds[:, 2] / 2  # x2
        boxes[:, 3] = preds[:, 1] + preds[:, 3] / 2  # y2

        # letterbox 좌표 → 원본 이미지 좌표
        scale = min(self.input_size / img_w, self.input_size / img_h)
        dx = (self.input_size - img_w * scale) / 2
        dy = (self.input_size - img_h * scale) / 2

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale

        # 범위 클리핑
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), confidences.tolist(),
            self.conf_threshold, self.nms_threshold
        )

        results = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                results.append({
                    'bbox': boxes[i].astype(int).tolist(),
                    'confidence': float(confidences[i]),
                    'class_id': int(class_ids[i]),
                })

        return results

    @property
    def is_ready(self):
        return self._ready

    def destroy(self):
        """리소스 해제"""
        self._ready = False
        self.context = None
        self.engine = None
        self._d_input = None
        self._d_output = None
        self._stream = None
        logger.info("검출기 리소스 해제 완료")
