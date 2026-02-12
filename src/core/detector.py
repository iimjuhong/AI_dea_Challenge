import ctypes
import logging
import os
import time
from collections import deque

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


# --- ctypes CUDA Runtime wrapper (pycuda 대체) ---

class CudaRT:
    """ctypes를 사용한 CUDA Runtime API 래퍼

    PyCUDA 없이 cudaMalloc/cudaMemcpy/cudaFree/cudaStream 등을 직접 호출한다.
    """

    # cudaMemcpyKind enum
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def __init__(self):
        # libcudart.so 로드 시도
        lib_paths = [
            '/usr/local/cuda-12.6/lib64/libcudart.so',
            '/usr/local/cuda/lib64/libcudart.so',
            'libcudart.so',
        ]
        self._lib = None
        for path in lib_paths:
            try:
                self._lib = ctypes.CDLL(path)
                logger.info(f"CUDA Runtime 로드 성공: {path}")
                break
            except OSError:
                continue

        if self._lib is None:
            raise RuntimeError("libcudart.so를 로드할 수 없습니다")

        # 함수 시그니처 설정
        self._lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.cudaMalloc.restype = ctypes.c_int

        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaFree.restype = ctypes.c_int

        self._lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        self._lib.cudaMemcpy.restype = ctypes.c_int

        self._lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.cudaStreamCreate.restype = ctypes.c_int

        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamDestroy.restype = ctypes.c_int

        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamSynchronize.restype = ctypes.c_int

    def malloc(self, size):
        """GPU 메모리 할당, 디바이스 포인터(int) 반환"""
        ptr = ctypes.c_void_p()
        err = self._lib.cudaMalloc(ctypes.byref(ptr), size)
        if err != 0:
            raise RuntimeError(f"cudaMalloc 실패 (err={err})")
        return ptr.value

    def free(self, d_ptr):
        """GPU 메모리 해제"""
        err = self._lib.cudaFree(ctypes.c_void_p(d_ptr))
        if err != 0:
            logger.warning(f"cudaFree 경고 (err={err})")

    def memcpy_htod(self, d_ptr, host_arr):
        """호스트 numpy array → 디바이스 복사"""
        err = self._lib.cudaMemcpy(
            ctypes.c_void_p(d_ptr),
            host_arr.ctypes.data_as(ctypes.c_void_p),
            host_arr.nbytes,
            self.cudaMemcpyHostToDevice,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy HtoD 실패 (err={err})")

    def memcpy_dtoh(self, host_arr, d_ptr):
        """디바이스 → 호스트 numpy array 복사"""
        err = self._lib.cudaMemcpy(
            host_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(d_ptr),
            host_arr.nbytes,
            self.cudaMemcpyDeviceToHost,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy DtoH 실패 (err={err})")

    def stream_create(self):
        """CUDA 스트림 생성, 핸들(int) 반환"""
        handle = ctypes.c_void_p()
        err = self._lib.cudaStreamCreate(ctypes.byref(handle))
        if err != 0:
            raise RuntimeError(f"cudaStreamCreate 실패 (err={err})")
        return handle.value

    def stream_synchronize(self, stream_handle):
        """CUDA 스트림 동기화"""
        err = self._lib.cudaStreamSynchronize(ctypes.c_void_p(stream_handle))
        if err != 0:
            raise RuntimeError(f"cudaStreamSynchronize 실패 (err={err})")

    def stream_destroy(self, stream_handle):
        """CUDA 스트림 파괴"""
        err = self._lib.cudaStreamDestroy(ctypes.c_void_p(stream_handle))
        if err != 0:
            logger.warning(f"cudaStreamDestroy 경고 (err={err})")


# CudaRT 싱글톤
try:
    _cuda_rt = CudaRT()
    CUDA_AVAILABLE = True
except RuntimeError as e:
    _cuda_rt = None
    CUDA_AVAILABLE = False
    logger.warning(f"CUDA Runtime 초기화 실패: {e}")


class YOLOv8Detector:
    """TensorRT 기반 YOLOv8 객체 검출기 (ctypes CUDA 사용)

    기능:
    - ONNX → TensorRT 엔진 변환 및 캐싱
    - FP16 추론 (Jetson Orin Nano GPU 활용)
    - NMS 후처리 포함
    - FPS 측정
    """

    PERSON_CLASS_ID = 0

    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.45,
                 input_size=640, fp16=True, target_classes=None):
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
        self._ready = False

        # FPS 측정
        self._fps = 0.0
        self._frame_times = deque(maxlen=30)

        # 최근 검출 수
        self._last_det_count = 0

    def initialize(self):
        """엔진 로드 또는 빌드 후 추론 준비"""
        if not TRT_AVAILABLE or not CUDA_AVAILABLE:
            logger.error("TensorRT 또는 CUDA Runtime이 없어 검출기를 초기화할 수 없습니다")
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
        """입출력 GPU/CPU 버퍼 할당 (ctypes CUDA)"""
        # 입력 텐서
        input_name = self.engine.get_tensor_name(0)
        input_shape = self.engine.get_tensor_shape(input_name)
        input_nbytes = int(np.prod(input_shape)) * np.float32().itemsize
        self._input_buf = np.zeros(input_shape, dtype=np.float32)
        self._d_input = _cuda_rt.malloc(input_nbytes)

        # 출력 텐서
        output_name = self.engine.get_tensor_name(1)
        output_shape = self.engine.get_tensor_shape(output_name)
        output_nbytes = int(np.prod(output_shape)) * np.float32().itemsize
        self._output_buf = np.zeros(output_shape, dtype=np.float32)
        self._d_output = _cuda_rt.malloc(output_nbytes)

        # I/O 텐서 주소 설정 (execute_v2 용)
        self.context.set_tensor_address(input_name, self._d_input)
        self.context.set_tensor_address(output_name, self._d_output)

        self._input_name = input_name
        self._output_name = output_name
        self._bindings = [self._d_input, self._d_output]

        logger.info(f"입력: {input_name} {list(input_shape)}, "
                    f"출력: {output_name} {list(output_shape)}")

    def detect(self, frame):
        """프레임에서 객체 검출

        Args:
            frame: BGR numpy array (OpenCV 프레임)

        Returns:
            list of dict: [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'class_id': int}, ...]
        """
        if not self._ready:
            return []

        t_start = time.time()

        img_h, img_w = frame.shape[:2]
        input_tensor = self._preprocess(frame)

        # GPU로 입력 전송 (동기)
        np.copyto(self._input_buf, input_tensor)
        _cuda_rt.memcpy_htod(self._d_input, self._input_buf)

        # 추론 (execute_v2 — 동기식)
        self.context.execute_v2(self._bindings)

        # 결과 수신 (동기)
        _cuda_rt.memcpy_dtoh(self._output_buf, self._d_output)

        # 후처리
        detections = self._postprocess(self._output_buf, img_w, img_h)

        # FPS 측정
        elapsed = time.time() - t_start
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 0:
            self._fps = len(self._frame_times) / sum(self._frame_times)

        self._last_det_count = len(detections)
        return detections

    def _preprocess(self, frame):
        """YOLOv8 입력 전처리: letterbox resize + normalize"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dy, dx = (self.input_size - new_h) // 2, (self.input_size - new_w) // 2
        canvas[dy:dy + new_h, dx:dx + new_w] = img

        tensor = canvas.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        tensor = np.expand_dims(tensor, axis=0)
        return np.ascontiguousarray(tensor)

    def _postprocess(self, output, img_w, img_h):
        """YOLOv8 출력 후처리: 좌표 변환 + NMS

        YOLOv8 출력 형태: [1, 84, 8400] (cx, cy, w, h, class_scores...)
        """
        preds = output[0].T

        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confidences > self.conf_threshold
        if self.target_classes is not None:
            class_mask = np.isin(class_ids, self.target_classes)
            mask = mask & class_mask

        preds = preds[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(preds) == 0:
            return []

        boxes = preds[:, :4].copy()
        boxes[:, 0] = preds[:, 0] - preds[:, 2] / 2
        boxes[:, 1] = preds[:, 1] - preds[:, 3] / 2
        boxes[:, 2] = preds[:, 0] + preds[:, 2] / 2
        boxes[:, 3] = preds[:, 1] + preds[:, 3] / 2

        scale = min(self.input_size / img_w, self.input_size / img_h)
        dx = (self.input_size - img_w * scale) / 2
        dy = (self.input_size - img_h * scale) / 2

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

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

    def get_fps(self):
        """현재 추론 FPS 반환"""
        return round(self._fps, 1)

    def get_detection_count(self):
        """마지막 프레임의 검출 수 반환"""
        return self._last_det_count

    @property
    def is_ready(self):
        return self._ready

    def destroy(self):
        """리소스 해제"""
        self._ready = False
        self.context = None
        self.engine = None
        if self._d_input is not None and _cuda_rt is not None:
            _cuda_rt.free(self._d_input)
            self._d_input = None
        if self._d_output is not None and _cuda_rt is not None:
            _cuda_rt.free(self._d_output)
            self._d_output = None
        logger.info("검출기 리소스 해제 완료")
