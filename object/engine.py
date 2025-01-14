import os
from pathlib import Path
import cv2
import scipy
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from torchvision import transforms
from jetracer.nvidia_racecar import NvidiaRacecar
import time
from typing import Tuple
from datetime import datetime
import pytz
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO


class YOLOEngine:
    def __init__(self, model_path, sensor_id=0, input_size=(640, 640)):
        self.model_path = model_path  # 엔진 경로
        self.input_size = input_size  # 모델 입력 크기
        self.car = NvidiaRacecar()  # JetRacer 객체 생성
        self.stop_signals = ["traffic_red", "traffic_yellow", "sign_stop"]
        self.slow_signals = ["sign_speed_30", "traffic_off", "sign_slow"]
        self.go_signals = ["traffic_green"]
        self.very_slow_signals = ["sign_kid"]
        self.default_speed = 0.20
        self.slow_speed = 0.19
        self.stop_speed = 0
        self.stop_seconds = 0.1
        self.very_slow_speed = 0.18
        self.conf_threshold = 0.25
        self.sensor_id = sensor_id  # GStreamer 사용 시 카메라 ID 리스트
        # self.load_engine()
        self.model = YOLO(model_path)
        # self.cap = [cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=id, flip_method=0),
        #                              cv2.CAP_GSTREAMER) for id in self.sensor_id]
        print(cv2.getBuildInformation())
        # self.cap = cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=sensor_id), cv2.CAP_GSTREAMER)
        # self.cap = [
        #     cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=id, flip_method=0), cv2.CAP_GSTREAMER)
        #     for id in self.sensor_id
        # ]
        # self.cap = cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=sensor_id, flip_method=0), cv2.CAP_GSTREAMER)


    def gstreamer_pipeline(self, sensor_id=0, capture_width=1280, capture_height=720,
                           display_width=640, display_height=480, framerate=30, flip_method=0):
        """GStreamer 파이프라인 설정"""
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
            f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width={display_width}, height={display_height}, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        )

    def load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self.allocate_buffers()

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            buffer = cuda.mem_alloc(size * dtype().itemsize)
            bindings.append(int(buffer))
            if self.engine.binding_is_input(binding):
                inputs.append(buffer)
            else:
                outputs.append(buffer)
        return inputs, outputs, bindings

    def preprocess(self, frame):
        """이미지를 YOLO 모델 입력 형식으로 전처리"""
        h, w = frame.shape[:2]
        input_h, input_w = self.input_size

        scale = min(input_w / w, input_h / h)
        resized_w, resized_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(frame, (resized_w, resized_h))

        padded_img = np.full((input_h, input_w, 3), 128, dtype=np.uint8)
        pad_top = (input_h - resized_h) // 2
        pad_left = (input_w - resized_w) // 2
        padded_img[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized_img

        img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def infer(self, frame):
        """YOLO 모델로 추론 실행"""
        # 모델 추론 실행
        output = self.model(frame)  # YOLO 모델 추론
        cv2.imwrite("results.png", output[0].plot())  # 시각화 결과 저장

        # 출력 결과 파싱
        results = []
        for detection in output[0].boxes:  # YOLOv8 모델에서 추론된 객체 리스트
            box = detection.xyxy[0].cpu().numpy()  # 경계 상자 좌표 [x1, y1, x2, y2]
            conf = detection.conf.cpu().numpy()  # 신뢰도 점수
            cls = int(detection.cls.cpu().numpy())  # 클래스 ID
            results.append([box[0], box[1], box[2], box[3], conf, cls])  # [x1, y1, x2, y2, conf, cls]

        # 결과를 Numpy 배열로 변환
        results_array = np.array(results)
        return results_array


    def parse_output(self, output):
        """YOLOv8 모델 출력 파싱 및 우선순위에 따른 신호 선택"""
        output = output.reshape(-1, 6)
        class_ids = output[:, 5].astype(int)
        confidences = output[:, 4]
        boxes = output[:, :4]

        detections = [(class_id, conf, box) for class_id, conf, box in zip(class_ids, confidences, boxes) if conf > self.conf_threshold]

        if not detections:
            return "traffic_off"  # 아무 신호도 없을 때

        traffic_signals = []  # 신호등 감지 결과
        signs = []            # 표지판 감지 결과

        # 감지된 객체를 신호등과 표지판으로 분류
        for class_id, conf, box in detections:
            signal_name = self.model.names[int(class_id)]
            # if signal_name in self.stop_signals + self.slow_signals + self.go_signals + self.very_slow_signals:
            if "traffic" in signal_name:
                traffic_signals.append((signal_name, conf, box))
            elif "sign" in signal_name:
                signs.append((signal_name, conf, box))

        # 1. 신호등 우선 처리
        if traffic_signals:
            # 신호등 중 가장 신뢰도가 높은 신호 선택
            traffic_signals.sort(key=lambda x: x[1], reverse=True)  # 신뢰도 기준 내림차순 정렬
            return traffic_signals[0][0]  # 가장 신뢰도 높은 신호 반환

        # 2. 표지판 중 바운딩 박스 큰 것 처리
        if signs:
            signs.sort(key=lambda x: (x[2][2] - x[2][0]) * (x[2][3] - x[2][1]), reverse=True)  # 바운딩 박스 면적 기준 정렬
            return signs[0][0]  # 가장 큰 표지판 반환

        return None # "traffic_off"  # 기본값 반환

    def detect(self, frame):
        """단일 프레임에서 신호등 상태 감지"""
        output = self.infer(frame)
        return self.parse_output(output)

    def detect_from_camera(self):
        """카메라로부터 프레임 읽어 신호등 상태 감지"""
        ret, frame = self.cap.read()
        if not ret:
            print(f"Failed to capture frame from camera {self.sensor_id}.")
            return None
        return self.detect(frame)

    
    def run(self, frame):
        # Input -> frame: HxWxC의 np.ndarray 
        # Return -> throttle: jetracer에 넣어줄 실제 속도 값 
        ## frame -> speed
        # infer
        # speed
        # return
        
        # 1. YOLO 모델로 추론
        detected_class = self.detect(frame)

        # 2. 추론된 클래스에 따라 속도 결정
        if detected_class in self.stop_signals:
            print(f"Detected stop signal: {detected_class}. Stopping the car.")
            return self.stop_speed
        elif detected_class in self.go_signals:
            print(f"Detected go signal: {detected_class}. Driving forward.")
            return self.default_speed
        elif detected_class in self.slow_signals:
            print(f"Detected slow signal: {detected_class}. Slowing down.")
            return self.slow_speed
        elif detected_class in self.very_slow_signals:
            print(f"Detected very slow signal: {detected_class}. Driving very slowly.")
            return self.very_slow_speed
        else:
            print(f"Unknown signal detected ({detected_class}).")
            return None

        