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
import heapq
from jetracer.nvidia_racecar import NvidiaRacecar
from scipy.ndimage import binary_dilation
from scipy.ndimage import zoom  # for resizing
import time
from typing import Tuple
from datetime import datetime
import pytz
import torch
import matplotlib.pyplot as plt


class LaneEngine:
    def __init__(self, trt_engine_path, input_size=(288, 800), image_size=(480, 640), num_lanes=2, num_grid=100, cls_num_per_lane=56, save=False):
        # 생성자 초기화
        self.trt_engine_path = trt_engine_path
        self.input_size = input_size
        self.image_size = image_size
        self.num_lanes = num_lanes
        self.num_grid = num_grid
        self.cls_num_per_lane = cls_num_per_lane
        self.save = save
        self.row_anchor = [
            64,  68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284
        ]

        # TensorRT 엔진 로드
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(self.trt_engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 메모리 할당
        self.h_input = cuda.pagelocked_empty(trt.volume((1, 3, *self.input_size)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume((1, self.num_grid + 1, len(self.row_anchor), self.num_lanes)), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.cuda_stream = cuda.Stream()

    def preprocess(self, image):
        # 입력 이미지를 전처리 (크기 조정 및 정규화)
        image = cv2.resize(image, self.input_size[::-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def rescale_row_anchor(self):
        # row_anchor를 현재 이미지 크기에 맞게 조정
        scale_f = lambda x: int((x * 1.0 / self.input_size[0]) * self.image_size[0])
        return list(map(scale_f, self.row_anchor))

    def infer(self, image):
        # 추론 수행
        np.copyto(self.h_input, image.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.cuda_stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.cuda_stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()
        outputs = self.h_output.reshape((self.num_grid + 1, len(self.row_anchor), self.num_lanes))
        return outputs

    def postprocess(self, outputs):
        # 추론 결과 후처리
        rescaled_row_anchor = self.rescale_row_anchor()
        out_loc = np.argmax(outputs, axis=0)
        prob = scipy.special.softmax(outputs[:-1, :, :], axis=0)
        idx = np.arange(self.num_grid).reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        loc[out_loc == self.num_grid] = self.num_grid
        out_loc = loc

        # 차선 좌표 계산
        lanes = []
        for i in range(out_loc.shape[1]):
            out_i = out_loc[:, i]
            lane = [
                int(round((loc + 0.5) * self.image_size[1] / (self.num_grid - 1))) if loc != self.num_grid else -2
                for loc in out_i
            ]
            lanes.append(lane)

        coords = []
        for lane in lanes:
            coord = []
            for i, y in enumerate(rescaled_row_anchor):
                if lane[i] == -2:
                    continue
                coord.append((lane[i], y))
            coords.append(coord)

        return coords

    def visualize(self, image_path, coords):
        # 차선을 이미지 위에 시각화
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for coord in coords:
            for i in range(len(coord) - 1):
                cv2.circle(image, coord[i], 5, (255, 0, 0), -1)

        return Image.fromarray(image)

    def run(self, frame):
        print("Running LaneEngine...")
        # 입력 프레임 전처리
        preprocessed_image = self.preprocess(frame)

        # 추론 수행
        outputs = self.infer(preprocessed_image)

        # 후처리하여 차선 좌표 얻기
        coords = self.postprocess(outputs)

        # 차선이 감지되지 않으면 None 반환
        if not coords or all(len(coord) == 0 for coord in coords):
            return None

        # 차선 마스크 생성 (0과 255로 이루어진 이진화 이미지)
        lane_mask = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        for coord in coords:
            for x, y in coord:
                if 0 <= x < self.image_size[1] and 0 <= y < self.image_size[0]:
                    lane_mask[y, x] = 255

        # 시각화 결과 저장 (self.save가 True일 경우)
        # if self.save:
        #     combined = np.zeros((self.image_size[0], self.image_size[1] * 2, 3), dtype=np.uint8)
        #     original_resized = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
        #     lane_mask_colored = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        #     combined[:, :self.image_size[1]] = original_resized
        #     combined[:, self.image_size[1]:] = lane_mask_colored
        #     base_name = os.path.basename(image_path).split('.')[0]

        #     # 저장 경로
        #     output_path = f"/sample_data/{base_name}_lane_detection.jpg"
        #     cv2.imwrite(output_path, combined)
        #     print(f"시각화 결과가 저장되었습니다: {output_path}")

        if self.save:
            combined = np.zeros((self.image_size[0], self.image_size[1] * 2, 3), dtype=np.uint8)
            original_resized = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            lane_mask_colored = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            combined[:, :self.image_size[1]] = original_resized
            combined[:, self.image_size[1]:] = lane_mask_colored
            
            # 파일 이름을 현재 시간 기반으로 생성
            output_path = f"/sample_data/lane_detection_{int(time.time())}.jpg"
            cv2.imwrite(output_path, combined)
            print(f"시각화 결과가 저장되었습니다: {output_path}")


        return lane_mask