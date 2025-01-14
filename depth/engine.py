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
from lane.engine import LaneEngine

class DepthEngine:
    def __init__(self,
                vis=False,
                tensorRT_engine_path="/home/ircv/projects/deep_daiv/DeepDrive/depth/metric3D_vit_small.engine",
                input_size = (308, 504),
                cam_intrinsics=np.array([
                                            [809.5233, 0, 339.2379],  # f_x, 0, c_x
                                            [0, 808.7865, 265.3243],  # 0, f_y, c_y
                                            [0, 0, 1]
                                        ]),
                grid_size=100,
                cell_size=0.02,
                pitch_angle_degree=-18,
                min_height=0.05,
                max_height=0.1,
                min_occupied=30, ## _obstacleCheck를 위한 변수라 사실상 필요 없을듯?
                # initial_steering=0,
                # initial_throttle=0
                ):

        self.vis = vis
        self.input_size = input_size
        self.cam_intrinsics = cam_intrinsics
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.pitch_angle = np.radians(pitch_angle_degree)
        self.min_height = min_height
        self.max_height = max_height
        self.min_occupied = min_occupied

        # self.camera = self._initialize_camera()
        self.engine = self._load_engine(tensorRT_engine_path)
        # self.car = NvidiaRacecar(throttle_gain=1)
        self.lane_engine = LaneEngine(trt_engine_path="/home/ircv/projects/deep_daiv/DeepDrive/lane/64192.trt"
                                      , input_size=input_size, image_size=(480, 640), num_lanes=2, num_grid=100, cls_num_per_lane=56, save=True)

        self.model_input_size = 1862784
        self.model_output_size = input_size[0] * input_size[1] * 4

        self.d_input = cuda.mem_alloc(int(self.model_input_size))
        self.d_output = cuda.mem_alloc(int(self.model_output_size))
        self.stream = cuda.Stream()
        # self.context = self.engine.create_execution_context()

        # self._initialize_car(initial_steering, initial_throttle)

        self.rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(self.pitch_angle), -np.sin(self.pitch_angle)],
            [0, np.sin(self.pitch_angle), np.cos(self.pitch_angle)]
        ])

        if vis:
            self.dir_name = self._make_dir()

    def _make_dir(self):

        # KST 타임존 설정
        kst = pytz.timezone("Asia/Seoul")

        # 현재 시간을 KST로 가져오기
        now = datetime.now(kst)

        # 현재 년월일시분초 로 디렉토리 생성
        formatted_time = now.strftime("%Y%m%d%H%M%S")
        dir_name = f"occupancy_map_{formatted_time}"
        os.mkdir(dir_name)
        return dir_name


    def _initialize_car(self, initial_steering, initial_throttle):
        self.car.steering = initial_steering
        self.car.throttle = initial_throttle

    def _initialize_camera(self):

        camera = cv2.VideoCapture(
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
            "videoconvert ! appsink",
            cv2.CAP_GSTREAMER
        )
        if not camera.isOpened():
            raise RuntimeError("CSI camera가 인식되지 않습니다.")
        return camera

    # def _load_engine(self, tensorRT_engine_path):
    #     trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    #     with open(tensorRT_engine_path, 'rb') as f:
    #         engine_data = f.read()
    #     return trt_runtime.deserialize_cuda_engine(engine_data)

    def _load_engine(self, tensorRT_engine_path):
        trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(tensorRT_engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {tensorRT_engine_path}")
        
        self.context = engine.create_execution_context()  # ExecutionContext 생성
        return engine


    def _take_picture_using_CSICamera(self):
        last_saved_time = time.time()

        # Capture frame from CSI camera
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("CSI 카메라로 사진을 찍지 못했습니다.")

        ## 카메라 돌리고 밝기 및 선명도 키우기
        frame = self._frame_preprocess(frame)
        return frame

    def _frame_preprocess(self, frame):
        # Rotate the frame 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # 밝기 증가
        brightness_value = 30
        bright_image = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_value)

        # 샤프닝 필터 적용
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        frame = cv2.filter2D(bright_image, -1, kernel)
        return frame

    def _prepare_input(self, rgb_image: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
            h, w = rgb_image.shape[:2]
            scale = min(input_size[0] / h, input_size[1] / w)
            rgb = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

            intrinsic_scaled = [
                                self.cam_intrinsics[0, 0] * scale,
                                self.cam_intrinsics[1, 1] * scale,
                                self.cam_intrinsics[0, 2] * scale,
                                self.cam_intrinsics[1, 2] * scale
                                ]

            padding = [123.675, 116.28, 103.53]  # Mean values for normalization (e.g., ImageNet)
            h, w = rgb.shape[:2]
            pad_h = input_size[0] - h
            pad_w = input_size[1] - w
            pad_h_half = pad_h // 2
            pad_w_half = pad_w // 2
            pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
            rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)

            # Preprocess image to match the input tensor shape (1, 3, H, W)
            rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
            return rgb, intrinsic_scaled, pad_info

    def _infer(self, image_input: np.ndarray):
            # Ensure input is contiguous
            image_input = np.ascontiguousarray(image_input)

            output_shape = self.engine.get_tensor_shape("pred_depth")

            # Transfer data to device
            cuda.memcpy_htod_async(self.d_input, image_input, self.stream)


            # Run inference
            self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)

            # Prepare output array with exact shape
            output_data = np.empty(output_shape, dtype=np.float32)

            # Transfer result back to host
            cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
            self.stream.synchronize()
            return output_data

    def _metric3D_tensorRT(self, input_image):

            # Read and preprocess the image
            rgb_image = input_image[:, :, ::-1]
            ori_shape = rgb_image.shape
            image_input, intrinsic_scaled, pad_info = self._prepare_input(rgb_image, self.input_size)

            # Add batch dimension to the input (1, 3, H, W)
            image_input = np.expand_dims(image_input, axis=0)

            # Perform inference
            depth_map = self._infer(image_input)

            # depth_map을 2로 나누기
            pred_depth = depth_map / 2.0
            # pred_depth = depth_map

            # squeeze와 같은 효과: 불필요한 차원 제거
            pred_depth = pred_depth.squeeze()

            # pad_info를 사용해 패딩 제거
            pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1],
                                    pad_info[2] : pred_depth.shape[1] - pad_info[3]]

            # 원본 크기로 upsampling (bilinear 방식)
            scale_h = ori_shape[0] / pred_depth.shape[0]
            scale_w = ori_shape[1] / pred_depth.shape[1]
            pred_depth_resized = zoom(pred_depth, (scale_h, scale_w), order=1)  # order=1은 bilinear

            # metric 스케일로 변환
            canonical_to_real_scale = intrinsic_scaled[0] / 1000.0  # 1000은 기본 초점 거리
            pred_depth_metric = pred_depth_resized * canonical_to_real_scale

            # depth 값을 0~300으로 클램프
            pred_depth_np = np.clip(pred_depth_metric, 0, 300)

            # return rgb_image, depth_map_resized
            return rgb_image, pred_depth_np

    def _depth_to_occupancy_map(self, rgb_image, depth_map, lane_mask):
        # 카메라 내부 파라미터
        f_x, f_y = self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1]  # 초점 거리
        c_x, c_y = self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2]  # 중심점

        # 필요한 변수들
        height, width = depth_map.shape

        # 3D 좌표 변환
        depth_map = depth_map.astype(float)  # depth_map을 float 타입으로 변환
        y_indices, x_indices = np.indices((height, width))  # (i, j) 인덱스를 2D 배열로 생성

        # 유효한 깊이 값만 필터링
        valid_depth_mask = depth_map > 0

        # 3D 카메라 좌표 계산
        X_camera = (x_indices - c_x) * depth_map / f_x
        Y_camera = (y_indices - c_y) * depth_map / f_y
        Z_camera = depth_map

        # 회전된 좌표 계산
        camera_coordinates = np.stack([X_camera, Y_camera, Z_camera], axis=-1)  # (height, width, 3)
        world_coordinates = np.einsum('ij,klj->kli', self.rotation_matrix, camera_coordinates)  # (height, width, 3)


        x_range = [-self.grid_size * self.cell_size / 2, self.grid_size * self.cell_size / 2]
        z_range = [0, self.grid_size * self.cell_size]

        # Load coordinates (assuming coords is loaded from the given .npy file)
        x_coords = world_coordinates[:, :, 0].reshape(-1)
        y_coords = world_coordinates[:, :, 1].reshape(-1)
        z_coords = world_coordinates[:, :, 2].reshape(-1)

        # Filter based on y condition (e.g., y > 0.5)
        y_mask = (self.min_height < y_coords) & (y_coords < self.max_height)
        x_coords = x_coords[y_mask]
        z_coords = z_coords[y_mask]

        # # Initialize binary occupancy map
        occupancy_map = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Mask points that are within the valid range
        mask = (x_range[0] <= x_coords) & (x_coords <= x_range[1]) & (z_range[0] <= z_coords) & (z_coords <= z_range[1])

        # Filter valid points
        x_valid = x_coords[mask]
        z_valid = z_coords[mask]

        # Compute grid indices for valid points
        x_indices = ((x_valid - x_range[0]) / (x_range[1] - x_range[0]) * self.grid_size).astype(int)
        z_indices = ((z_valid - z_range[0]) / (z_range[1] - z_range[0]) * self.grid_size).astype(int)

        # Ensure indices are within bounds
        x_indices = np.clip(x_indices, 0, self.grid_size - 1)
        z_indices = np.clip(z_indices, 0, self.grid_size - 1)

        # Mark the corresponding cells in the occupancy map as occupied
        occupancy_map[z_indices, x_indices] = 1 # 장애물

        # 차선 정보 추가_________________________________________________________________________
        lane_mask_flat = lane_mask.reshape(-1)  # 1D로 변환
        lane_x_coords = world_coordinates[:, :, 0].reshape(-1)[lane_mask_flat > 0]  # 활성화된 차선 픽셀의 X 좌표
        lane_z_coords = world_coordinates[:, :, 2].reshape(-1)[lane_mask_flat > 0]  # 활성화된 차선 픽셀의 Z 좌표

        # 차선 좌표 필터링
        lane_mask_valid = (x_range[0] <= lane_x_coords) & (lane_x_coords <= x_range[1]) & \
                          (z_range[0] <= lane_z_coords) & (lane_z_coords <= z_range[1])
        lane_x_coords = lane_x_coords[lane_mask_valid]
        lane_z_coords = lane_z_coords[lane_mask_valid]

        # 차선 좌표를 격자 인덱스로 변환
        lane_x_indices = ((lane_x_coords - x_range[0]) / (x_range[1] - x_range[0]) * self.grid_size).astype(int)
        lane_z_indices = ((lane_z_coords - z_range[0]) / (z_range[1] - z_range[0]) * self.grid_size).astype(int)

        # 인덱스가 격자 범위를 벗어나지 않도록 클리핑
        lane_x_indices = np.clip(lane_x_indices, 0, self.grid_size - 1)
        lane_z_indices = np.clip(lane_z_indices, 0, self.grid_size - 1)

        # 차선 정보 추가 (2로 설정)
        occupancy_map[lane_z_indices, lane_x_indices] = 2

        # 결과 출력
        return occupancy_map, world_coordinates[:, :, 1]


    def _visualize(self, rgb_image, depth_map, occupancy_map):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

        ## input image 그리기
        ax[0].imshow(rgb_image)
        ax[0].set_title('RGB image')

        # Depth map 정규화 후 그리기
        depth_map_log_scaled = np.log(depth_map - np.min(depth_map) + 1)  # 1을 더해 0을 방지

        ax[1].imshow(depth_map_log_scaled, cmap='plasma')
        ax[1].set_title('Predicted Depth Map')

        ## Occupancy Map 그리기
        ax[2].imshow(occupancy_map, cmap='gray', origin='lower')
        ax[2].set_title('Occupancy Map')

        # x축 중앙을 0으로 설정
        map_width = occupancy_map.shape[1]
        tick_positions = np.linspace(0, map_width - 1, 5)  # 5개의 주요 tick 위치
        tick_labels = np.linspace(-map_width // 2, map_width // 2, 5).astype(int)  # 중앙 0을 기준으로 -와 +

        ax[2].set_xticks(tick_positions)
        ax[2].set_xticklabels(tick_labels)


        # Save the figure to a file[]
        plt.tight_layout()

        # KST 타임존 설정
        kst = pytz.timezone("Asia/Seoul")

        # 현재 시간을 KST로 가져오기
        now = datetime.now(kst)

        image_name = f"{self.dir_name}/{now.strftime('%H%M%S%f')[:-3]}.png"
        plt.savefig(image_name)

        plt.close()

    def _obstacleCheck(self, occupancy_map):

        half = self.grid_size // 2

        left_map, right_map = occupancy_map[half:, :half], occupancy_map[half:, half:]
        left_map_sum, right_map_sum = np.sum(left_map), np.sum(right_map)
        total_sum = left_map_sum + right_map_sum


        print(left_map_sum, right_map_sum)

        # 배열의 크기
        rows, cols = occupancy_map.shape

        # 각 좌표의 인덱스 생성
        y_indices, x_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

        centroid = (0, 0)
        # 무게중심 계산
        total_weight = occupancy_map.sum()
        if total_weight == 0:
            # 배열이 전부 0인 경우 무게중심은 정의되지 않음
            centroid = (0, 0)
        else:
            x_centroid = (x_indices * occupancy_map).sum() / total_weight
            y_centroid = (y_indices * occupancy_map).sum() / total_weight
            centroid = (y_centroid, x_centroid)


        if 2 < centroid[1] < half:
            print("---------------------무게중심:", centroid, "오오오오오오오오오")
            print("오른쪽")
            self.car.steering = 0.3

        elif centroid[1] > half+2:
            print("---------------------무게중심:", centroid, "왼왼왼왼왼왼왼왼왼")
            print("왼쪽")
            self.car.steering = -0.3
        elif centroid[1] == 0:
            print("---------------------무게중심:", centroid, "가가가가가가가가가")
            print("가운데")
            self.car.steering = 0

    def run(self, frame):
        print("Running DepthEngine...")
        start = time.time()

        # Step 0: CSI 카메라로 사진을 찍고 저장하기
        # frame = self._take_picture_using_CSICamera()
        # print("사진찍기 완료")


        # Step 1: 사진파일로부터 rgb정보, depth 정보 추출하기
        rgb_image, depth_map = self._metric3D_tensorRT(frame)
        print("Depth info 추출 완료")

        # Step 2: LaneEngine으로 차선 정보 추출
        lane_mask = self.lane_engine.run(frame)  # LaneEngine 호출
        print("차선 정보 추출 완료")

        # Step 2: Depth 정보를 Occupancy map으로 변환하기
        occupancy_map, Y_world = self._depth_to_occupancy_map(rgb_image, depth_map, lane_mask)
        print("Occupany Map 추출 완료")

        # # (Optinal) Step 4: 바퀴 움직이기
        # self._obstacleCheck(occupancy_map)
        # print("바퀴 이동 완료")


        if self.vis:
            # (Optional) Step 3: Occupancy map 시각화하기
            self._visualize(rgb_image, depth_map, occupancy_map)
            overlay_z_on_rgb(rgb_image, depth_map, self.cam_intrinsics, Y_world) ## z값과 이미지를 겹쳐놓는 함수
            print("Depth Engine 시각화 완료")

        end = time.time()

        print(f"전체시간 : {end - start:.4f}")

        return occupancy_map