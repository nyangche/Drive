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

class TrackingEngine:
    def __init__(
        self,
        max_steer=0.4,
        control_method="pid",
        kp=0.1,
        kd=0.01,
        ki=0.01,
        throttle=0.75,
        max_throttle=0.15,
        car_width_cells=2,  # JetRacer 너비를 Occupancy Map 셀 단위로 설정
        car_length_cells=2  # JetRacer 길이를 Occupancy Map 셀 단위로 설정
    ):
        self.max_steer = max_steer
        self.control_method = control_method
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.previous_error = 0
        self.integral_error = 0
        self.car = NvidiaRacecar()
        self.throttle = min(throttle, max_throttle)
        self.car_width_cells = car_width_cells
        self.car_length_cells = car_length_cells

    def expand_obstacles(self, occupancy_map):
        """장애물 주변을 JetRacer 크기만큼 병렬 연산으로 확장"""
        structure = np.ones((2 * self.car_width_cells + 1, 2 * self.car_length_cells + 1))
        expanded_map = binary_dilation(occupancy_map, structure=structure).astype(int)
        return expanded_map

    def find_goal(self, occupancy_map, threshold=10):
        """상단부에서 가장 긴 통로의 중간값을 Goal로 설정 (임계값 기반)"""
        row_data = occupancy_map[10]
        goal_candidates = np.where(row_data == 0)[0]
        if len(goal_candidates) > 0:
            longest_chunk, current_chunk = [], []
            for col in goal_candidates:
                if not current_chunk or col == current_chunk[-1] + 1:
                    current_chunk.append(col)
                    if len(current_chunk) >= threshold:
                        return (10, current_chunk[len(current_chunk) // 2])  # 임계값 도달 시 즉시 Goal 반환
                else:
                    if len(current_chunk) > len(longest_chunk):
                        longest_chunk = current_chunk
                    current_chunk = [col]
            if len(longest_chunk) > 0:
                goal_col = longest_chunk[len(longest_chunk) // 2]
                return (10, goal_col)
        return None

    def a_star(self, occupancy_grid, start, goal):
        """A* 알고리즘 (후진 방향 제거)"""
        grid_size = occupancy_grid.shape[0]
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}

        directions = [(-1, 0), (0, -1), (0, 1)]  # 후진(1, 0) 제거

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                    if occupancy_grid[neighbor] == 1:
                        continue
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def calculate_steering(self, path, current_position):
        """PID 기반 조향 계산"""
        if len(path) < 2:
            return 0
        target_index = min(50, len(path) - 1)
        target_position = path[target_index]
        error = target_position[1] - current_position[1]
        steer = self.kp * error + self.kd * (error - self.previous_error) + self.ki * self.integral_error
        self.previous_error = error
        self.integral_error += error
        return np.clip(steer, -self.max_steer, self.max_steer)

    def run(self, occupancy_map):
        print("Running TrackingEngine...")
        """경로 탐색 및 주행"""
        if occupancy_map is None:
            return 0, 0

        # 1. 장애물 확장
        expanded_map = self.expand_obstacles(occupancy_map)

        # 2. 시작점 수정: 중앙 하단부
        start = (expanded_map.shape[0] - 1, expanded_map.shape[1] // 2)

        # 3. Goal 탐색
        goal = self.find_goal(expanded_map)
        print(f"설정된 Goal: {goal}")

        if goal:
            path = self.a_star(expanded_map, start, goal)
            if path:
                print(f"탐색된 경로: {path}")
                steer = self.calculate_steering(path, start)
                self.car.steering = steer
                self.car.throttle = self.throttle
                return self.throttle, steer
            else:
                print("경로 탐색 실패.")
        else:
            print("Goal을 찾을 수 없습니다.")

        self.car.steering = 0
        self.car.throttle = 0
        return 0, 0