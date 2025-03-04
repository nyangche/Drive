import os
import cv2
import numpy as np

from object.engine import YOLOEngine
from lane.engine import LaneEngine
from depth.engine import DepthEngine
from planning.engine import TrackingEngine
from jetracer.nvidia_racecar import NvidiaRacecar

class State:
    def __init__(self, input_path, is_video=False):
        self.is_video = is_video

        if is_video:
            # 비디오 파일 처리
            self.camera = cv2.VideoCapture(input_path)
            if not self.camera.isOpened():
                raise RuntimeError(f"비디오 파일을 열 수 없습니다: {input_path}")
        else:
            # 이미지 폴더 처리
            self.image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if not self.image_files:
                raise RuntimeError("이미지 파일이 존재하지 않습니다.")
            self.image_files.sort()
            self.image_index = 0

        ########## Object ##########
        self.yolo_engine = YOLOEngine(
            model_path="object/yolov8s.engine",
            input_size=(640, 640),
            sensor_id=0
        )
        ########## Lane ##########
        self.lane_engine = LaneEngine(
            vis=True,
            trt_engine_path="lane/64192.trt",
            input_size=(64, 192),
            image_size=(480, 640),
            num_lanes=2,
            num_grid=50,
            cls_num_per_lane=28
        )
        ########## Depth ##########
        self.depth_engine = DepthEngine(
            lane_engine=self.lane_engine,
            vis=True,
            tensorRT_engine_path="depth/depth_anything_vits14_308.trt",
            input_size=(308, 308),
            cam_intrinsics=np.array([
                [809.5233, 0, 339.2379],
                [0, 808.7865, 265.3243],
                [0, 0, 1]
            ]),
            grid_size=100,
            cell_size=0.02,
            pitch_angle_degree=-18,
            min_height=-np.inf,
            max_height=-0.
        )
        ########## Planning ##########
        self.track_engine = TrackingEngine(
            max_steer=0.4,
            control_method="pid",
            kp=0.1,
            kd=0.01,
            ki=0.01,
            throttle=0.75,
            max_throttle=0.15,
            car_width_cells=2,
            car_length_cells=2
        )

        self.car = NvidiaRacecar(
            throttle_gain=1.0,
            steering_gain=1.0
        )

    def run(self):
        try:
            if self.is_video:
                # 비디오 프레임 읽기
                ret, frame = self.camera.read()
                if not ret:
                    print("비디오 재생 완료")
                    return False
            else:
                # 이미지 파일 읽기
                if self.image_index >= len(self.image_files):
                    print("이미지 처리 완료")
                    return False
                frame = cv2.imread(self.image_files[self.image_index])
                self.image_index += 1

            if frame is None:
                raise RuntimeError("프레임을 읽을 수 없습니다.")

            # YOLO 속도 가져오기
            # yolo_speed = self.yolo_engine.run(frame)

            # LaneEngine을 사용하여 lane_mask 생성
            # lane_mask = self.lane_engine.run(frame)

            # DepthEngine에서 점유 맵 계산
            cv2.imwrite('frame.png', frame)
            occupancy_map = self.depth_engine.run(frame)

            # Planning 엔진을 사용한 속도 및 조향 값 계산
            # speed, steer = self.track_engine.run(occupancy_map)

            # YOLO가 감지한 속도가 있다면 우선 사용
            # speed = yolo_speed if yolo_speed is not None else speed

            # 결과 출력
            # print(f"Speed: {speed}, Steering: {steer}")

        except Exception as e:
            print(f"Error during run: {e}")
            self.reset()

        return True

    def reset(self):
        self.car.throttle = 0
        self.car.steering = 0

if __name__ == "__main__":
    # input_path를 이미지 폴더 또는 비디오 파일 경로로 설정
    input_path = "img"  # 이미지 폴더 또는 비디오 파일 경로 입력
    is_video = False  # True면 비디오 파일, False면 이미지 폴더

    state = State(input_path, is_video=is_video)

    try:
        while state.run():
            pass
    except KeyboardInterrupt:
        print("프로그램이 중단되었습니다.")
    finally:
        state.reset()