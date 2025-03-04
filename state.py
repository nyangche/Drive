import cv2
import numpy as np

from object.engine import YOLOEngine
from lane.engine import LaneEngine
from depth.engine import DepthEngine
from planning.engine import TrackingEngine
from jetracer.nvidia_racecar import NvidiaRacecar


class State:
    def __init__(self):
        ########## Object ##########
        self.yolo_engine = YOLOEngine(
            model_path="/home/ircv/projects/deep_daiv/DeepDrive/object/yolov8s.engine", 
            input_size=(640, 640),     
            sensor_id=0
        )
        ########## Lane ##########
        self.lane_engine = LaneEngine(
            vis=True,
            trt_engine_path = "/home/ircv/projects/deep_daiv/DeepDrive/lane/64192.trt",
            input_size=(64, 192),
            image_size=(480, 640),
            num_lanes=2,
            num_grid=50,
            cls_num_per_lane = 28
        )
        ########## Depth ##########
        self.depth_engine = DepthEngine(
            lane_engine=self.lane_engine,
            vis=True,                       
            tensorRT_engine_path="/home/ircv/projects/deep_daiv/DeepDrive/depth/depth_anything_vits14_308.trt",  
            input_size=(308, 504),           
            cam_intrinsics=np.array([
                [809.5233, 0, 339.2379],   
                [0, 808.7865, 265.3243],
                [0, 0, 1]
            ]),
            grid_size=100,     
            cell_size=0.02, 
            pitch_angle_degree=-18,
            min_height=0.05,
            max_height=0.1                     
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

        self.camera = cv2.VideoCapture(
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
            "videoconvert ! appsink",
            cv2.CAP_GSTREAMER
        )
        if not self.camera.isOpened():
            raise RuntimeError("CSI camera가 인식되지 않습니다.")
        else:
            print("Number of cameras initialized: 1") 
        self.car = NvidiaRacecar(
            throttle_gain=1.0,           
            steering_gain=1.0             
        )


    def run(self):
        try:
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to read from camera")

            # YOLO 속도 가져오기
            yolo_speed = self.yolo_engine.run(frame)
            
            # LaneEngine을 사용하여 lane_mask 생성
            lane_mask = self.lane_engine.run(frame)


            # DepthEngine에서 점유 맵 계산
            occupancy_map = self.depth_engine.run(frame, lane_mask)


            # Planning 엔진을 사용한 속도 및 조향 값 계산
            speed, steer = self.track_engine.run(occupancy_map)

            # YOLO가 감지한 속도가 있다면 우선 사용
            speed = yolo_speed if yolo_speed is not None else speed

            # 차에 속도 및 조향값 설정
            self.car.throttle = speed
            self.car.steering = steer

        except Exception as e:
            print(f"Error during run: {e}")
            self.reset()
    
    def reset(self):
        self.car.throttle = 0
        self.car.steering = 0

if __name__ == "__main__":
    state = State()
    try:
        while True:
            state.run()
    except:
        state.reset()
        
