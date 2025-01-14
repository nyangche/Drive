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
        ########## Depth ##########
        self.depth_engine = DepthEngine(
            vis=True,                       
            tensorRT_engine_path="/home/ircv/projects/deep_daiv/DeepDrive/depth/metric3D_vit_small.engine",  
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
        ########## Lane ##########
        self.lane_engine = LaneEngine(
            trt_engine_path = "/home/ircv/projects/deep_daiv/new/lane/64192.trt",
            input_size=(64, 192),
            image_size=(480, 640),
            num_lanes=2,
            num_grid=50,
            cls_num_per_lane = 28
        )
        self.camera = cv2.VideoCapture(
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
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

            yolo_speed = self.yolo_engine.run(frame)
            occupancy_map = self.depth_engine.run(frame)
            speed, steer = self.track_engine.run(occupancy_map)

            speed = yolo_speed if yolo_speed is not None else speed
            
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
        
