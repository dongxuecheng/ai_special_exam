import cv2
import torch
from shapely.geometry import box, Polygon
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from utils.tool import IoU
from globals import stop_event,redis_client
from config import SAVE_IMG_PATH,POST_IMG_PATH4,BASKET_CLEANING_VIDEO_SOURCES,BASKET_CLEANING_MODEL_SOURCES
from globals import basket_suspension_flag,basket_warning_zone_flag,basket_steel_wire_flag


def init_basket_cleaning_detection():
    # global platform_setup_steps_detect_num,platform_setup_final_result,platform_setup_steps_img
    # platform_setup_final_result=[0]*14
    # platform_setup_steps_detect_num=[0]*14
    # platform_setup_steps_img=[False]*14
    # redis_client.delete("platform_setup_order")
    # for i in range(1, 14):
    #     redis_client.delete(f"platform_setup_{i}")
    #     redis_client.delete(f"platform_setup_{i}_img")
    pass

def start_basket_cleaning_detection(start_events):
    threads = []
    for model_path, video_source in zip(BASKET_CLEANING_VIDEO_SOURCES, BASKET_CLEANING_MODEL_SOURCES):
        event = threading.Event()
        start_events.append(event)
        thread = threading.Thread(target=process_video, args=(model_path, video_source,event))
        threads.append(thread)
        thread.daemon=True
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
        print("吊篮清洗子线程运行结束")


def point_in_region(point, region):#判断点是否在多边形内
    is_inside = cv2.pointPolygonTest(region.reshape((-1, 1, 2)), point, False)
    return is_inside >= 0

def process_video(model_path, video_source,start_event):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        if stop_event.is_set():#控制停止推理
            break

        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 != 0:
                continue

            results = model.predict(frame,conf=0.6,verbose=False)

            
            #悬挂机构区域，分为四个区域
            SUSPENSION_REGION = np.array([[[668, 310], [800, 310], [800, 1070], [668, 1070]],
                               [[1690, 310], [1750, 310], [1750, 710], [1690, 710]],
                               [[1350, 340], [1405, 340], [1405, 720], [1350, 720]],
                               [[550, 385], [635, 385], [635, 880], [550, 880]]], np.int32)
            
            STEEL_WIRE_REGION = np.array([[[0, 0], [0, 0], [0, 0], [0, 0]],[[668, 310], [800, 310], [800, 1070], [668, 1070]],[[0, 0], [0, 0], [0, 0], [0, 0]],[[668, 310], [800, 310], [800, 1070], [668, 1070]]], np.int32)#钢丝绳区域，暂时没有钢丝绳的区域

            global basket_suspension_flag,basket_warning_zone_flag,basket_steel_wire_flag
            for r in results:
                if model_path==BASKET_CLEANING_MODEL_SOURCES[0] and not basket_suspension_flag:#D4悬挂机构
                    boxes=r.boxes.xyxy#人体的检测框
                    keypoints = r.keypoints.xy  
                    confidences = r.keypoints.conf  
                    for i in range(len(boxes)):
                        left_elbow, right_elbow, left_wrist, right_wrist = keypoints[i][7:11].tolist()#获取左右手腕和左右肘的坐标
                        points = [left_elbow, right_elbow, left_wrist, right_wrist]
                        is_inside1 = all(point_in_region(point, SUSPENSION_REGION[0]) for point in points)
                        is_inside2 = all(point_in_region(point, SUSPENSION_REGION[1]) for point in points)
                        is_inside3 = all(point_in_region(point, SUSPENSION_REGION[2]) for point in points)
                        is_inside4 = all(point_in_region(point, SUSPENSION_REGION[3]) for point in points)
                        if is_inside1 or is_inside2 or is_inside3 or is_inside4:
                            basket_suspension_flag=True#悬挂机构
                            
                elif model_path==BASKET_CLEANING_MODEL_SOURCES[1]:#D5吊篮悬挂

                    boxes = r.boxes.xyxy  
                    confidences = r.boxes.conf 
                    classes = r.boxes.cls  

                    basket_warning_zone_flag=False#当检测不到则为False
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]
                        if label=='warning_zone':
                            basket_warning_zone_flag=True

                elif model_path==BASKET_CLEANING_MODEL_SOURCES[2]:#D6,pose
                    boxes=r.boxes.xyxy#人体的检测框
                    keypoints = r.keypoints.xy  
                    confidences = r.keypoints.conf  
                    for i in range(len(boxes)):
                        left_elbow, right_elbow, left_wrist, right_wrist = keypoints[i][7:11].tolist()#获取左右手腕和左右肘的坐标
                        points = [left_elbow, right_elbow, left_wrist, right_wrist]
                        is_inside1 = all(point_in_region(point, STEEL_WIRE_REGION[0]) for point in points)
                        is_inside2 = all(point_in_region(point, STEEL_WIRE_REGION[1]) for point in points)
                        is_inside3 = all(point_in_region(point, STEEL_WIRE_REGION[2]) for point in points)
                        is_inside4 = all(point_in_region(point, STEEL_WIRE_REGION[3]) for point in points)
                        if is_inside1 or is_inside2 or is_inside3 or is_inside4:
                            basket_steel_wire_flag=True#悬挂机构
            


                start_event.set()          
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



        

    