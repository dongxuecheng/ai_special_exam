import cv2
import torch
from shapely.geometry import box, Polygon
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from utils.tool import IoU
from globals import stop_event,redis_client
from config import SAVE_IMG_PATH,POST_IMG_PATH6,EQUIPMENT_CLEANING_VIDEO_SOURCES,EQUIPMENT_CLEANING_MODEL_SOURCES
from config import EQUIPMENT_SAFETY_ROPE_REGION,EQUIPMENT_WORK_ROPE_REGION,EQUIPMENT_ANCHOR_DEVICE_REGION,EQUIPMENT_CLEANING_OPERATION_REGION,EQUIPMENT_WARNING_ZONE_REGION
from globals import equipment_cleaning_flag,person_position

def init_equipment_cleaning_detection():
    global equipment_cleaning_flag
    equipment_cleaning_flag=[False]*12
    
    for i in range(1, 12):
        redis_client.delete(f"equipment_step_{i}")
    redis_client.delete("equipment_cleaning_order")

    pass

def start_equipment_cleaning_detection(start_events):
    threads = []
    for model_path, video_source in zip(EQUIPMENT_CLEANING_MODEL_SOURCES,EQUIPMENT_CLEANING_VIDEO_SOURCES):
        event = threading.Event()
        start_events.append(event)
        thread = threading.Thread(target=process_video, args=(model_path, video_source,event))
        threads.append(thread)
        thread.daemon=True
        thread.start()
        print("单人吊具清洗子线程运行中")

    # Wait for any threads to complete
    for thread in threads:
        thread.join()
        print("单人吊具清洗子线程运行结束")


def point_in_region(point, region):#判断点是否在多边形内
    is_inside = cv2.pointPolygonTest(region.reshape((-1, 1, 2)), point, False)
    return is_inside >= 0

def save_image_and_redis(redis_client, results, step_name, save_path, post_path):

    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    imgpath = f"{save_path}/{step_name}_{save_time}.jpg"
    postpath = f"{post_path}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()
    if step_name == "equipment_step_1":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in EQUIPMENT_WARNING_ZONE_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "equipment_step_2":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in EQUIPMENT_ANCHOR_DEVICE_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "equipment_step_4":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in EQUIPMENT_WORK_ROPE_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "equipment_step_5":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in EQUIPMENT_SAFETY_ROPE_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "equipment_step_10":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in EQUIPMENT_CLEANING_OPERATION_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    # elif step_name == "equipment_step_4":
    #     annotated_frame = cv2.polylines(annotated_frame, BASKET_PLATFORM_REGION.reshape(-1, 1, 2), isClosed=True, color=(0, 255, 0), thickness=4)
    # elif step_name == "equipment_step_6":
    #     annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in BASKET_SAFETY_LOCK_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    
    cv2.imwrite(imgpath, annotated_frame)
    redis_client.set(step_name, postpath)
    redis_client.rpush("equipment_cleaning_order", step_name)




def process_video(model_path, video_source,start_event):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)
    
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)  # 设置缓冲区大小为 10 帧    
    while cap.isOpened():
        if stop_event.is_set():#控制停止推理
            break

        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 != 0:
                continue

            results = model.predict(frame,conf=0.3,verbose=False)
            global equipment_cleaning_flag,person_position

            #person_position=[0,0,0,0]
            for r in results:
                
                if model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[0] or model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[1]:#D3 detect
                    boxes = r.boxes.xyxy  
                    confidences = r.boxes.conf 
                    classes = r.boxes.cls  
                
                    # basket_warning_zone_flag=False#当检测不到则为False
                    # basket_cleaning_up_flag=False
                    equipment_cleaning_flag[0]=False
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]
                        if label=='warning_zone' and confidence>0.7:
                            centerx=(x1+x2)/2
                            centery=(y1+y2)/2
                            point_in_region_flag=point_in_region([centerx,centery],EQUIPMENT_WARNING_ZONE_REGION)#警戒区划分区域
                            if point_in_region_flag and not equipment_cleaning_flag[0]:
                                equipment_cleaning_flag[0]=True
                                print("警戒区")

                        elif label=='person':
                            person_position=[x1,y1,x2,y2]
                            print("检测到人")
                            
                            centerx=(x1+x2)/2
                            centery=(y1+y2)/2
                            point_in_region_flag=point_in_region([centerx,centery],EQUIPMENT_ANCHOR_DEVICE_REGION)#挂点装置是否检测到人
                            if point_in_region_flag and not equipment_cleaning_flag[1]:
                                equipment_cleaning_flag[1]=True
                                print("挂点装置区域有人")

                        elif label=='seating_plate':
                            if(IoU(person_position,[x1,y1,x2,y2])>0.3):            
                                equipment_cleaning_flag[2]=True
                                equipment_cleaning_flag[7]=True
                                print("座板和人有交集")
                            else:
                                print("检测到座板")

                        elif label=='u_lock':
                            equipment_cleaning_flag[6]=True
                            print("u型锁")

                        elif label=='self_locking':
                            equipment_cleaning_flag[5]=True
                            print("自锁器")

                        elif label=='safety_belt':
                            equipment_cleaning_flag[8]=True
                            print("检查安全带挂设")
                        # elif label=='brush' and confidence>0.6:
                        #     #equipment_cleaning_flag[10]=True
                        #     brush_flag=True

                    if not equipment_cleaning_flag[0] and equipment_cleaning_flag[7]:#当检测不到警戒区时,判定未拆除警戒区域
                        equipment_cleaning_flag[11]=True




                elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[2]:#D6,pose
                    boxes=r.boxes.xyxy#人体的检测框
                    keypoints = r.keypoints.xy  
                    confidences = r.keypoints.conf  

                    basket_person_flag=False#当检测不到则为False
                    for i in range(len(boxes)):
                        #当有检测框，则说明有人
                        #basket_person_flag=True
                        left_wrist, right_wrist = keypoints[i][9:11].tolist()#获取左右手腕和左右肘的坐标
                        points = [left_wrist, right_wrist]
                        if not equipment_cleaning_flag[3]:
                            is_inside1 = any(point_in_region(point, EQUIPMENT_WORK_ROPE_REGION) for point in points)
                            if is_inside1:
                                equipment_cleaning_flag[3]=True#工作绳
                                print("工作绳")
                        if not equipment_cleaning_flag[4]:
                            is_inside = any(point_in_region(point, EQUIPMENT_SAFETY_ROPE_REGION) for point in points)
                            if is_inside:
                                equipment_cleaning_flag[4]=True#安全绳
                                print("安全绳")

                    
                elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[3]:#d8 目标检测
                    boxes = r.boxes.xyxy  
                    confidences = r.boxes.conf 
                    classes = r.boxes.cls  
                    
                    brush_flag=False
                    #basket_safety_lock_flag=False#当检测不到则为False
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]
                        # if label=='safety_belt':
                        #     equipment_cleaning_flag[8]=True
                        if label=='brush':
                            brush_flag=True
                            is_inside = any(point_in_region(point,EQUIPMENT_CLEANING_OPERATION_REGION) for point in points)#刷子是否在指定区域
                            if is_inside:
                                equipment_cleaning_flag[9]=True

                    if equipment_cleaning_flag[11] and not brush_flag:
                        equipment_cleaning_flag[10]=True
            
            if model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[0] or model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[1]:
                if equipment_cleaning_flag[0] and not redis_client.exists("equipment_step_1"):
                    save_image_and_redis(redis_client, results, "equipment_step_1", SAVE_IMG_PATH, POST_IMG_PATH6)
                if equipment_cleaning_flag[1] and not redis_client.exists("equipment_step_2"):
                    save_image_and_redis(redis_client, results, "equipment_step_2", SAVE_IMG_PATH, POST_IMG_PATH6)
                if equipment_cleaning_flag[2] and not redis_client.exists("equipment_step_3"):
                    save_image_and_redis(redis_client, results, "equipment_step_3", SAVE_IMG_PATH, POST_IMG_PATH6)
                if equipment_cleaning_flag[5] and not redis_client.exists("equipment_step_6"):
                    save_image_and_redis(redis_client, results, "equipment_step_6", SAVE_IMG_PATH, POST_IMG_PATH6)
                if equipment_cleaning_flag[6] and not redis_client.exists("equipment_step_7"):
                    save_image_and_redis(redis_client, results, "equipment_step_7", SAVE_IMG_PATH, POST_IMG_PATH6)
                if equipment_cleaning_flag[7] and not redis_client.exists("equipment_step_8"):
                    save_image_and_redis(redis_client, results, "equipment_step_8", SAVE_IMG_PATH, POST_IMG_PATH6)
                if equipment_cleaning_flag[8] and not redis_client.exists("equipment_step_9"):
                    save_image_and_redis(redis_client, results, "equipment_step_9", SAVE_IMG_PATH, POST_IMG_PATH6)
                if equipment_cleaning_flag[11] and not redis_client.exists("equipment_step_12"):
                    save_image_and_redis(redis_client, results, "equipment_step_12", SAVE_IMG_PATH, POST_IMG_PATH6)
                

            elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[2]:
                if equipment_cleaning_flag[3] and not redis_client.exists("equipment_step_4"):
                    save_image_and_redis(redis_client, results, "equipment_step_4", SAVE_IMG_PATH, POST_IMG_PATH6)
                if equipment_cleaning_flag[4] and not redis_client.exists("equipment_step_5"):
                    save_image_and_redis(redis_client, results, "equipment_step_5", SAVE_IMG_PATH, POST_IMG_PATH6)
                
            elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[3]:#d6目标检测
                if equipment_cleaning_flag[9] and not redis_client.exists("equipment_step_10"):
                    save_image_and_redis(redis_client, results, "equipment_step_10", SAVE_IMG_PATH, POST_IMG_PATH6)
                if equipment_cleaning_flag[10] and not redis_client.exists("equipment_step_11"):
                    save_image_and_redis(redis_client, results, "equipment_step_11", SAVE_IMG_PATH, POST_IMG_PATH6)
            start_event.set()          
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



        

    