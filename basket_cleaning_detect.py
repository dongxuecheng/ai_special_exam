import cv2
import torch
from shapely.geometry import box, Polygon
import threading
import numpy as np
import queue
from datetime import datetime
from ultralytics import YOLO

from utils.tool import IoU
from globals import stop_event,redis_client
from config import SAVE_IMG_PATH,POST_IMG_PATH5,BASKET_CLEANING_VIDEO_SOURCES,BASKET_CLEANING_MODEL_SOURCES
from config import BASKET_SUSPENSION_REGION,BASKET_STEEL_WIRE_REGION,BASKET_PLATFORM_REGION,BASKET_LIFTING_REGION_L,BASKET_LIFTING_REGION_R,BASKET_ELECTRICAL_SYSTEM_REGION,BASKET_SAFETY_LOCK_REGION,BASKET_EMPTY_LOAD_REGION,BASKET_CLEANING_OPERATION_REGION,BASKET_WARNING_ZONE_REGION
from globals import basket_suspension_flag,basket_warning_zone_flag,basket_steel_wire_flag,basket_platform_flag,basket_person_flag
from globals import basket_electrical_system_flag,basket_lifting_flag,basket_safety_lock_flag,basket_safety_belt_flag,basket_empty_load_flag,basket_cleaning_operation_flag,basket_cleaning_up_flag


def init_basket_cleaning_detection():
    global basket_suspension_flag,basket_warning_zone_flag,basket_steel_wire_flag,basket_platform_flag,basket_electrical_system_flag,basket_lifting_flag,basket_safety_lock_flag,basket_safety_belt_flag,basket_cleaning_up_flag,basket_cleaning_operation_flag,basket_empty_load_flag,basket_person_flag,basket_warning_zone_exist_flag
    basket_suspension_flag=False
    basket_warning_zone_flag=False
    basket_steel_wire_flag=False
    basket_platform_flag=False
    basket_electrical_system_flag=False
    basket_lifting_flag=False
    basket_safety_lock_flag=False
    basket_safety_belt_flag=False
    basket_cleaning_up_flag=False
    basket_cleaning_operation_flag=False
    basket_empty_load_flag=False
    basket_person_flag=False
    basket_warning_zone_exist_flag=False
    
    for i in range(1, 13):
        redis_client.delete(f"basket_step_{i}")
        print(f"删除{f'basket_step_{i}'}")
    redis_client.delete("basket_cleaning_order")

    pass

def start_basket_cleaning_detection(start_events):
    threads = []
    for model_path, video_source in zip(BASKET_CLEANING_MODEL_SOURCES,BASKET_CLEANING_VIDEO_SOURCES):
        event = threading.Event()
        start_events.append(event)
        thread = threading.Thread(target=process_video, args=(model_path, video_source,event))
        threads.append(thread)
        thread.daemon=True
        thread.start()
        print("吊篮清洗子线程运行中")

    # Wait for any threads to complete
    for thread in threads:
        thread.join()
        print("吊篮清洗子线程运行结束")


def point_in_region(point, region):#判断点是否在多边形内
    is_inside = cv2.pointPolygonTest(region.reshape((-1, 1, 2)), point, False)
    return is_inside >= 0

def save_image_and_redis(redis_client, results, step_name, save_path, post_path):

    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    imgpath = f"{save_path}/{step_name}_{save_time}.jpg"
    postpath=f"{post_path}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()
    if step_name == "basket_step_1":
        annotated_frame = cv2.polylines(annotated_frame, [BASKET_WARNING_ZONE_REGION.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_2":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in BASKET_SUSPENSION_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_3":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in BASKET_STEEL_WIRE_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_4":
        annotated_frame = cv2.polylines(annotated_frame, [BASKET_PLATFORM_REGION.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
        #cv2.fillPoly(annotated_frame, [BASKET_PLATFORM_REGION], (0, 255, 0))#填充区域
    elif step_name == "basket_step_5":
        annotated_frame = cv2.polylines(annotated_frame, [BASKET_LIFTING_REGION_L.reshape(-1, 1, 2),BASKET_LIFTING_REGION_R.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_6":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in BASKET_SAFETY_LOCK_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_7":
        annotated_frame = cv2.polylines(annotated_frame, [BASKET_ELECTRICAL_SYSTEM_REGION.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
    # elif step_name=="basket_step_8":
    #     annotated_frame = cv2.polylines(annotated_frame, [BASKET_EMPTY_LOAD_REGION.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
    # elif step_name=="basket_step_9":
    #     annotated_frame = cv2.polylines(annotated_frame, [BASKET_SAFETY_LOCK_REGION.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name=="basket_step_10":
        annotated_frame = cv2.polylines(annotated_frame, [BASKET_CLEANING_OPERATION_REGION.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)

    
    cv2.imwrite(imgpath, annotated_frame)
    redis_client.set(step_name, postpath)
    redis_client.rpush("basket_cleaning_order", step_name)



def process_video(model_path, video_source,start_event):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)
    
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 设置缓冲区大小为 10 帧    
    while cap.isOpened():
        if stop_event.is_set():#控制停止推理
            break

        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 != 0:
                continue

            results = model.predict(frame,conf=0.4,verbose=False)

            global basket_suspension_flag,basket_warning_zone_flag,basket_steel_wire_flag,basket_platform_flag,basket_electrical_system_flag,basket_lifting_flag
            global basket_safety_lock_flag,basket_safety_belt_flag,basket_cleaning_up_flag,basket_cleaning_operation_flag,basket_empty_load_flag,basket_person_flag,basket_warning_zone_exist_flag
            global BASKET_PLATFORM_REGION,BASKET_LIFTING_REGION_L,BASKET_LIFTING_REGION_R,BASKET_ELECTRICAL_SYSTEM_REGION,BASKET_WARNING_ZONE_REGION
            for r in results:
                if model_path==BASKET_CLEANING_MODEL_SOURCES[0] and not basket_suspension_flag:#D4悬挂机构
                    boxes=r.boxes.xyxy#人体的检测框
                    keypoints = r.keypoints.xy  
                    confidences = r.keypoints.conf  
                    for i in range(len(boxes)):
                        left_wrist, right_wrist = keypoints[i][9:11].tolist()#获取左右手腕和左右肘的坐标
                        points = [left_wrist, right_wrist]
                        is_inside1 = any(point_in_region(point, BASKET_SUSPENSION_REGION[0]) for point in points)
                        is_inside2 = any(point_in_region(point, BASKET_SUSPENSION_REGION[1]) for point in points)
                        is_inside3 = any(point_in_region(point, BASKET_SUSPENSION_REGION[2]) for point in points)
                        is_inside4 = any(point_in_region(point, BASKET_SUSPENSION_REGION[3]) for point in points)
                        if is_inside1 or is_inside2 or is_inside3 or is_inside4:
                            basket_suspension_flag=True#悬挂机构
                            print("悬挂机构")
                            
                # elif model_path==BASKET_CLEANING_MODEL_SOURCES[1]:#D5吊篮悬挂

                #     boxes = r.boxes.xyxy  
                #     confidences = r.boxes.conf 
                #     classes = r.boxes.cls  

                #     basket_warning_zone_flag=False#当检测不到则为False
                #     basket_cleaning_up_flag=False
                #     for i in range(len(boxes)):
                #         x1, y1, x2, y2 = boxes[i].tolist()
                #         confidence = confidences[i].item()
                #         cls = int(classes[i].item())
                #         label = model.names[cls]
                #         if label=='warning_zone':
                #             basket_warning_zone_flag=True
                #         elif label=='brush':
                #             basket_cleaning_up_flag=True
                        # elif label=='person':

                elif model_path==BASKET_CLEANING_MODEL_SOURCES[1]:#D5,detect
                    boxes = r.boxes.xyxy  
                    confidences = r.boxes.conf 
                    classes = r.boxes.cls  
                    basket_warning_zone_exist_flag=False
                    #basket_warning_zone_flag=False
                    #basket_safety_lock_flag=False#当检测不到则为False
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]

                        # if label=='brush':

                        #     #is_inside = any(point_in_region([(x1+x2)/2,(y1+y2)/2],BASKET_CLEANING_OPERATION_REGION))
                        #     is_inside = point_in_region([(x1+x2)/2,(y1+y2)/2],BASKET_CLEANING_OPERATION_REGION)
                        #     if is_inside:
                        #         basket_cleaning_operation_flag=True
                        #         print("清洗作业")
                            #print("刷子")

                        if label=='warning_zone':
                            basket_warning_zone_exist_flag=True
                            # centerx=(x1+x2)/2
                            # centery=(y1+y2)/2
                            # point_in_region_flag=point_in_region([centerx,centery],BASKET_WARNING_ZONE_REGION)#警戒区划分区域
                            # if point_in_region_flag:
                            #     basket_warning_zone_flag=True
                        # if label=='safety_belt' and basket_warning_zone_flag:#
                        #     basket_safety_belt_flag=True

                elif model_path==BASKET_CLEANING_MODEL_SOURCES[2]:#D6,pose
                    boxes=r.boxes.xyxy#人体的检测框
                    keypoints = r.keypoints.xy  
                    confidences = r.keypoints.conf  

                    basket_person_flag=False#空载的人，当检测不到则为False
                    for i in range(len(boxes)):
                        #当有检测框，则说明有人

                        head_points=keypoints[i][0:5].tolist()#获得头部的五个关键点
                        if any(point_in_region(point, BASKET_WARNING_ZONE_REGION) for point in head_points):
                            basket_person_flag=True#表面人在吊篮内

                        left_wrist, right_wrist = keypoints[i][9:11].tolist()#获取左右手腕和左右肘的坐标
                        points = [left_wrist, right_wrist]
                        if not basket_steel_wire_flag:
                            is_inside1 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[0]) for point in points)
                            is_inside2 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[1]) for point in points)
                            #is_inside3 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[2]) for point in points)
                            #is_inside4 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[3]) for point in points)
                            if is_inside1 or is_inside2:
                                basket_steel_wire_flag=True#钢丝绳
                                print("钢丝绳")
                        if not basket_platform_flag:
                            is_inside = any(point_in_region(point, BASKET_PLATFORM_REGION) for point in points)
                            if is_inside:
                                basket_platform_flag=True
                                print("平台")
                        if not basket_lifting_flag:
                            is_inside1 = any(point_in_region(point, BASKET_LIFTING_REGION_L) for point in points)
                            is_inside2 = any(point_in_region(point, BASKET_LIFTING_REGION_R) for point in points)
                            #print(is_inside1,is_inside2)   
                            if is_inside1 or is_inside2:
                                basket_lifting_flag=True
                                print("提升机")
                        if not basket_electrical_system_flag:
                            is_inside = any(point_in_region(point, BASKET_ELECTRICAL_SYSTEM_REGION) for point in points)
                            if is_inside:
                                basket_electrical_system_flag=True
                                print("电气系统")
                        if not basket_safety_lock_flag:
                            is_inside1 = any(point_in_region(point, BASKET_SAFETY_LOCK_REGION[0]) for point in points)
                            is_inside2 = any(point_in_region(point, BASKET_SAFETY_LOCK_REGION[1]) for point in points)
                            
                            if is_inside1 or is_inside2:
                                basket_safety_lock_flag=True
                                print("安全锁")

                    #print(point_in_region([709,1017],BASKET_PLATFORM_REGION))
                    if not basket_person_flag and point_in_region([709,1017],BASKET_PLATFORM_REGION):
                        basket_empty_load_flag=True 
                    
                elif model_path==BASKET_CLEANING_MODEL_SOURCES[3]:#d6目标检测
                    boxes = r.boxes.xyxy  
                    confidences = r.boxes.conf 
                    classes = r.boxes.cls  
                    basket_warning_zone_flag=False
                    basket_warning_zone_exist_flag=False
                    #basket_safety_lock_flag=False#当检测不到则为False
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]

                        if label=='brush':

                            #is_inside = any(point_in_region([(x1+x2)/2,(y1+y2)/2],BASKET_CLEANING_OPERATION_REGION))
                            is_inside = point_in_region([(x1+x2)/2,(y1+y2)/2],BASKET_CLEANING_OPERATION_REGION)
                            if is_inside:
                                basket_cleaning_operation_flag=True
                                print("清洗作业")
                            #print("刷子")

                        elif label=='warning_zone':
                            basket_warning_zone_exist_flag=True
                            centerx=(x1+x2)/2
                            centery=(y1+y2)/2
                            point_in_region_flag=point_in_region([centerx,centery],BASKET_WARNING_ZONE_REGION)#警戒区划分区域
                            if point_in_region_flag:
                                basket_warning_zone_flag=True
                        if label=='safety_belt' and basket_warning_zone_flag:#
                            basket_safety_belt_flag=True

                elif model_path==BASKET_CLEANING_MODEL_SOURCES[4]:#d6分割
                    boxes = r.boxes.xyxy
                    masks = r.masks.xy
                    classes = r.boxes.cls 
                    hoist=[]
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        cls = int(classes[i].item())
                        label = model.names[cls]
                        if label=="basket":
                            BASKET_PLATFORM_REGION = np.array(masks[i].tolist(), np.int32)
                        elif label=="hoist_l" and BASKET_LIFTING_REGION_L.size == 0:
                            BASKET_LIFTING_REGION_L = np.array(masks[i].tolist(),np.int32)
                        elif label=="hoist_r" and BASKET_LIFTING_REGION_R.size == 0:
                            BASKET_LIFTING_REGION_R = np.array(masks[i].tolist(),np.int32)
                        elif label=="electricalSystem":
                            BASKET_ELECTRICAL_SYSTEM_REGION = np.array(masks[i].tolist(), np.int32)
                    
                    # 检查是否所有hoist masks具有相同形状
                    # if all(h.shape == hoist[0].shape for h in hoist):
                    #     BASKET_LIFTING_REGION = np.vstack(hoist)  # 堆叠成二维数组
                    # else:
                    #     # 如果形状不同，可以手动处理，比如补齐/裁剪
                    #     # 这里假设你想补齐到最大的形状
                    #     max_shape = max(h.shape for h in hoist)
                    #     padded_hoist = [np.pad(h, ((0, max_shape[0] - h.shape[0]), (0, max_shape[1] - h.shape[1])), mode='constant') for h in hoist]
                    #     BASKET_LIFTING_REGION = np.array(padded_hoist)
                    #BASKET_LIFTING_REGION = np.array(hoist,np.int32)
                    #print("hoist",hoist.shape)

                            


            if model_path==BASKET_CLEANING_MODEL_SOURCES[0] and not redis_client.exists("basket_step_2") and basket_suspension_flag:#D4悬挂机构 
                save_image_and_redis(redis_client, results, "basket_step_2", SAVE_IMG_PATH, POST_IMG_PATH5)

            elif model_path==BASKET_CLEANING_MODEL_SOURCES[3]:
                if basket_warning_zone_flag and not redis_client.exists("basket_step_1"):#警戒区
                    save_image_and_redis(redis_client, results, "basket_step_1", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("警戒区")

                elif redis_client.exists("basket_step_12") and not redis_client.exists("basket_step_11"):
                    save_image_and_redis(redis_client, results, "basket_step_11", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("清理现场")
                elif not basket_warning_zone_exist_flag and redis_client.exists("basket_step_1") and not redis_client.exists("basket_step_12"):
                    save_image_and_redis(redis_client, results, "basket_step_12", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("警戒区消失")
                elif basket_safety_belt_flag and not redis_client.exists("basket_step_9"):
                    save_image_and_redis(redis_client, results, "basket_step_9", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("检查安全带挂设")
                elif basket_cleaning_operation_flag and not redis_client.exists("basket_step_10"):
                    save_image_and_redis(redis_client, results, "basket_step_10", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("清洗作业")
                
            elif model_path==BASKET_CLEANING_MODEL_SOURCES[2]:#D6,pose

                if basket_steel_wire_flag and not redis_client.exists("basket_step_3"):
                    save_image_and_redis(redis_client, results, "basket_step_3", SAVE_IMG_PATH, POST_IMG_PATH5)

                elif basket_platform_flag and not redis_client.exists("basket_step_4"):
                    save_image_and_redis(redis_client, results, "basket_step_4", SAVE_IMG_PATH, POST_IMG_PATH5)

                elif basket_lifting_flag and not redis_client.exists("basket_step_5"):
                    save_image_and_redis(redis_client, results, "basket_step_5", SAVE_IMG_PATH, POST_IMG_PATH5)

                elif basket_electrical_system_flag and not redis_client.exists("basket_step_7"):
                    save_image_and_redis(redis_client, results, "basket_step_7", SAVE_IMG_PATH, POST_IMG_PATH5)

                elif basket_safety_lock_flag and not redis_client.exists("basket_step_6"):
                    save_image_and_redis(redis_client, results, "basket_step_6", SAVE_IMG_PATH, POST_IMG_PATH5)

                elif basket_empty_load_flag and not redis_client.exists("basket_step_8"):
                    save_image_and_redis(redis_client, results, "basket_step_8", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("空载")
            #else:#d6目标检测
        
            start_event.set()          
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



        

    