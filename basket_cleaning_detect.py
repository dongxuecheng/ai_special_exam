import cv2
import torch
from shapely.geometry import box, Polygon

import numpy as np
import queue
from datetime import datetime
from ultralytics import YOLO

from utils.tool import IoU
from config import SAVE_IMG_PATH,POST_IMG_PATH5,BASKET_CLEANING_MODEL_SOURCES
from config import BASKET_SUSPENSION_REGION,BASKET_STEEL_WIRE_REGION,BASKET_PLATFORM_REGION,BASKET_LIFTING_REGION_L,BASKET_LIFTING_REGION_R,BASKET_ELECTRICAL_SYSTEM_REGION,BASKET_SAFETY_LOCK_REGION,BASKET_EMPTY_LOAD_REGION,BASKET_CLEANING_OPERATION_REGION,BASKET_WARNING_ZONE_REGION



def point_in_region(point, region):#判断点是否在多边形内
    is_inside = cv2.pointPolygonTest(region.reshape((-1, 1, 2)), point, False)
    return is_inside >= 0

def save_image_and_redis(basket_cleaning_imgs,basket_cleaning_order, results, step_name, save_path, post_path):

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
    
    elif step_name=="basket_step_10":
        annotated_frame = cv2.polylines(annotated_frame, [BASKET_CLEANING_OPERATION_REGION.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)

    
    cv2.imwrite(imgpath, annotated_frame)
    basket_cleaning_imgs[step_name] = postpath
    basket_cleaning_order.append(step_name)




def process_video(model_path, video_source, start_event, stop_event, basket_cleaning_flag, basket_cleaning_order, basket_cleaning_imgs, basket_cleaning_warning_zone_flag,basket_seg_region):
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
            
            if model_path== BASKET_CLEANING_MODEL_SOURCES[2]:#D4悬挂机构
                results = model.predict(frame,conf=0.6,verbose=False)
            else:
                results = model.predict(frame,conf=0.4,verbose=False)

            for r in results:
                if model_path==BASKET_CLEANING_MODEL_SOURCES[0] and not basket_cleaning_flag[1]:#D4悬挂机构
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
                            #basket_suspension_flag=True#悬挂机构
                            basket_cleaning_flag[1]=True
                            


                elif model_path==BASKET_CLEANING_MODEL_SOURCES[1]:#D5,detect
                    boxes = r.boxes.xyxy  
                    confidences = r.boxes.conf 
                    classes = r.boxes.cls  
                    basket_cleaning_warning_zone_flag[0]=False

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]



                        if label=='warning_zone':
                            basket_cleaning_warning_zone_flag[0]=True   
                            #basket_cleaning_flag[0]=True




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
                        if not basket_cleaning_flag[2]:
                            is_inside1 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[0]) for point in points)
                            is_inside2 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[1]) for point in points)
                            #is_inside3 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[2]) for point in points)
                            #is_inside4 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[3]) for point in points)
                            if is_inside1 or is_inside2:
                                #basket_steel_wire_flag=True#钢丝绳
                                basket_cleaning_flag[2]=True

                        if not basket_cleaning_flag[3]:
                            is_inside = any(point_in_region(point,basket_seg_region['platform']) for point in points)
                            if is_inside:
                                #basket_platform_flag=True
                                basket_cleaning_flag[3]=True

                        if not basket_cleaning_flag[4]:
                            is_inside1 = any(point_in_region(point,basket_seg_region['hoist_l']) for point in points)
                            is_inside2 = any(point_in_region(point,basket_seg_region['hoist_r']) for point in points)
                            #print(is_inside1,is_inside2)   
                            if is_inside1 or is_inside2:
                                #basket_lifting_flag=True
                                basket_cleaning_flag[4]=True

                        if not basket_cleaning_flag[5]:
                            is_inside1 = any(point_in_region(point, BASKET_SAFETY_LOCK_REGION[0]) for point in points)
                            is_inside2 = any(point_in_region(point, BASKET_SAFETY_LOCK_REGION[1]) for point in points)
                            
                            if is_inside1 or is_inside2:
                                #basket_safety_lock_flag=True
                                basket_cleaning_flag[5]=True

                        if not basket_cleaning_flag[6]:
                            is_inside = any(point_in_region(point,basket_seg_region['electricalSystem']) for point in points)
                            if is_inside:
                                #basket_electrical_system_flag=True
                                basket_cleaning_flag[6]=True



                    #print(point_in_region([709,1017],BASKET_PLATFORM_REGION))
                    if not basket_person_flag and'platform' in basket_seg_region  and point_in_region([709,1017],basket_seg_region['platform']):
                        basket_cleaning_flag[7]=True 
                    
                elif model_path==BASKET_CLEANING_MODEL_SOURCES[3]:#d6目标检测
                    boxes = r.boxes.xyxy  
                    confidences = r.boxes.conf 
                    classes = r.boxes.cls  
                    basket_cleaning_warning_zone_flag[1]=False
                    brush_flag=False
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]

                        if label=='brush':
                            brush_flag=True
                            #is_inside = any(point_in_region([(x1+x2)/2,(y1+y2)/2],BASKET_CLEANING_OPERATION_REGION))
                            is_inside = point_in_region([(x1+x2)/2,(y1+y2)/2],BASKET_CLEANING_OPERATION_REGION)
                            if is_inside:
                                basket_cleaning_flag[9]=True
                                
                            #print("刷子")

                        elif label=='warning_zone':
                            basket_cleaning_warning_zone_flag[1]=True
                            centerx=(x1+x2)/2
                            centery=(y1+y2)/2
                            point_in_region_flag=point_in_region([centerx,centery],BASKET_WARNING_ZONE_REGION)#警戒区划分区域
                            if point_in_region_flag:
                                basket_cleaning_flag[0]=True
                    
                    if not basket_cleaning_warning_zone_flag[0] and not basket_cleaning_warning_zone_flag[1] and basket_cleaning_flag[0]:#当检测不到警戒区时,判定未拆除警戒区域
                        basket_cleaning_flag[11]=True
                        print("拆除警戒区域-----------")
                    if basket_cleaning_flag[11] and not brush_flag:
                        basket_cleaning_flag[10]=True

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
                            basket_seg_region["platform"]=np.array(masks[i].tolist(), np.int32)
                        elif label=="hoist_l":
                            basket_seg_region["hoist_l"]=np.array(masks[i].tolist(),np.int32)
                        elif label=="hoist_r":
                            basket_seg_region["hoist_r"]=np.array(masks[i].tolist(),np.int32)
                        elif label=="electricalSystem":
                            basket_seg_region["electricalSystem"]=np.array(masks[i].tolist(), np.int32)
                
                elif model_path==BASKET_CLEANING_MODEL_SOURCES[5]:
                    boxes = r.boxes.xyxy  
                    confidences = r.boxes.conf 
                    classes = r.boxes.cls  

                    safety_belt_position=[0,0,0,0]
                    self_locking_position=[0,0,0,0]


                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]
                        if label=='safety_belt':
                            safety_belt_position=[x1,y1,x2,y2]

                        elif label=='self_lock':
                            self_locking_position=[x1,y1,x2,y2]

                    
                    if IoU(safety_belt_position,self_locking_position)>0:
                        #equipment_cleaning_flag[5]=True
                        basket_cleaning_flag[8]=True
                        
                    

                        

            if model_path==BASKET_CLEANING_MODEL_SOURCES[0] and 'basket_step_2' not in basket_cleaning_imgs and basket_cleaning_flag[1]:#D4悬挂机构 
                save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_2", SAVE_IMG_PATH, POST_IMG_PATH5)
                print("悬挂机构检测完毕")


            elif model_path==BASKET_CLEANING_MODEL_SOURCES[2]:#D6,pose


                if basket_cleaning_flag[2] and 'basket_step_3' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_3", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("钢丝绳")
                elif basket_cleaning_flag[3] and 'basket_step_4' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_4", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("平台")
                elif basket_cleaning_flag[4] and 'basket_step_5' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_5", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("提升机")
                elif basket_cleaning_flag[5] and 'basket_step_6' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_6", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("安全锁")
                elif basket_cleaning_flag[6] and 'basket_step_7' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_7", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("电气系统")
                elif basket_cleaning_flag[7] and 'basket_step_8' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_8", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("空载")

            elif model_path==BASKET_CLEANING_MODEL_SOURCES[3]:

                if basket_cleaning_flag[0] and 'basket_step_1' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_1", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("警戒区")
                elif basket_cleaning_flag[11] and 'basket_step_12' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_12", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("警戒区消失")
                elif basket_cleaning_flag[9] and 'basket_step_10' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_10", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("清洗作业")
                elif basket_cleaning_flag[10] and 'basket_step_11' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_11", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("清理现场")
            
            elif model_path==BASKET_CLEANING_MODEL_SOURCES[5]:
                if basket_cleaning_flag[8] and 'basket_step_9' not in basket_cleaning_imgs:
                    save_image_and_redis(basket_cleaning_imgs, basket_cleaning_order, results, "basket_step_9", SAVE_IMG_PATH, POST_IMG_PATH5)
                    print("安全带挂设")
        
            start_event.set()          
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



        

    