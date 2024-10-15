import cv2
import torch
from shapely.geometry import box, Polygon
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from utils.tool import IoU
from config import SAVE_IMG_PATH,POST_IMG_PATH6,EQUIPMENT_CLEANING_VIDEO_SOURCES,EQUIPMENT_CLEANING_MODEL_SOURCES
from config import EQUIPMENT_SAFETY_ROPE_REGION,EQUIPMENT_WORK_ROPE_REGION,EQUIPMENT_ANCHOR_DEVICE_REGION,EQUIPMENT_CLEANING_OPERATION_REGION,EQUIPMENT_WARNING_ZONE_REGION,EQUIPMENT_SELF_LOCKING_DEVICE_REGION



def point_in_region(point, region):#判断点是否在多边形内
    is_inside = cv2.pointPolygonTest(region.reshape((-1, 1, 2)), point, False)
    return is_inside >= 0


def save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order,results, step_name, save_path, post_path):

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

    cv2.imwrite(imgpath, annotated_frame)
    equipment_cleaning_imgs[step_name]=postpath

    equipment_cleaning_order.append(step_name)

def video_decoder(rtsp_url, frame_queue_list,start_event, stop_event):
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        if stop_event.is_set():#控制停止推理
            print("视频流关闭")
            break
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 != 0:
            continue
        if rtsp_url==EQUIPMENT_CLEANING_VIDEO_SOURCES[0]:
            frame_queue_list[0].put(frame)
            frame_queue_list[1].put(frame)
        elif rtsp_url==EQUIPMENT_CLEANING_VIDEO_SOURCES[1]:
            frame_queue_list[2].put(frame)
            frame_queue_list[3].put(frame)
            frame_queue_list[4].put(frame)


        start_event.set()  
    cap.release()


def process_video(model_path, video_source,start_event,stop_event,equipment_cleaning_flag,equipment_cleaning_imgs,equipment_cleaning_order,person_position,equipment_warning_zone_flag):
    model = YOLO(model_path)
    while True:
        if stop_event.is_set():#控制停止推理
            print("停止推理")
            break
        
        if video_source.empty():
        # 队列为空，跳过处理
            continue
        
        frame = video_source.get()
        results = model.track(frame,conf=0.3,verbose=False,persist=True,tracker="bytetrack.yaml")
        for r in results:
            
            if model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[0]:#D3 detect
                boxes = r.boxes.xyxy  
                confidences = r.boxes.conf 
                classes = r.boxes.cls  

                equipment_warning_zone_flag[0]=False

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    confidence = confidences[i].item()
                    cls = int(classes[i].item())
                    label = model.names[cls]
                    if label=='warning_zone' and confidence>0.5:
                        
                        centerx=(x1+x2)/2
                        centery=(y1+y2)/2
                        equipment_warning_zone_flag[0]=True
                        point_in_region_flag=point_in_region([centerx,centery],EQUIPMENT_WARNING_ZONE_REGION)#警戒区划分区域
                        if point_in_region_flag:
                            equipment_cleaning_flag[0]=True
                            equipment_warning_zone_flag[0]=True


                    elif label=='seating_plate':
                        if(IoU(person_position,[x1,y1,x2,y2])>0.3):            
                            equipment_cleaning_flag[2]=True
                        if (x1 >= 1248 and y1 >= 223 and x2 <= 1625 and y2 <= 1026):
                            equipment_cleaning_flag[7]=True

                    elif label=='u_lock':
                        equipment_cleaning_flag[6]=True
                        #print("u型锁")


                if not equipment_warning_zone_flag[0] and not equipment_warning_zone_flag[1] and equipment_cleaning_flag[0]:#当检测不到警戒区时,判定未拆除警戒区域
                    equipment_cleaning_flag[11]=True
                    print("拆除警戒区域-----------")


            elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[1]:#D3 detect
                boxes = r.boxes.xyxy  
                confidences = r.boxes.conf 
                classes = r.boxes.cls  
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    confidence = confidences[i].item()
                    cls = int(classes[i].item())
                    label = model.names[cls]

                    if label=='person':
                        #person_position=[x1,y1,x2,y2]
                        person_position[0]=x1
                        person_position[1]=y1
                        person_position[2]=x2
                        person_position[3]=y2
                        #print("检测到人"+str(person_position))
                        
                        centerx=(x1+x2)/2
                        centery=(y1+y2)/2
                        point_in_region_flag=point_in_region([centerx,centery],EQUIPMENT_ANCHOR_DEVICE_REGION)#挂点装置是否检测到人
                        if point_in_region_flag and not equipment_cleaning_flag[1]:
                            equipment_cleaning_flag[1]=True

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
                            #print("工作绳")
                    if not equipment_cleaning_flag[4]:
                        is_inside = any(point_in_region(point, EQUIPMENT_SAFETY_ROPE_REGION) for point in points)
                        if is_inside:
                            equipment_cleaning_flag[4]=True#安全绳
                            equipment_cleaning_flag[5]=True
                            #print("安全绳")
                    if not equipment_cleaning_flag[5]:#TODO 暂时不用
                        is_inside = any(point_in_region(point, EQUIPMENT_SELF_LOCKING_DEVICE_REGION) for point in points)
                        if is_inside:
                            equipment_cleaning_flag[5]=True

                
            elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[3]:#d8 目标检测  
                boxes = r.boxes.xyxy  
                confidences = r.boxes.conf 
                classes = r.boxes.cls  
                
                brush_flag=False
                equipment_warning_zone_flag[1]=False

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    confidence = confidences[i].item()
                    cls = int(classes[i].item())
                    label = model.names[cls]
                    # if label=='safety_belt':
                    #     equipment_cleaning_flag[8]=True
                    if label=='brush':
                        brush_flag=True
                        #print("检测到刷子")
                        is_inside = point_in_region([(x1+x2)/2,(y1+y2)/2],EQUIPMENT_CLEANING_OPERATION_REGION)#刷子是否在指定区域
                        if is_inside:
                            equipment_cleaning_flag[9]=True

                    if label=='warning_zone' and confidence>0.7:
                        equipment_warning_zone_flag[1]=True


                if equipment_cleaning_flag[11] and not brush_flag:
                    equipment_cleaning_flag[10]=True
            
            elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[4]:
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
                    equipment_cleaning_flag[8]=True
                    print("安全带挂设完毕")
        
        if model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[0]:
            if equipment_cleaning_flag[0] and 'equipment_step_1' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order, results, "equipment_step_1", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("安全警戒")

            if equipment_cleaning_flag[2] and 'equipment_step_3' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order, results, "equipment_step_3", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("座板图片保存")
            if equipment_cleaning_flag[5] and 'equipment_step_6' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order,results,  "equipment_step_6", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("自锁器")
            if equipment_cleaning_flag[6] and 'equipment_step_7' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order, results, "equipment_step_7", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("u型锁")
            if equipment_cleaning_flag[7] and 'equipment_step_8' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order,results,  "equipment_step_8", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("座板和人有交集图片保存")

            if equipment_cleaning_flag[11] and 'equipment_step_12' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order, results, "equipment_step_12", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("拆除警戒区域")
            if equipment_cleaning_flag[10] and equipment_cleaning_flag[11] and equipment_cleaning_flag[9] and 'equipment_step_11' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order,results, "equipment_step_11", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("清洗现场")
            
        elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[1]:
            if equipment_cleaning_flag[1] and 'equipment_step_2' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order,results, "equipment_step_2", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("挂点装置")

        elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[2]:
            if equipment_cleaning_flag[3] and 'equipment_step_4' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order,results,  "equipment_step_4", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("工作绳")
            if equipment_cleaning_flag[4] and 'equipment_step_5' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order, results, "equipment_step_5", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("安全绳")
            
        elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[3]:#d6目标检测
            if equipment_cleaning_flag[9] and 'equipment_step_10' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order,results,  "equipment_step_10", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("清洗操作区域")

        elif model_path==EQUIPMENT_CLEANING_MODEL_SOURCES[4]:#d8目标检测
            if equipment_cleaning_flag[8] and 'equipment_step_9' not in equipment_cleaning_imgs:
                save_image_and_redis(equipment_cleaning_imgs, equipment_cleaning_order,results,  "equipment_step_9", SAVE_IMG_PATH, POST_IMG_PATH6)
                print("安全带挂设")
        start_event.set() 
           

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



        

    