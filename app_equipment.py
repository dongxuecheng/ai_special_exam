#1 start_detection #开启检测服务

#2 stop_detection #停止检测服务

#3 reset_status #获取复位检测状态

#4 start_exam #开始焊接考核

#5 exam_status #获取焊接考核状态

#6 stop_exam #停止焊接考核



import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
import re
from fastapi import FastAPI
import uvicorn
import logging
from fastapi.staticfiles import StaticFiles
import multiprocessing as mp
from multiprocessing import Queue
from shapely.geometry import box, Polygon
from utils import IoU
from config import SAVE_IMG_PATH_EQUIPMENT_K2,URL_IMG_PATH_EQUIPMENT_K2,WEIGHTS_EQUIPMENT,VIDEOS_EQUIPMENT
from config import EQUIPMENT_CLEANING_OPERATION_REGION,EQUIPMENT_WORK_ROPE_REGION,EQUIPMENT_SAFETY_ROPE_REGION,EQUIPMENT_ANCHOR_DEVICE_REGION,EQUIPMENT_WARNING_ZONE_REGION



#焊接考核的穿戴
app = FastAPI()
# 挂载目录作为静态文件路径
app.mount("/images", StaticFiles(directory="images"))
# 获得uvicorn服务器的日志记录器
logging = logging.getLogger("uvicorn")

# 全局变量
processes = []
start_events = []  # 存储每个进程的启动事件
stop_events = []  # 存储每个进程的停止事件

#mp.Array性能较高，适合大量写入的场景
equipment_cleaning_flag = mp.Array('b', [False] * 12)
person_position = mp.Array('f', [0.0] * 4)  # 创建一个长度为4的共享数组，并初始化为0.0,用于在多个线程中间传递浮点型变量#用于存储人的位置信息
#mp.Value适合单个值的场景，性能较慢
manager = mp.Manager()
equipment_cleaning_order = manager.list()#用于存储各个步骤的顺序
equipment_cleaning_imgs = manager.dict()  #用于存储各个步骤的图片
frame_queue_list = [Queue(maxsize=50) for _ in range(5)] 
exam_status_flag = mp.Value('b', False)  # 创建一个共享变量，并初始化为False,用于在多个线程中间传递变量,表示是否开始考核,True表示开始考核
equipment_warning_zone_flag=mp.Array('b', [False] * 2)#存储两个视角下的警戒区域的检测结果

def point_in_region(point, region):#判断点是否在多边形内
    is_inside = cv2.pointPolygonTest(region.reshape((-1, 1, 2)), point, False)
    return is_inside >= 0

def rect_polgyon_iou(rect, polgyon):
    #rect_shapely = box(x1,y1, x2, y2)#使用shapely库创建的矩形
    rect_shapely = box(rect[0],rect[1],rect[2],rect[3])
    polgyon_shapely = Polygon(polgyon.tolist()) #shapely计算矩形检测框和多边形的iou使用
    intersection = rect_shapely.intersection(polgyon_shapely)
    # 计算并集
    union = rect_shapely.union(polgyon_shapely)
    # 计算 IoU
    iou = intersection.area / union.area

    return iou

def save_image(equipment_cleaning_imgs, equipment_cleaning_order,results, step_name):

    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    img_path = f"{SAVE_IMG_PATH_EQUIPMENT_K2}/{step_name}_{save_time}.jpg"
    url_path = f"{URL_IMG_PATH_EQUIPMENT_K2}/{step_name}_{save_time}.jpg"
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

    cv2.imwrite(img_path, annotated_frame)
    equipment_cleaning_imgs[step_name]=url_path
    equipment_cleaning_order.append(step_name)
    logging.info(f"{step_name}已完成")

def fetch_video_stream(rtsp_url, frame_queue_list, start_event, stop_event):  # 拉取视频流到队列中
    #队列与对应的模型
    #frame_queue_list[0]:吊具正面检查座板
    #frame_queue_list[1]:吊具正面检查人
    #frame_queue_list[2]:吊具顶部姿态估计
    #frame_queue_list[3]:吊具顶部，检查警戒区刷子
    #frame_queue_list[4]:吊具顶部，检查安全带挂设
    cap = cv2.VideoCapture(rtsp_url)
    index = VIDEOS_EQUIPMENT.index(rtsp_url)
    while cap.isOpened():
        if stop_event.is_set():  # 控制停止推理
            logging.info("fetch_video_stream is stopped")
            break
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 8 != 0:
            continue
        if not start_event.is_set():
            start_event.set()
            logging.info(f'fetch_video_stream{rtsp_url}')
        if index==0:
            frame_queue_list[0].put_nowait(frame)
            frame_queue_list[1].put_nowait(frame)
        elif index==1:
            frame_queue_list[2].put_nowait(frame)
            frame_queue_list[3].put_nowait(frame)
            frame_queue_list[4].put_nowait(frame)
    cap.release()

def infer_yolo(model_path,video_source, start_event, stop_event,equipment_cleaning_flag,equipment_cleaning_imgs,equipment_cleaning_order,person_position,equipment_warning_zone_flag):#YOLO模型推理
    model = YOLO(model_path)
    while True:      
        if stop_event.is_set():
            logging.info(f"{model_path} infer_yolo is stopped")
            break        
        if video_source.empty():
            continue
        frame = video_source.get()
        #results = model.track(frame,verbose=False,conf=0.5,device='0',tracker="bytetrack.yaml")
        if model_path==WEIGHTS_EQUIPMENT[2]:
            results = model.predict(frame, verbose=False, conf=0.3,classes=[0])
        else:
            results = model.predict(frame, verbose=False, conf=0.3)
        
        if not start_event.is_set():
            start_event.set()
            logging.info(f"{model_path} infer_yolo is running")
        #for r in results:

        if model_path==WEIGHTS_EQUIPMENT[0]:#D3 detect
            boxes = results[0].boxes.xyxy  
            confidences = results[0].boxes.conf 
            classes = results[0].boxes.cls  

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
                        #print("警戒区域")
                        #equipment_warning_zone_flag[0]=True


                elif label=='seating_plate':
                    if(IoU(person_position,[x1,y1,x2,y2])>0.3):            
                        equipment_cleaning_flag[2]=True
                    if (x1 >= 946 and y1 >= 220 and x2 <= 1380 and y2 <= 1013) and equipment_cleaning_flag[2]:
                        equipment_cleaning_flag[7]=True

                elif label=='u_lock':
                    equipment_cleaning_flag[6]=True
                    #print("u型锁")


            if not equipment_warning_zone_flag[0] and not equipment_warning_zone_flag[1] and equipment_cleaning_flag[0] and equipment_cleaning_flag[9]:#当检测不到警戒区时,判定未拆除警戒区域
                equipment_cleaning_flag[11]=True
                print("拆除警戒区域-----------")


        elif model_path==WEIGHTS_EQUIPMENT[1]:#D3 detect
            boxes = results[0].boxes.xyxy  
            confidences = results[0].boxes.conf 
            classes = results[0].boxes.cls  
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

        elif model_path==WEIGHTS_EQUIPMENT[2]:#D6,pose
            boxes=results[0].boxes.xyxy#人体的检测框
            keypoints = results[0].keypoints.xy  
            confidences = results[0].keypoints.conf  

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
                # if not equipment_cleaning_flag[5]:#TODO 暂时不用
                #     is_inside = any(point_in_region(point, EQUIPMENT_SELF_LOCKING_DEVICE_REGION) for point in points)
                #     if is_inside:
                #         equipment_cleaning_flag[5]=True

            
        elif model_path==WEIGHTS_EQUIPMENT[3]:#d8 目标检测  
            boxes = results[0].boxes.xyxy  
            confidences = results[0].boxes.conf 
            classes = results[0].boxes.cls  
            
            brush_flag=False
            equipment_warning_zone_flag[1]=False

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                confidence = confidences[i].item()
                cls = int(classes[i].item())
                label = model.names[cls]
                # centerx=(x1+x2)/2
                # centery=(y1+y2)/2
                # if label=='safety_belt':
                #     equipment_cleaning_flag[8]=True
                if label=='brush':
                    brush_flag=True
                    #print("检测到刷子")
                    is_inside = point_in_region([(x1+x2)/2,(y1+y2)/2],EQUIPMENT_CLEANING_OPERATION_REGION)#刷子是否在指定区域
                    if is_inside:
                        equipment_cleaning_flag[9]=True

                if label=='warning_zone' and confidence>0.7:
                    # point_in_region_flag=point_in_region([centerx,centery],EQUIPMENT_WARNING_ZONE_REGION)#警戒区划分区域
                    # if point_in_region_flag:
                    equipment_warning_zone_flag[1]=True
                    #equipment_cleaning_flag[0]=True


            if equipment_cleaning_flag[11] and not brush_flag:
                equipment_cleaning_flag[10]=True
        
        elif model_path==WEIGHTS_EQUIPMENT[4]:
                boxes = results[0].boxes.xyxy  
                confidences = results[0].boxes.conf 
                classes = results[0].boxes.cls  

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


        
        steps = {
            WEIGHTS_EQUIPMENT[0]: [
            (equipment_cleaning_flag[0], 'equipment_step_1', "安全警戒"),
            (equipment_cleaning_flag[2], 'equipment_step_3', "座板图片保存"),
            (equipment_cleaning_flag[5], 'equipment_step_6', "自锁器"),
            (equipment_cleaning_flag[6], 'equipment_step_7', "u型锁"),
            (equipment_cleaning_flag[7], 'equipment_step_8', "座板和人有交集图片保存"),
            (equipment_cleaning_flag[11], 'equipment_step_12', "拆除警戒区域"),
            (equipment_cleaning_flag[10], 'equipment_step_11', "清洗现场")
            ],
            WEIGHTS_EQUIPMENT[1]: [
            (equipment_cleaning_flag[1], 'equipment_step_2', "挂点装置")
            ],
            WEIGHTS_EQUIPMENT[2]: [
            (equipment_cleaning_flag[3], 'equipment_step_4', "工作绳"),
            (equipment_cleaning_flag[4], 'equipment_step_5', "安全绳")
            ],
            WEIGHTS_EQUIPMENT[3]: [
            (equipment_cleaning_flag[9], 'equipment_step_10', "清洗操作区域")
            ],
            WEIGHTS_EQUIPMENT[4]: [
            (equipment_cleaning_flag[8], 'equipment_step_9', "安全带挂设")
            ]
        }
        if exam_status_flag.value:
            for flag, step, message in steps.get(model_path, []):
                if flag and step not in equipment_cleaning_imgs:
                    save_image(equipment_cleaning_imgs, equipment_cleaning_order, results, step)
                    logging.info(message)
            



def reset_shared_variables():
    global frame_queue_list
    exam_status_flag.value = False
    init_exam_variables()
    for queue in frame_queue_list:
        while not queue.empty():
            queue.get()
            logging.info("frame_queue_list is empty")

def init_exam_variables():
    for i in range(len(equipment_cleaning_flag)):
        equipment_cleaning_flag[i] = False    
    equipment_cleaning_imgs.clear()
    equipment_cleaning_order[:]=[]

    person_position[0] = 0.0
    person_position[1] = 0.0
    person_position[2] = 0.0
    person_position[3] = 0.0

    equipment_warning_zone_flag[0]=False
    equipment_warning_zone_flag[1]=False



@app.get('/start_detection')
def start_detection():
    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        for video_source in VIDEOS_EQUIPMENT:
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            stop_event=mp.Event()
            process = mp.Process(target=fetch_video_stream, args=(video_source,frame_queue_list, start_event, stop_event))
            stop_events.append(stop_event)
            start_events.append(start_event)  # 加入 start_events 列表，因为start_events是列表，append或clear不需要加global
            processes.append(process)
            process.start()

        for model_path, video_source in zip(WEIGHTS_EQUIPMENT, frame_queue_list):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            stop_event=mp.Event()
            process = mp.Process(target=infer_yolo, args=(model_path,video_source, start_event, stop_event,equipment_cleaning_flag,equipment_cleaning_imgs,equipment_cleaning_order,person_position,equipment_warning_zone_flag))
            start_events.append(start_event)  # 加入 start_events 列表
            stop_events.append(stop_event)
            processes.append(process)
            process.start()
        # 等待所有进程的 start_event 被 set
        for event in start_events:
            event.wait()  # 等待每个进程通知它已经成功启动
            
        logging.info("start_detection is success")
        exam_status_flag.value = False#表示没有开始考核
        return {"status": "SUCCESS"}
    else:
        logging.info("welding_reset_detection——ALREADY_RUNNING")
        return {"status": "ALREADY_RUNNING"}
    
@app.get('/start_exam')
def start_exam():#发送开启AI服务时
    if not exam_status_flag.value:  # 防止重复开启检测服务
        exam_status_flag.value = True
        init_exam_variables()
        logging.info('start_exam')
        return {"status": "SUCCESS"}
    else:
        logging.info("start_exam is already running")
        return {"status": "ALREADY_RUNNING"}

            
@app.get('/exam_status')
def exam_status():
    if not equipment_cleaning_order:#使用not来判断列表是否为空
        logging.info('equipment_cleaning_order is none')
        return {"status": "NONE"}
    else:
        json_array = [
            {"step": re.search(r'equipment_step_(\d+)', value).group(1), "image": equipment_cleaning_imgs.get(value)}
            for value in equipment_cleaning_order
        ]
        return {"status": "SUCCESS", "data": json_array}

@app.get('/stop_exam')
def stop_exam():
    if exam_status_flag.value:
        exam_status_flag.value = False
        logging.info('stop_exam')
        return {"status": "SUCCESS"}
    
    
#停止多进程函数的写法
def stop_inference_internal():
    if processes:  # 检查是否有子进程正在运行
            # 停止所有进程
        for stop_event in stop_events:
            stop_event.set()
        for process in processes:
            if process.is_alive():
                #打印当前进程的pid
                process.join(timeout=1)  # 等待1秒
                if process.is_alive():
                    logging.warning('Process did not terminate, forcing termination')
                    process.terminate()  # 强制终止子进程
                
        processes.clear()  # 清空进程列表，释放资源
        start_events.clear()
        stop_events.clear()
        logging.info('detection stopped')
        return True
    else:
        logging.info('No inference stopped')
        return False

@app.get('/stop_detection')
def stop_detection():
    if stop_inference_internal():      
        reset_shared_variables()  
        return {"status": "DETECTION_STOPPED"}
    else:
        return {"status": "No_detection_running"}

if __name__ == "__main__":
    uvicorn.run(app, host="172.16.20.163", port=5006)
    #uvicorn.run(app, host="127.0.0.1", port=5006)
