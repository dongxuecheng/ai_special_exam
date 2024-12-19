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
from config import SAVE_IMG_PATH_BASKET_K2,URL_IMG_PATH_BASKET_K2,WEIGHTS_BASKET,VIDEOS_BASKET
from config import BASKET_SUSPENSION_REGION,BASKET_STEEL_WIRE_REGION,BASKET_SAFETY_LOCK_REGION,BASKET_CLEANING_OPERATION_REGION
from config import BASKET_WARNING_ZONE_REGION



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
basket_cleaning_flag = mp.Array('b', [False] * 12)
basket_cleaning_warning_zone_flag=mp.Array('b', [False] * 2)#存储两个视角下的警戒区域的检测结果
#mp.Value适合单个值的场景，性能较慢
manager = mp.Manager()
basket_cleaning_order = manager.list()#用于存储各个步骤的顺序
basket_cleaning_imgs = manager.dict()  #用于存储各个步骤的图片
basket_seg_region=manager.dict()#用于存储吊篮的分割区域
frame_queue_list = [Queue(maxsize=50) for _ in range(6)]  # 创建5个队列，用于存储视频帧
exam_status_flag = mp.Value('b', False)  # 创建一个共享变量，并初始化为False,用于在多个线程中间传递变量,表示是否开始考核,True表示开始考核

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

def save_image(basket_cleaning_imgs,basket_cleaning_order, results, step_name,basket_seg_region):

    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    img_path = f"{SAVE_IMG_PATH_BASKET_K2}/{step_name}_{save_time}.jpg"
    url_path=f"{URL_IMG_PATH_BASKET_K2}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()
    if step_name == "basket_step_1":
        annotated_frame = cv2.polylines(annotated_frame, [BASKET_WARNING_ZONE_REGION.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_2":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in BASKET_SUSPENSION_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_3":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in BASKET_STEEL_WIRE_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_4":
        annotated_frame = cv2.polylines(annotated_frame, [basket_seg_region['platform'].reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_5":
        annotated_frame = cv2.polylines(annotated_frame, [basket_seg_region['hoist_l'].reshape(-1, 1, 2),basket_seg_region['hoist_r'].reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_6":
        annotated_frame = cv2.polylines(annotated_frame, [region.reshape(-1, 1, 2) for region in BASKET_SAFETY_LOCK_REGION], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name == "basket_step_7":
        annotated_frame = cv2.polylines(annotated_frame, [basket_seg_region['electricalSystem'].reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
    elif step_name=="basket_step_10":
        annotated_frame = cv2.polylines(annotated_frame, [BASKET_CLEANING_OPERATION_REGION.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)

    cv2.imwrite(img_path, annotated_frame)
    basket_cleaning_imgs[step_name] = url_path
    basket_cleaning_order.append(step_name)
    logging.info(f"{step_name}完成")

def fetch_video_stream(rtsp_url, frame_queue_list, start_event, stop_event):  # 拉取视频流到队列中
    #队列与对应的模型
    #frame_queue_list[0]:吊篮悬挂机构
    #frame_queue_list[1]:正面，检查警戒区
    #frame_queue_list[2]:吊篮顶部姿态
    #frame_queue_list[3]:吊篮顶部视角，检查警戒区，刷子
    #frame_queue_list[4]:吊篮顶部分割
    #frame_queue_list[5]:吊篮顶部姿态，检查安全带

    cap = cv2.VideoCapture(rtsp_url)
    index = VIDEOS_BASKET.index(rtsp_url)
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
        if index == 2:#吊篮顶部数据需要放入三个队列中
            frame_queue_list[index+1].put_nowait(frame)
            frame_queue_list[index+2].put_nowait(frame)
            frame_queue_list[index+3].put_nowait(frame)
        frame_queue_list[index].put_nowait(frame)
    cap.release()

def infer_yolo(model_path,video_source, start_event, stop_event,basket_cleaning_flag,basket_cleaning_imgs,basket_cleaning_warning_zone_flag,basket_cleaning_order,basket_seg_region):#YOLO模型推理
    model = YOLO(model_path)
    while True:      
        if stop_event.is_set():
            logging.info(f"{model_path} infer_yolo is stopped")
            break        
        if video_source.empty():
            continue
        frame = video_source.get()
        #results = model.track(frame,verbose=False,conf=0.5,device='0',tracker="bytetrack.yaml")
        results = model.predict(frame, verbose=False, conf=0.3)
        #results = model.predict('rtsp://admin:yaoan1234@192.168.10.217/cam/realmonitor?channel=1&subtype=0', verbose=False, conf=0.3)
        
        if not start_event.is_set():
            start_event.set()
            logging.info(f"{model_path} infer_yolo is running")
        #for r in results:

        if model_path==WEIGHTS_BASKET[1]:#D5,detect
            boxes = results[0].boxes.xyxy  
            confidences = results[0].boxes.conf 
            classes = results[0].boxes.cls  
            basket_cleaning_warning_zone_flag[0]=False
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                confidence = confidences[i].item()
                cls = int(classes[i].item())
                label = model.names[cls]

                if label=='warning_zone':
                    centerx=(x1+x2)/2
                    centery=(y1+y2)/2
                    # point_in_region_flag=point_in_region([centerx,centery],BASKET_WARNING_ZONE_REGION_FORNT)#警戒区划分区域
                    # if point_in_region_flag:
                    basket_cleaning_flag[0]=True
                    basket_cleaning_warning_zone_flag[0]=True   

        elif model_path==WEIGHTS_BASKET[0] and not basket_cleaning_flag[1]:#D4悬挂机构
            boxes=results[0].boxes.xyxy#人体的检测框
            keypoints = results[0].keypoints.xy  
            confidences = results[0].keypoints.conf  
            for i in range(len(boxes)):
                left_wrist, right_wrist = keypoints[i][9:11].tolist()  # 获取左右手腕和左右肘的坐标
                points = [left_wrist, right_wrist]
                if any(any(point_in_region(point, region) for region in BASKET_SUSPENSION_REGION) for point in points):
                    basket_cleaning_flag[1] = True
                    
        elif model_path==WEIGHTS_BASKET[3]:#d6目标检测
            boxes = results[0].boxes.xyxy  
            confidences = results[0].boxes.conf 
            classes = results[0].boxes.cls  
            basket_cleaning_warning_zone_flag[1]=False
            brush_flag=False
            safety_belt_position=[0,0,0,0]
            self_locking_position=[0,0,0,0]
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                confidence = confidences[i].item()
                cls = int(classes[i].item())
                label = model.names[cls]

                if label=='brush' and confidence>0.7:
                    brush_flag=True
                    is_inside = point_in_region([(x1+x2)/2,(y1+y2)/2],BASKET_CLEANING_OPERATION_REGION)
                    if is_inside:
                        basket_cleaning_flag[9]=True
                        
                    #print("刷子")

                elif label=='warning_zone':
                    
                    centerx=(x1+x2)/2
                    centery=(y1+y2)/2
                    point_in_region_flag=point_in_region([centerx,centery],BASKET_WARNING_ZONE_REGION)#警戒区划分区域
                    if point_in_region_flag:
                        basket_cleaning_flag[0]=True
                        basket_cleaning_warning_zone_flag[1]=True
                
            #     elif label=='safety_belt':
            #         safety_belt_position=[x1,y1,x2,y2]

            #     elif label=='self_lock':
            #         self_locking_position=[x1,y1,x2,y2]

            
            # if IoU(safety_belt_position,self_locking_position)>0:
            #     basket_cleaning_flag[8]=True#安全带挂上自锁器
            
            if not basket_cleaning_warning_zone_flag[0] and not basket_cleaning_warning_zone_flag[1] and basket_cleaning_flag[0] and basket_cleaning_flag[9]:#当检测不到警戒区时,判定未拆除警戒区域
                basket_cleaning_flag[11]=True
                print("拆除警戒区域-----------")
            if basket_cleaning_flag[11] and not brush_flag:
                basket_cleaning_flag[10]=True

        elif model_path==WEIGHTS_BASKET[5]:#d6安全带挂设
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
                #equipment_cleaning_flag[5]=True
                basket_cleaning_flag[8]=True

        elif model_path==WEIGHTS_BASKET[4]:#d6分割
            boxes = results[0].boxes.xyxy
            if results[0].masks is not None:
                masks = results[0].masks.xy
            else:
                masks = [] 
            classes = results[0].boxes.cls 
            confidences = results[0].boxes.conf 
            if basket_cleaning_flag[7]:#如果分割结束，后面就不进行检测 
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                confidence = confidences[i].item()
                cls = int(classes[i].item())
                label = model.names[cls]
                if label=="basket" and confidence>0.7:
                    basket_seg_region["platform"]=np.array(masks[i].tolist(), np.int32)
                elif label=="hoist_l":
                    basket_seg_region["hoist_l"]=np.array(masks[i].tolist(),np.int32)
                elif label=="hoist_r":
                    basket_seg_region["hoist_r"]=np.array(masks[i].tolist(),np.int32)
                elif label=="electricalSystem":
                    basket_seg_region["electricalSystem"]=np.array(masks[i].tolist(), np.int32)
            

                    

        elif model_path==WEIGHTS_BASKET[2]:#D6,pose
            boxes=results[0].boxes.xyxy#人体的检测框
            keypoints = results[0].keypoints.xy  
            confidences = results[0].keypoints.conf  

            basket_person_flag=False#空载的人，当检测不到则为False
            for i in range(len(boxes)):
                #当有检测框，则说明有人

                head_points=keypoints[i][0:5].tolist()#获得头部的五个关键点
                if 'platform' in basket_seg_region and any(point_in_region(point, basket_seg_region['platform']) for point in head_points):
                    basket_person_flag=True#表面人在吊篮内

                left_wrist, right_wrist = keypoints[i][9:11].tolist()#获取左右手腕和左右肘的坐标
                points = [left_wrist, right_wrist]
                if not basket_cleaning_flag[2]:
                    is_inside1 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[0]) for point in points)
                    is_inside2 = any(point_in_region(point, BASKET_STEEL_WIRE_REGION[1]) for point in points)
                    if is_inside1 or is_inside2:
                        basket_cleaning_flag[2]=True

                if not basket_cleaning_flag[3] and 'platform' in basket_seg_region:
                    is_inside = any(point_in_region(point,basket_seg_region['platform']) for point in points)
                    if is_inside:
                        basket_cleaning_flag[3]=True

                if not basket_cleaning_flag[4] and 'hoist_l' in basket_seg_region and 'hoist_r' in basket_seg_region:
                    is_inside1 = any(point_in_region(point,basket_seg_region['hoist_l']) for point in points)
                    is_inside2 = any(point_in_region(point,basket_seg_region['hoist_r']) for point in points)   
                    if is_inside1 or is_inside2:
                        basket_cleaning_flag[4]=True

                if not basket_cleaning_flag[5]:
                    is_inside1 = any(point_in_region(point, BASKET_SAFETY_LOCK_REGION[0]) for point in points)
                    is_inside2 = any(point_in_region(point, BASKET_SAFETY_LOCK_REGION[1]) for point in points)
                    
                    if is_inside1 or is_inside2:
                        basket_cleaning_flag[5]=True

                if not basket_cleaning_flag[6] and 'electricalSystem' in basket_seg_region:
                    is_inside = any(point_in_region(point,basket_seg_region['electricalSystem']) for point in points)
                    if is_inside:
                        basket_cleaning_flag[6]=True

            if not basket_person_flag and 'platform' in basket_seg_region and not basket_cleaning_flag[7]:
                #basket_cleaning_flag[7]=True
                if rect_polgyon_iou([446,883,765,1163],basket_seg_region['platform'])>0.01:
                    basket_cleaning_flag[7]=True


            #检测人是否在吊篮内
            # if not basket_person_flag and 'basket' in basket_seg_region and not basket_cleaning_flag[7]:
            #     #basket_cleaning_flag[7]=True
            #     if rect_polgyon_iou([446,883,765,1163],basket_seg_region['platform'])>0.01:
            #         basket_cleaning_flag[7]=True
            #         logging.info("空载")

            #不检测人是否在吊篮内
            # if 'platform' in basket_seg_region and not basket_cleaning_flag[7]:
            #     #basket_cleaning_flag[7]=True
            #     if rect_polgyon_iou([446,883,765,1163],basket_seg_region['platform'])>0:
            #         basket_cleaning_flag[7]=True
            #         logging.info("空载")

            


        # steps = [
        #     (WEIGHTS_BASKET[1], basket_cleaning_flag[1], "basket_step_2"),
        #     (WEIGHTS_BASKET[4], basket_cleaning_flag[2], "basket_step_3"),
        #     (WEIGHTS_BASKET[4], basket_cleaning_flag[3], "basket_step_4"),
        #     (WEIGHTS_BASKET[4], basket_cleaning_flag[4], "basket_step_5"),
        #     (WEIGHTS_BASKET[4], basket_cleaning_flag[5], "basket_step_6"),
        #     (WEIGHTS_BASKET[4], basket_cleaning_flag[6], "basket_step_7"),
        #     (WEIGHTS_BASKET[4], basket_cleaning_flag[7], "basket_step_8"),
        #     (WEIGHTS_BASKET[2], basket_cleaning_flag[0], "basket_step_1"),
        #     (WEIGHTS_BASKET[2], basket_cleaning_flag[11], "basket_step_12"),
        #     (WEIGHTS_BASKET[2], basket_cleaning_flag[9], "basket_step_10"),
        #     (WEIGHTS_BASKET[2], basket_cleaning_flag[10], "basket_step_11"),
        #     (WEIGHTS_BASKET[2], basket_cleaning_flag[8], "basket_step_9"),
        # ]

        # if exam_status_flag.value:
        #     for model_name, flag, step in steps:
        #         if model_path == model_name and step not in basket_cleaning_imgs and flag:
        #             save_image(basket_cleaning_imgs, basket_cleaning_order, results, step, basket_seg_region)

        steps = [
            (WEIGHTS_BASKET[0], basket_cleaning_flag[1], "basket_step_2", "悬挂机构检测完毕"),
            (WEIGHTS_BASKET[2], basket_cleaning_flag[2], "basket_step_3", "钢丝绳"),
            (WEIGHTS_BASKET[2], basket_cleaning_flag[3], "basket_step_4", "平台"),
            (WEIGHTS_BASKET[2], basket_cleaning_flag[4], "basket_step_5", "提升机"),
            (WEIGHTS_BASKET[2], basket_cleaning_flag[5], "basket_step_6", "安全锁"),
            (WEIGHTS_BASKET[2], basket_cleaning_flag[6], "basket_step_7", "电气系统"),
            (WEIGHTS_BASKET[2], basket_cleaning_flag[7], "basket_step_8", "空载"),
            (WEIGHTS_BASKET[3], basket_cleaning_flag[0], "basket_step_1", "警戒区"),
            (WEIGHTS_BASKET[3], basket_cleaning_flag[11], "basket_step_12", "警戒区消失"),
            (WEIGHTS_BASKET[3], basket_cleaning_flag[9], "basket_step_10", "清洗作业"),
            (WEIGHTS_BASKET[3], basket_cleaning_flag[10], "basket_step_11", "清理现场"),
            (WEIGHTS_BASKET[5], basket_cleaning_flag[8], "basket_step_9", "安全带挂设"),
        ]

        if exam_status_flag.value:
            for model_name, flag, step, message in steps:
                if model_path == model_name and flag and step not in basket_cleaning_imgs:
                    save_image(basket_cleaning_imgs, basket_cleaning_order, results, step, basket_seg_region)
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
    for i in range(len(basket_cleaning_flag)):
        basket_cleaning_flag[i] = False    
    basket_cleaning_imgs.clear()
    basket_seg_region.clear()
    basket_cleaning_order[:]=[]
    basket_cleaning_warning_zone_flag[0]=False
    basket_cleaning_warning_zone_flag[1]=False



@app.get('/start_detection')
def start_detection():
    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        for video_source in VIDEOS_BASKET:
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            stop_event=mp.Event()
            process = mp.Process(target=fetch_video_stream, args=(video_source,frame_queue_list, start_event, stop_event))
            stop_events.append(stop_event)
            start_events.append(start_event)  # 加入 start_events 列表，因为start_events是列表，append或clear不需要加global
            processes.append(process)
            process.start()

        for model_path, video_source in zip(WEIGHTS_BASKET, frame_queue_list):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            stop_event=mp.Event()
            process = mp.Process(target=infer_yolo, args=(model_path,video_source, start_event, stop_event,basket_cleaning_flag,basket_cleaning_imgs,basket_cleaning_warning_zone_flag,basket_cleaning_order,basket_seg_region))
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
    if not basket_cleaning_order:#使用not来判断列表是否为空
        logging.info('basket_cleaning_order is none')
        return {"status": "NONE"}
    else:
        json_array = [
            {"step": re.search(r'basket_step_(\d+)', value).group(1), "image": basket_cleaning_imgs.get(value)}
            for value in basket_cleaning_order
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
    uvicorn.run(app, host="172.16.20.163", port=5005)
    #uvicorn.run(app, host="127.0.0.1", port=5005)
