#1 start_detection #开启检测服务

#2 stop_detection #停止检测服务

#3 reset_status #获取复位检测状态

#4 start_exam #开始焊接考核

#5 exam_status #获取焊接考核状态

#6 stop_exam #停止焊接考核

import cv2
from ultralytics import YOLO
from shapely.geometry import box, Polygon
from datetime import datetime
from utils.tool import IoU
import re
from fastapi import FastAPI
import uvicorn
import logging
from fastapi.staticfiles import StaticFiles
import multiprocessing as mp
from config import WELDING_MODEL_PATHS, WELDING_VIDEO_SOURCES, SAVE_IMG_PATH, POST_IMG_PATH2,WELDING_REGION1, WELDING_REGION2, WELDING_REGION3
from multiprocessing import Queue

#焊接考核的穿戴
app = FastAPI()
# 挂载目录作为静态文件路径
app.mount("/images", StaticFiles(directory="static/images"))
# 获得uvicorn服务器的日志记录器
logging = logging.getLogger("uvicorn")

# 全局变量
processes = []
start_events = []  # 存储每个进程的启动事件
stop_events = []  # 存储每个进程的停止事件

#mp.Array性能较高，适合大量写入的场景
welding_reset_flag = mp.Array('b', [False] * 5) # 创建一个长度为5的共享数组，并初始化为False,用于在多个线程中间传递变量
welding_exam_flag = mp.Array('b', [False] * 14)  # 创建一个长度为5的共享数组，并初始化为False,用于在多个线程中间传递变量
#mp.Value适合单个值的场景，性能较慢
manager = mp.Manager()
welding_reset_imgs = manager.dict()  #用于存储各个步骤的图片
welding_exam_imgs = manager.dict()  #用于存储焊接考核各个步骤的图片
welding_exam_order = manager.list()#用于存储焊接考核各个步骤的顺序

exam_status_flag = mp.Value('b', False)  # 创建一个共享变量，并初始化为False,用于在多个线程中间传递变量,表示是否开始考核,True表示开始考核
frame_queue_list = [Queue(maxsize=50) for _ in range(5)]  # 创建6个队列，用于存储视频帧


def fetch_video_stream(rtsp_url, frame_queue_list, start_event, stop_event):  # 拉取视频流到队列中
    cap = cv2.VideoCapture(rtsp_url)
    index = WELDING_VIDEO_SOURCES.index(rtsp_url)
    while cap.isOpened():
        if stop_event.is_set():  # 控制停止推理
            logging.info("fetch_video_stream is stopped")
            break
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 != 0:
            continue
        if not start_event.is_set():
            start_event.set()
            logging.info(f'fetch_video_stream{rtsp_url}')
        frame_queue_list[index].put_nowait(frame)
    cap.release()

def infer_yolo(model_path, video_source, start_event, stop_event,welding_reset_flag, welding_reset_imgs,welding_exam_flag, welding_exam_imgs,welding_exam_order):#YOLO模型推理
    model = YOLO(model_path)
    while True:      
        if stop_event.is_set():
            logging.info(f"{model_path} infer_yolo is stopped")
            break
        
        if video_source.empty():
            continue
        frame = video_source.get()
        #results = model.track(frame,verbose=False,conf=0.5,device='0',tracker="bytetrack.yaml")
        results = model.predict(frame,verbose=False,conf=0.5,device='0')
        if not start_event.is_set():
            start_event.set()
            logging.info(f"{model_path} infer_yolo is running")

        if model_path == WELDING_MODEL_PATHS[1]:#分类模型
            if results[0].probs.top1conf>0.8:
                label=model.names[results[0].probs.top1]
                if label == "component":
                    welding_reset_flag[3]=True 
                    welding_exam_flag[3]=True
                elif label == "empty":
                    welding_reset_flag[3]=False
                    welding_exam_flag[10]=False
                elif label=='welding':
                    welding_exam_flag[6]=True#表示有焊接
                elif label=='sweep' and welding_exam_flag[6]==True:
                    welding_exam_flag[11]=True#表示有打扫,可能有误识别                
        else:
            boxes = results[0].boxes.xyxy  # 提取所有检测到的边界框坐标
            confidences = results[0].boxes.conf  # 提取所有检测到的置信度
            classes = results[0].boxes.cls  # 提取所有检测到的类别索引

            if model_path == WELDING_MODEL_PATHS[4]:
                welding_reset_flag[2]=False

                
            #当画面没有油桶时，给个初值为安全
            if model_path == WELDING_MODEL_PATHS[2]:
                welding_reset_flag[0]=True
                welding_exam_flag[0]=True #表示油桶在安全区域

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                confidence = confidences[i].item()
                cls = int(classes[i].item())
                label = model.names[cls]

                if label=="dump":#检测油桶
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    center_point = (int(x_center), int(y_center))
                    is_inside = cv2.pointPolygonTest(WELDING_REGION2.reshape((-1, 1, 2)), center_point, False)

                    
                    if is_inside>=0 :
                        welding_reset_flag[0]=False #表示油桶在危险区域
                        welding_exam_flag[0]=False
                    else:
                        welding_reset_flag[0]=True
                        welding_exam_flag[0]=True 

                if label=='turnon':#检测总开关
                    welding_reset_flag[1]=True
                    welding_exam_flag[1]=True
                if label=='turnoff':#检测总开关
                    welding_reset_flag[1]=False
                    if welding_exam_flag[1]:
                        welding_exam_flag[12]=True
                        #TODO:最后一步打扫卫生，先临时赋值
                        welding_exam_flag[13]=True

                if label== "open":#检测焊机开关
                    welding_reset_flag[4] = True
                    welding_exam_flag[4] = True

                if label=="close":#检测焊机开关
                    welding_reset_flag[4] = False
                    if welding_exam_flag[4]:#表示已经检测到焊机开关打开过了,但是现在没有打开
                        welding_exam_flag[8] = True

                if label=="grounding_wire" :
                    rect_shapely = box(x1,y1, x2, y2)#使用shapely库创建的矩形
                    WELDING_REGION3_shapely = Polygon(WELDING_REGION3.tolist()) #shapely计算矩形检测框和多边形的iou使用
                    intersection = rect_shapely.intersection(WELDING_REGION3_shapely)
                            # 计算并集
                    union = rect_shapely.union(WELDING_REGION3_shapely)
                            # 计算 IoU
                    iou = intersection.area / union.area


                    if iou>0 :
                        welding_reset_flag[2]=True #表示搭铁线连接在焊台上
                        welding_exam_flag[2]=True
                    else:
                        welding_reset_flag[2]=False #表示未连接上
                        if welding_exam_flag[2]:#表示已经检测到搭铁线连接上了,但是现在没有连接上
                            welding_exam_flag[9]=True#焊接考核第十步
                
                if label=="mask":
                    #mask=True #表示戴面罩
                    iou=IoU(boxes[i].tolist(),WELDING_REGION1)
                    welding_exam_flag[5]=True if iou>0 else False #表示戴面罩


                if label=="gloves":
                    if confidence>0.5:
                        welding_exam_flag[7]=True#表示戴手套
                    else:
                        welding_exam_flag[7]=False

        reset_steps = {
            WELDING_MODEL_PATHS[3]: (welding_reset_flag[1], 'reset_step_2', "当前总开关没有复位"),
            WELDING_MODEL_PATHS[2]: (welding_reset_flag[0], 'reset_step_1', "当前油桶没有复位"),
            WELDING_MODEL_PATHS[0]: (welding_reset_flag[4], 'reset_step_5', "当前焊机开关没有复位"),
            WELDING_MODEL_PATHS[4]: (welding_reset_flag[2], 'reset_step_3', "搭铁线没有复位"),
            WELDING_MODEL_PATHS[1]: (welding_reset_flag[3], 'reset_step_4', "当前焊件没有复位")
        }

        if not exam_status_flag.value and model_path in reset_steps:
            flag, step, message = reset_steps[model_path]
            if flag and step not in welding_reset_imgs:
                logging.info(message)
                save_image_reset(welding_reset_imgs, results, step)


        exam_steps = {
            WELDING_MODEL_PATHS[1]: [
            (welding_exam_flag[11], 'welding_exam_12'),
            (welding_exam_flag[3], 'welding_exam_4'),
            (welding_exam_flag[10], 'welding_exam_11'),
            (welding_exam_flag[6], 'welding_exam_7')
            ],
            WELDING_MODEL_PATHS[2]: [
            (welding_exam_flag[0], 'welding_exam_1'),
            (welding_exam_flag[13], 'welding_exam_14')
            ],
            WELDING_MODEL_PATHS[3]: [
            (welding_exam_flag[1], 'welding_exam_2'),
            (welding_exam_flag[12], 'welding_exam_13')
            ],
            WELDING_MODEL_PATHS[0]: [
            (welding_exam_flag[4], 'welding_exam_5'),
            (welding_exam_flag[8], 'welding_exam_9')
            ],
            WELDING_MODEL_PATHS[4]: [
            (welding_exam_flag[7], 'welding_exam_8'),
            (welding_exam_flag[2], 'welding_exam_3'),
            (welding_exam_flag[9], 'welding_exam_10'),
            (welding_exam_flag[5], 'welding_exam_6')
            ]
        }

        if exam_status_flag.value and model_path in exam_steps:
            for flag, step in exam_steps[model_path]:
                if flag and step not in welding_exam_imgs:
                    save_image_exam(welding_exam_imgs, results, step, welding_exam_order)
        

def save_image_reset(welding_reset_imgs,results, step_name):#保存图片
    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    imgpath = f"{SAVE_IMG_PATH}/{step_name}_{save_time}.jpg"
    postpath = f"{POST_IMG_PATH2}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()
    cv2.imwrite(imgpath, annotated_frame)
    welding_reset_imgs[step_name]=postpath

def save_image_exam(welding_exam_imgs,results, step_name,welding_exam_order):
    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    imgpath = f"{SAVE_IMG_PATH}/{step_name}_{save_time}.jpg"
    postpath = f"{POST_IMG_PATH2}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()
    cv2.imwrite(imgpath, annotated_frame)
    welding_exam_imgs[step_name]=postpath
    welding_exam_order.append(step_name)
    logging.info(f"{step_name}完成")

def reset_shared_variables():
    global frame_queue_list
    exam_status_flag.value = False
    init_reset_variables()
    init_exam_variables()
    for queue in frame_queue_list:
        while not queue.empty():
            queue.get()
            logging.info("frame_queue_list is empty")

def init_exam_variables():
    for i in range(len(welding_exam_flag)):
        welding_exam_flag[i] = False    
    welding_exam_imgs.clear()
    welding_exam_order[:]=[]


def init_reset_variables():
    for i in range(len(welding_reset_flag)):
        welding_reset_flag[i] = False
    welding_reset_imgs.clear()

@app.get('/start_detection')
def start_detection():#发送开启AI服务时，检测复位
    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        for video_source in WELDING_VIDEO_SOURCES:
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            stop_event=mp.Event()
            process = mp.Process(target=fetch_video_stream, args=(video_source,frame_queue_list, start_event, stop_event))
            stop_events.append(stop_event)
            start_events.append(start_event)  # 加入 start_events 列表，因为start_events是列表，append或clear不需要加global
            processes.append(process)
            process.start()

        for model_path, video_source in zip(WELDING_MODEL_PATHS, frame_queue_list):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            stop_event=mp.Event()
            process = mp.Process(target=infer_yolo, args=(model_path,video_source, start_event, stop_event, welding_reset_flag, welding_reset_imgs,welding_exam_flag, welding_exam_imgs,welding_exam_order))
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

@app.get('/reset_status')#获取复位检测状态
def reset_status():
    if not any(welding_reset_flag):  # 表明不需要复位,如果 welding_reset_flag 列表中的所有元素都为 False，则 any(welding_reset_flag) 返回 False，not any(welding_reset_flag) 返回 True。
        logging.info('reset_all!')
        return {"status": "RESET_ALL"}
    else:
        logging.info('reset_all is false')
        json_array = [
            {"resetStep": re.search(r'reset_step_(\d+)', key).group(1), "image": value}
            for key, value in welding_reset_imgs.items()
        ]
        init_reset_variables()#初始化复位变量
        return {"status": "NOT_RESET_ALL", "data": json_array}

            
@app.get('/exam_status')
def exam_status():
    if not welding_exam_order:#使用not来判断列表是否为空
        logging.info('welding_exam_order is none')
        return {"status": "NONE"}
    else:
        json_array = [
            {"step": re.search(r'welding_exam_(\d+)', value).group(1), "image": welding_exam_imgs.get(value)}
            for value in welding_exam_order
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
                logging.info(f"process.pid:{process.pid}")
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
    uvicorn.run(app, host="172.16.20.163", port=5002)
