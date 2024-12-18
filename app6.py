import re

from fastapi import FastAPI
import uvicorn
import logging
from fastapi.staticfiles import StaticFiles

from equipment_cleaning_detect import process_video,video_decoder
from config import EQUIPMENT_CLEANING_MODEL_SOURCES, EQUIPMENT_CLEANING_VIDEO_SOURCES
from multiprocessing import Queue
import multiprocessing as mp

#焊接考核的穿戴
app = FastAPI()
# 挂载目录作为静态文件路径
app.mount("/images", StaticFiles(directory="static/images"))
# 获得uvicorn服务器的日志记录器
logging = logging.getLogger("uvicorn")


# 全局变量
processes = []
stop_event = mp.Event()
#mp.Array性能较高，适合大量写入的场景
equipment_cleaning_flag = mp.Array('b', [False] * 12)  # 创建一个长度为12的共享数组，并初始化为False,用于在多个线程中间传递变量
person_postion = mp.Array('f', [0.0] * 4)  # 创建一个长度为4的共享数组，并初始化为0.0,用于在多个线程中间传递浮点型变量#用于存储人的位置信息
equipment_warning_zone_flag=mp.Array('b', [False] * 2)#存储两个视角下的警戒区域的检测结果
#mp.Value适合单个值的场景，性能较慢
manager = mp.Manager()
equipment_cleaning_order = manager.list()#用于存储各个步骤的顺序
equipment_cleaning_imgs = manager.dict()  #用于存储各个步骤的图片
frame_queue_list = [Queue(maxsize=50) for _ in range(5)]  # 创建6个队列，用于存储视频帧

# 清空并重新初始化所有变量
def reset_shared_variables():
    global frame_queue_list
    for i in range(len(equipment_cleaning_flag)):
        equipment_cleaning_flag[i] = False
    
    person_postion[0] = 0.0
    person_postion[1] = 0.0
    person_postion[2] = 0.0
    person_postion[3] = 0.0

    equipment_warning_zone_flag[0]=False
    equipment_warning_zone_flag[1]=False

    # 2. 清空 equipment_cleaning_order
    equipment_cleaning_order[:] = []  # 使用切片来清空 ListProxy
    
    # 3. 清空 equipment_cleaning_imgs
    equipment_cleaning_imgs.clear()
    frame_queue_list = [Queue(maxsize=50) for _ in range(5)]
    # for queue in frame_queue_list:
    #     while not queue.empty():
    #         queue.get()

@app.get('/equipment_cleaning_detection')
def equipment_cleaning_detection():  # 开启平台搭设检测

    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        stop_event.clear()

        # 使用本地的 start_events 列表，不使用 Manager
        start_events = []  # 存储每个进程的启动事件
        for video_source in EQUIPMENT_CLEANING_VIDEO_SOURCES:
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            start_events.append(start_event)  # 加入 start_events 列表
            process = mp.Process(target=video_decoder, args=(video_source,frame_queue_list, start_event, stop_event))
            processes.append(process)
            process.start()
            logging.info("拉流子进程运行中")
        
        # 启动多个进程进行设备清洗检测
        for model_path, video_source in zip(EQUIPMENT_CLEANING_MODEL_SOURCES, frame_queue_list):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            start_events.append(start_event)  # 加入 start_events 列表

            process = mp.Process(target=process_video, args=(model_path, video_source, start_event, stop_event,equipment_cleaning_flag,equipment_cleaning_imgs,equipment_cleaning_order,person_postion,equipment_warning_zone_flag))
            processes.append(process)
            process.start()
            logging.info("单人吊具清洗子进程运行中")

        logging.info('start_equipment_cleaning_detection')
        reset_shared_variables()

        # 等待所有进程的 start_event 被 set
        for event in start_events:
            event.wait()  # 等待每个进程通知它已经成功启动

        #return jsonify({"status": "SUCCESS"}), 200
        return {"status": "SUCCESS"}

    else:
        logging.info("reset_detection already running")
        return {"status": "ALREADY_RUNNING"}

@app.get('/equipment_cleaning_status')
def equipment_cleaning_status():#获取平台搭设状态状态
    if len(equipment_cleaning_order)==0:#平台搭设步骤还没有一个完成
        logging.info('equipment_cleaning_order is none')

        return {"status": "NONE"}
    
    else:
        json_array = []
        for value in equipment_cleaning_order:
            match = re.search(r'equipment_step_(\d+)', value)
            step_number = match.group(1)
            json_object = {"step": step_number, "image": equipment_cleaning_imgs.get(f"equipment_step_{step_number}")}
            json_array.append(json_object) 

        return {"status": "SUCCESS","data":json_array}

@app.get('/equipment_cleaning_finish')
def equipment_cleaning_finish():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测

    stop_inference_internal()
    logging.info('bequipment_cleaning_order')
    return {"status": "SUCCESS"}




#停止多进程函数的写法
def stop_inference_internal():
    global processes
    if processes:  # 检查是否有子进程正在运行
        stop_event.set()  # 设置停止事件标志，通知所有子进程停止运行
        
        # 等待所有子进程结束
        for process in processes:
            if process.is_alive():
                process.join(timeout=1)  # 等待1秒
                if process.is_alive():
                    logging.warning('Process did not terminate, forcing termination')
                    process.terminate()  # 强制终止子进程
        
        processes = []  # 清空进程列表，释放资源
        logging.info('detection stopped')
        reset_shared_variables()
        return True
    else:
        logging.info('No inference stopped')
        return False

@app.get('/stop_detection')
def stop_detection():
    if stop_inference_internal():
        logging.info('detection stopped')
        #reset_shared_variables()
        return {"status": "DETECTION_STOPPED"}
    else:
        logging.info('No_detection_running')
        return {"status": "No_detection_running"}

if __name__ == "__main__":
    uvicorn.run(app, host="172.16.20.163", port=5006)