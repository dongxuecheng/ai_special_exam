import time

from platform_wearing_detect import video_decoder,process_video
from config import PLATFORM_WEARING_VIDEO_SOURCES,PLATFORM_WEARING_MODEL
from fastapi import FastAPI
import uvicorn
import logging
from fastapi.staticfiles import StaticFiles
import multiprocessing as mp
from multiprocessing import Queue
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
platform_wearing_human_in_postion=mp.Value('b', False)  # 用来判断人是否在指定位置
platform_wearing_items_nums=mp.Array('i', [0] * 3)  # 用来存储穿戴物品的数量
platform_wearing_detection_img_flag=mp.Value('b', False)  # 用来传递穿戴检测图片的标志，为真时，表示保存图片
#mp.Value适合单个值的场景，性能较慢
manager = mp.Manager()
platform_wearing_detection_img = manager.dict()  #用于存储检测焊接穿戴图片

frame_queue_list = [Queue(maxsize=50) for _ in range(2)] 

def reset_shared_variables():
    global frame_queue_list
    # 1. 重置 platform_wearing_human_in_postion
    platform_wearing_human_in_postion.value = False
    platform_wearing_detection_img_flag.value = False
    # 2. 清空 platform_wearing_items_nums
    for i in range(len(platform_wearing_items_nums)):
        platform_wearing_items_nums[i] = 0
    platform_wearing_detection_img.clear()
    frame_queue_list = [Queue(maxsize=50) for _ in range(2)] 
    # for queue in frame_queue_list:
    #     while not queue.empty():
    #         queue.get()
    
    logging.info("reset_shared_variables!!")

#@app.route('/wearing_detection', methods=['GET'])
@app.get('/wearing_detection')
def wearing_detection():
    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        stop_event.clear()
        # 使用本地的 start_events 列表，不使用 Manager
        start_events = []  # 存储每个进程的启动事件
        # 启动多个进程进行设备清洗检测

        #穿戴只需要拉一个视频流
        start_event = mp.Event()  # 为每个进程创建一个独立的事件
        start_events.append(start_event)  # 加入 start_events 列表
        process = mp.Process(target=video_decoder, args=(PLATFORM_WEARING_VIDEO_SOURCES,frame_queue_list, start_event, stop_event))
        processes.append(process)
        process.start()
        logging.info("拉流子进程运行中")

        for model_path, video_source in zip(PLATFORM_WEARING_MODEL, frame_queue_list):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            start_events.append(start_event)  # 加入 start_events 列表
            process = mp.Process(target=process_video, args=(model_path, video_source, start_event, stop_event,platform_wearing_human_in_postion, platform_wearing_items_nums, platform_wearing_detection_img_flag, platform_wearing_detection_img))
            processes.append(process)
            process.start()
            logging.info("平台穿戴检测子进程运行中")

        logging.info('start_wearing_detection')
        reset_shared_variables()

        # 等待所有进程的 start_event 被 set
        for event in start_events:
            event.wait()  # 等待每个进程通知它已经成功启动

        
        return {"status": "SUCCESS"}

    else:
        logging.info("wearing_detection already running")
        
        return {"status": "ALREADY_RUNNING"}
    

@app.get('/human_postion_status')
def human_postion_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
    if not platform_wearing_human_in_postion.value:
        logging.info('NOT_IN_POSTION')
        return {"status": "NOT_IN_POSTION"}
    else:
        logging.info('IN_POSTION')
        return {"status": "IN_POSTION"}

@app.get('/wearing_status')
def wearing_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测

    platform_wearing_detection_img_flag.value=True
    time.sleep(1)
    if 'wearing_img' not in platform_wearing_detection_img or not platform_wearing_human_in_postion.value:
        return {"status": "NONE"}
    
    wearing_items_list = ['belt', 'helmet',  'shoes']
    json_array = []
    for num, item in zip(platform_wearing_items_nums, wearing_items_list):
        json_object = {"name": item, "number": num}
        json_array.append(json_object)

    logging.info(json_array)
    image=platform_wearing_detection_img['wearing_img']
    logging.info(image)

    return {"status": "SUCCESS","data":json_array,"image":image}

               

@app.get('/end_wearing_exam')
def end_wearing_exam():
    #stop_inference_internal()
    #init_wearing_detection()
    reset_shared_variables()
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

#@app.route('/stop_detection', methods=['GET'])
@app.get('/stop_detection')
def stop_detection():
    #global inference_thread
    if stop_inference_internal():
        logging.info('detection stopped')
        return {"status": "DETECTION_STOPPED"}
    else:
        logging.info('No_detection_running')
        return {"status": "No_detection_running"}



if __name__ == '__main__':

    # Start the Flask server
    uvicorn.run(app, host="172.16.20.163", port=5003)
