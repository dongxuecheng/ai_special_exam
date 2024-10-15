import re
import time
from flask import Flask, jsonify,send_from_directory

#from globals import inference_thread, stop_event,lock,redis_client
import multiprocessing as mp
from welding_reset_detect import process_video as process_video_reset
from welding_exam_detect import process_video as process_video_exam
from config import WELDING_MODEL_PATHS, WELDING_VIDEO_SOURCES

app = Flask(__name__)

# 全局变量
processes = []
stop_event = mp.Event()
#mp.Array性能较高，适合大量写入的场景
welding_reset_flag = mp.Array('b', [False] * 5) # 创建一个长度为5的共享数组，并初始化为False,用于在多个线程中间传递变量
welding_exam_flag = mp.Array('b', [False] * 13)  # 创建一个长度为5的共享数组，并初始化为False,用于在多个线程中间传递变量
#mp.Value适合单个值的场景，性能较慢
manager = mp.Manager()
welding_reset_imgs = manager.dict()  #用于存储各个步骤的图片
welding_exam_imgs = manager.dict()  #用于存储焊接考核各个步骤的图片
welding_exam_order = manager.list()#用于存储焊接考核各个步骤的顺序
def reset_shared_variables():

    for i in range(len(welding_reset_flag)):
        welding_reset_flag[i] = False
    for i in range(len(welding_exam_flag)):
        welding_exam_flag[i] = False
    
    welding_reset_imgs.clear()
    welding_exam_imgs.clear()

    


@app.route('/reset_detection', methods=['GET'])
def reset_detection():#发送开启AI服务时，检测复位

    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        stop_event.clear()

        # 使用本地的 start_events 列表，不使用 Manager
        start_events = []  # 存储每个进程的启动事件

        # 启动多个进程进行设备清洗检测
        for model_path, video_source in zip(WELDING_MODEL_PATHS, WELDING_VIDEO_SOURCES):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            start_events.append(start_event)  # 加入 start_events 列表

            process = mp.Process(target=process_video_reset, args=(model_path,video_source, start_event, stop_event, welding_reset_flag, welding_reset_imgs))
            processes.append(process)
            process.start()
            print("焊接复位子进程运行中")

        app.logger.info('start_welding_reset_detection')
        reset_shared_variables()

        # 等待所有进程的 start_event 被 set
        for event in start_events:
            event.wait()  # 等待每个进程通知它已经成功启动

        return jsonify({"status": "SUCCESS"}), 200

    else:
        app.logger.info("welding_reset_detection")
        return jsonify({"status": "ALREADY_RUNNING"}), 200  

@app.route('/reset_status', methods=['GET']) 
def reset_status():#获取复位检测状态
    if sum(welding_reset_flag)==0:#表明不需要复位
        app.logger.info('reset_all is true')
        #此时复位的检测还在进行，需要停止复位检测
        stop_inference_internal()

        return jsonify({"status": "RESET_ALL"}), 200
    
    else:
        app.logger.info('reset_all is false')
        json_array = []
        for key,value in welding_reset_imgs.items():
            
            match = re.search(r'reset_step_(\d+)', key)
            step_number = match.group(1)
            json_object = {"resetStep": step_number, "image": value}
            json_array.append(json_object)

        reset_shared_variables()
        return jsonify({"status": "NOT_RESET_ALL","data":json_array}), 200

@app.route('/welding_detection', methods=['GET']) 
def welding_detection():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测

    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        stop_event.clear()

        start_events = []  # 存储每个进程的启动事件

        for model_path, video_source in zip(WELDING_MODEL_PATHS, WELDING_VIDEO_SOURCES):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            start_events.append(start_event)  # 加入 start_events 列表

            process = mp.Process(target=process_video_exam, args=(model_path,video_source, start_event, stop_event, welding_exam_flag, welding_exam_imgs,welding_exam_order))
            processes.append(process)
            process.start()
            print("焊接考核子进程运行中")

        app.logger.info('start_welding_exam_detection')
        reset_shared_variables()

        # 等待所有进程的 start_event 被 set
        for event in start_events:
            event.wait()  # 等待每个进程通知它已经成功启动

        return jsonify({"status": "SUCCESS"}), 200

    else:
        app.logger.info("welding_exam_detection")
        return jsonify({"status": "ALREADY_RUNNING"}), 200  
            

@app.route('/welding_status', methods=['GET']) 
def welding_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测

    
    if len(welding_exam_order)==0:#表示还没有检测到任何一个焊接步骤
        app.logger.info('welding_exam_order is none')
        return jsonify({"status": "NONE"}), 200

    else:


        json_array = []
        for value in welding_exam_order:
            match = re.search(r'welding_exam_(\d+)', value)
            step_number = match.group(1)
            json_object = {"step": step_number, "image": welding_exam_imgs.get(f"welding_exam_{step_number}")}
            json_array.append(json_object)
        return jsonify({"status": "SUCCESS","data":json_array}), 200

@app.route('/end_welding_exam', methods=['GET'])#点击考试结束按钮，停止检测，并复位
def end_welding_exam():
    stop_inference_internal()
    time.sleep(1)
    return reset_detection()

    
#停止多进程函数的写法
def stop_inference_internal():
    global processes
    if processes:  # 检查是否有子进程正在运行
        stop_event.set()  # 设置停止事件标志，通知所有子进程停止运行

        # 等待所有子进程结束
        for process in processes:
            if process.is_alive():
                process.join()  # 等待每个子进程结束
                
        processes = []  # 清空进程列表，释放资源
        app.logger.info('detection stopped')
        return True
    else:
        app.logger.info('No inference stopped')
        return False


@app.route('/stop_detection', methods=['GET'])
def stop_inference():
    #global inference_thread
    if stop_inference_internal():
        app.logger.info('detection stopped')
        return jsonify({"status": "DETECTION_STOPPED"}), 200
    else:
        app.logger.info('No_detection_running')
        return jsonify({"status": "No_detection_running"}), 200


@app.route('/images/<filename>')
def get_image(filename):
    app.logger.info('get_image'+filename)
    #pdb.set_trace()
    return send_from_directory('static/images', filename)


if __name__ == '__main__':
    # Start the Flask server
    app.run(debug=False, host='172.16.20.163', port=5002)
