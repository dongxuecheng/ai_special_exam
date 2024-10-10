
import time
from flask import Flask, jsonify,send_from_directory
from welding_wearing_detect import process_video
#from globals import inference_thread, stop_event,lock,redis_client
from config import WELDING_WEARING_MODEL, WELDING_WEARING_VIDEO_SOURCES
import multiprocessing as mp

#焊接考核的穿戴
app = Flask(__name__)


# 全局变量
processes = []
stop_event = mp.Event()
#mp.Array性能较高，适合大量写入的场景
welding_wearing_human_in_postion=mp.Value('b', False)  # 用来判断人是否在指定位置
welding_wearing_items_nums=mp.Array('i', [0] * 5)  # 用来存储穿戴物品的数量
welding_wearing_detection_img_flag=mp.Value('b', False)  # 用来传递穿戴检测图片的标志，为真时，表示保存图片
#mp.Value适合单个值的场景，性能较慢
manager = mp.Manager()

welding_wearing_detection_img = manager.dict()  #用于存储检测焊接穿戴图片

def reset_shared_variables():
    # 1. 重置 welding_wearing_human_in_postion
    welding_wearing_human_in_postion.value = False
    welding_wearing_detection_img_flag.value = False
    # 2. 清空 welding_wearing_items_nums
    for i in range(len(welding_wearing_items_nums)):
        welding_wearing_items_nums[i] = 0

    # 3. 清空 welding_wearing_detection_img
    welding_wearing_detection_img.clear()

# Define the /wearing_detection endpoint
@app.route('/wearing_detection', methods=['GET'])
def wearing_detection():
    # global inference_thread#当全局变量需要重新赋值时，需要用global关键字声明

    # if inference_thread is None or not inference_thread.is_alive():
    #     stop_event.clear()#stop_event不用global声明，因为不需要重新赋值，他只是调用了其方法，并没有重新赋值
        
    #     start_events = []#给每个线程一个事件，让我知道某个线程是否开始检测
    #     inference_thread = threading.Thread(target=start_wearing_detection,args=(start_events,))
    #     inference_thread.start()
    #     init_wearing_detection()

    #     # 等待所有YOLO线程开始检测，两个线程检测完毕时，才返回SUCCESS
    #     for event in start_events:
    #         event.wait()

    #     app.logger.info('start_wearing_detection')
    #     return jsonify({"status": "SUCCESS"}), 200
    
    # else:
    #     app.logger.info("start_wearing_detection already running")   
    #     return jsonify({"status": "ALREADY_RUNNING"}), 200
    if not any(p.is_alive() for p in processes):  # 防止重复开启检测服务
        stop_event.clear()

        # 使用本地的 start_events 列表，不使用 Manager
        start_events = []  # 存储每个进程的启动事件

        
        # 启动多个进程进行设备清洗检测
        for model_path, video_source in zip(WELDING_WEARING_MODEL, WELDING_WEARING_VIDEO_SOURCES):
            start_event = mp.Event()  # 为每个进程创建一个独立的事件
            start_events.append(start_event)  # 加入 start_events 列表

            process = mp.Process(target=process_video, args=(model_path, video_source, start_event, stop_event, welding_wearing_human_in_postion, welding_wearing_items_nums, welding_wearing_detection_img_flag, welding_wearing_detection_img))
            processes.append(process)
            process.start()
            print("焊接穿戴检测子进程运行中")

        app.logger.info('start_wearing_detection')
        reset_shared_variables()

        # 等待所有进程的 start_event 被 set
        for event in start_events:
            event.wait()  # 等待每个进程通知它已经成功启动

        return jsonify({"status": "SUCCESS"}), 200

    else:
        app.logger.info("wearing_detection already running")
        return jsonify({"status": "ALREADY_RUNNING"}), 200  
    

@app.route('/human_postion_status', methods=['GET']) 
def human_postion_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
    #global inference_thread
    # if redis_client.get("welding_wearing_human_in_postion")=='False':
    #     app.logger.info('NOT_IN_POSTION')
    #     return jsonify({"status": "NOT_IN_POSTION"}), 200
    if not welding_wearing_human_in_postion.value:
        app.logger.info('NOT_IN_POSTION')
        return jsonify({"status": "NOT_IN_POSTION"}), 200
    else:
        app.logger.info('IN_POSTION')
        return jsonify({"status": "IN_POSTION"}), 200

@app.route('/wearing_status', methods=['GET']) 
def wearing_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
    #global inference_thread
    #with lock:   
        #TODO 若出现异常再发送FAIL.
    #redis_client.set("welding_wearing_detection_img_flag",'True')
    welding_wearing_detection_img_flag.value=True
    time.sleep(1)
    # if not redis_client.exists("welding_wearing_items_nums") or not redis_client.exists("welding_wearing_detection_img"):
    #     return jsonify({"status": "NONE"}), 200##表示穿戴检测线程还未检测完
    if 'wearing_img' not in welding_wearing_detection_img or not welding_wearing_human_in_postion.value:
        return jsonify({"status": "NONE"}), 200
    else:
        #wearing_items_nums = redis_client.lrange("welding_wearing_items_nums", 0, -1)
        wearing_items_list = ['pants', 'jacket', 'helmet', 'gloves', 'shoes']
        json_array = []
        for num, item in zip(welding_wearing_items_nums, wearing_items_list):
            json_object = {"name": item, "number": num}
            json_array.append(json_object)

        app.logger.info(json_array)
        #image=redis_client.get("welding_wearing_detection_img")
        image=welding_wearing_detection_img['wearing_img']
        app.logger.info(image)

        return jsonify({"status": "SUCCESS","data":json_array,"image":image}), 200

               
@app.route('/end_wearing_exam', methods=['GET'])
def end_wearing_exam():
    #init_wearing_detection()
    reset_shared_variables()
    return jsonify({"status": "SUCCESS"}), 200

    

#停止多进程函数的写法
def stop_inference_internal():
    global processes
    if processes:  # 检查是否有子进程正在运行
        stop_event.set()  # 设置停止事件标志，通知所有子进程停止运行

        # 等待所有子进程结束
        for process in processes:
            if process.is_alive():
                process.join()  # 等待每个子进程结束
                print("焊接穿戴子进程运行结束")
        
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
    app.run(debug=False, host='172.16.20.163', port=5001)
