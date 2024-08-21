import re
import threading
import time
from flask import Flask, jsonify,send_from_directory

from welding_exam_detect import start_welding_detection,init_welding_detection
from welding_reset_detect import start_reset_detection,init_rest_detection
from globals import inference_thread, stop_event,lock,redis_client


app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Define the /wearing_detection endpoint

@app.route('/reset_detection', methods=['GET'])
def reset_detection():#发送开启AI服务时，检测复位
    global inference_thread#当全局变量需要重新赋值时，需要用global关键字声明

    if inference_thread is None or not inference_thread.is_alive():#防止重复开启检测服务
        #redis_client.set("log_in_flag",'False')

        stop_event.clear()

        start_events = []#给每个线程一个事件，让我知道某个线程是否开始检测
        inference_thread = threading.Thread(target=start_reset_detection,args=(start_events,))
        inference_thread.start()
        
            
        app.logger.info('start_reset_detection')
    
        init_rest_detection()
        #init_rest()#设置复位检测图片保存标志为False
        #redis_client.set("log_in_flag",'True')#设置登录标志为True,进入保存图片阶段
        #time.sleep(8)#等待3s，等待reset_post_path列表中有数据,然后返回给前端

        # 等待所有YOLO线程开始检测
        for event in start_events:
            event.wait()

        return jsonify({"status": "SUCCESS"}), 200
    
    else:
        app.logger.info("reset_detection already running")   
        return jsonify({"status": "ALREADY_RUNNING"}), 200    

@app.route('/reset_status', methods=['GET']) 
def reset_status():#获取复位检测状态
    if redis_client.get("welding_reset_flag")=='0':#表明不需要复位,welding_reset_flag是要复位的个数
        app.logger.info('reset_all is true')
        #此时复位的检测还在进行，需要停止复位检测
        stop_inference_internal()

        return jsonify({"status": "RESET_ALL"}), 200
    
    if redis_client.get("welding_reset_flag")>'0':#表面需要复位，并设置log_in_flag为True
        app.logger.info('reset_all is false')

        #发送需要复位的信息
        reset_post_path = redis_client.lrange("welding_reset_post_path", 0, -1)

        json_array = []
        for value in reset_post_path:
            
            match = re.search(r'resetStep(\d+)', value)
            step_number = match.group(1)
            json_object = {"resetStep": step_number, "image": value}
            json_array.append(json_object)

        init_rest_detection()
        app.logger.info(reset_post_path)
        return jsonify({"status": "NOT_RESET_ALL","data":json_array}), 200


@app.route('/welding_detection', methods=['GET']) 
def welding_detection():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
    global inference_thread

    if inference_thread is None or not inference_thread.is_alive():#防止重复开启检测服务

        stop_event.clear()#stop_event不用global声明，因为不需要重新赋值，他只是调用了其方法，并没有重新赋值

        start_events = []#给每个线程一个事件，让我知道某个线程是否开始检测
        inference_thread = threading.Thread(target=start_welding_detection,args=(start_events,))
        inference_thread.start()

        init_welding_detection()
        
        #等待所有YOLO线程开始检测
        for event in start_events:
            event.wait()

        return jsonify({"status": "SUCCESS"}), 200
    else:
        app.logger.info("welding_detection already running")   
        return jsonify({"status": "ALREADY_RUNNING"}), 200

            

@app.route('/welding_status', methods=['GET']) 
def welding_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
    #global inference_thread
    #with lock:   
    #TODO 若出现异常再发送FAIL.
    if redis_client.llen("welding_post_path") == 0:
        return jsonify({"status": "NONE"}), 200##表示还没有检测到任何一个焊接步骤
    
    welding_post_path = redis_client.lrange("welding_post_path", 0, -1)

    json_array = []
    for value in welding_post_path:
        match = re.search(r'step(\d+)', value)
        step_number = match.group(1)
        json_object = {"step": step_number, "image": value}
        json_array.append(json_object)

    #init_rest()
    app.logger.info(welding_post_path)
    return jsonify({"status": "SUCCESS","data":json_array}), 200

@app.route('/end_welding_exam', methods=['GET'])#点击考试结束按钮，停止检测，并复位
def end_welding_exam():
    stop_inference_internal()
    time.sleep(1)
    return reset_detection()
               

# def return_post_path():
#     app.logger.info("List elements:", redis_client.lrange("reset_post_path", 0, -1))
    
def stop_inference_internal():
    global inference_thread
    if inference_thread is not None and inference_thread.is_alive():
        stop_event.set()  # 设置停止事件标志，通知推理线程停止运行
        inference_thread.join()  # 等待推理线程结束
        inference_thread = None  # 释放线程资源
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
