import re
import threading
import time
from flask import Flask, jsonify,send_from_directory

from basket_cleaning_detect import start_basket_cleaning_detection
from platform_remove_detect import start_platform_remove_detection,init_platform_remove_detection
from globals import inference_thread, stop_event,redis_client


app = Flask(__name__)


# Define the /wearing_detection endpoint

@app.route('/basket_cleaning_detection', methods=['GET'])
def basket_cleaning_detection():#开启平台搭设检测
    global inference_thread#当全局变量需要重新赋值时，需要用global关键字声明

    if inference_thread is None or not inference_thread.is_alive():#防止重复开启检测服务
        #redis_client.set("log_in_flag",'False')

        stop_event.clear()

        start_events = []#给每个线程一个事件，让我知道某个线程是否开始检测
        inference_thread = threading.Thread(target=start_basket_cleaning_detection,args=(start_events,))
        inference_thread.start()
        
            
        app.logger.info('start_platform_setup_detection')
        #init_platform_setup_detection()


        # 等待所有YOLO线程开始检测
        for event in start_events:
            event.wait()

        return jsonify({"status": "SUCCESS"}), 200
    
    else:
        app.logger.info("reset_detection already running")   
        return jsonify({"status": "ALREADY_RUNNING"}), 200    

# @app.route('/basket_cleaning_status', methods=['GET']) 
# def basket_cleaning_status():#获取平台搭设状态状态
#     if not redis_client.exists('platform_setup_order'):#平台搭设步骤还没有一个完成
#         app.logger.info('platform_setup_order is none')

#         return jsonify({"status": "NONE"}), 200
    
#     else:

#         platform_setup_order = redis_client.lrange("platform_setup_order", 0, -1)

#         json_array = []
#         for value in platform_setup_order:
#             match = re.search(r'platform_setup_(\d+)', value)
#             step_number = match.group(1)
#             json_object = {"step": step_number, "image": redis_client.get(f"platform_setup_{step_number}_img"),'number':redis_client.get(f"platform_setup_{step_number}")}
#             json_array.append(json_object) 

#         return jsonify({"status": "SUCCESS","data":json_array}), 200


# @app.route('/basket_cleaning_finish', methods=['GET']) 
# def basket_cleaning_finish():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测

#     stop_inference_internal()
#     app.logger.info('platform_setup_finish')
#     return jsonify({"status": "SUCCESS"}), 200

            

# @app.route('/platform_remove_detection', methods=['GET']) 
# def platform_remove_detection():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
#     global inference_thread#当全局变量需要重新赋值时，需要用global关键字声明

#     if inference_thread is None or not inference_thread.is_alive():#防止重复开启检测服务
#         #redis_client.set("log_in_flag",'False')

#         stop_event.clear()

#         start_events = []#给每个线程一个事件，让我知道某个线程是否开始检测
#         inference_thread = threading.Thread(target=start_platform_remove_detection,args=(start_events,))
#         inference_thread.start()
        
            
#         app.logger.info('start_platform_remove_detection')
#         init_platform_remove_detection()


#         # 等待所有YOLO线程开始检测
#         for event in start_events:
#             event.wait()

#         return jsonify({"status": "SUCCESS"}), 200
    
#     else:
#         app.logger.info("reset_detection already running")   
#         return jsonify({"status": "ALREADY_RUNNING"}), 200    
    
# @app.route('/platform_remove_status', methods=['GET']) 
# def platform_remove_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
#     if not redis_client.exists('platform_remove_order'):#平台搭设步骤还没有一个完成
#         app.logger.info('platform_remove_order is none')

#         return jsonify({"status": "NONE"}), 200
    
#     else:

#         platform_setup_order = redis_client.lrange("platform_remove_order", 0, -1)

#         json_array = []
#         for num in platform_setup_order:

#             json_object = {"step": num, "image": redis_client.get(f"platform_remove_{num}_img")}
#             json_array.append(json_object) 

#         return jsonify({"status": "SUCCESS","data":json_array}), 200

# @app.route('/platform_remove_finish', methods=['GET'])#点击考试结束按钮，停止检测，并复位
# def platform_remove_finish():
#     stop_inference_internal()
#     app.logger.info('platform_remove_finish')
#     return jsonify({"status": "SUCCESS"}), 200


    
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
    return send_from_directory('static/images', filename)


if __name__ == '__main__':

    # Start the Flask server
    app.run(debug=False, host='172.16.20.163', port=5005)
