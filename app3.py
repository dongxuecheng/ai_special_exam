import threading
import time
from flask import Flask, jsonify,send_from_directory
from platform_wearing_detect import start_wearing_detection,init_wearing_detection
from globals import inference_thread, stop_event,lock,redis_client

#吊篮吊具的穿戴检测
app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Define the /wearing_detection endpoint
@app.route('/wearing_detection', methods=['GET'])
def wearing_detection():
    global inference_thread#当全局变量需要重新赋值时，需要用global关键字声明

    if inference_thread is None or not inference_thread.is_alive():
        stop_event.clear()#stop_event不用global声明，因为不需要重新赋值，他只是调用了其方法，并没有重新赋值
        
        start_events = []#给每个线程一个事件，让我知道某个线程是否开始检测
        inference_thread = threading.Thread(target=start_wearing_detection,args=(start_events,))
        inference_thread.start()
        init_wearing_detection()
        #print('start_wearing_detection')
        # 等待所有YOLO线程开始检测，两个线程检测完毕时，才返回SUCCESS
        for event in start_events:
            event.wait()

        app.logger.info('start_wearing_detection')
        return jsonify({"status": "SUCCESS"}), 200
    
    else:
        app.logger.info("start_wearing_detection already running")   
        return jsonify({"status": "ALREADY_RUNNING"}), 200
    

@app.route('/human_postion_status', methods=['GET']) 
def human_postion_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
    #global inference_thread
    if redis_client.get("platform_wearing_human_in_postion")=='False':
        app.logger.info('NOT_IN_POSTION')
        return jsonify({"status": "NOT_IN_POSTION"}), 200
    else:
        app.logger.info('IN_POSTION')
        return jsonify({"status": "IN_POSTION"}), 200

@app.route('/wearing_status', methods=['GET']) 
def wearing_status():#开始登录时，检测是否需要复位，若需要，则发送复位信息，否则开始焊接检测
    #global inference_thread
    with lock:   
        #TODO 若出现异常再发送FAIL.
        redis_client.set("platform_wearing_detection_img_flag",'True')
        time.sleep(1)
        if not redis_client.exists("platform_wearing_items_nums") or not redis_client.exists("platform_wearing_detection_img"):
            return jsonify({"status": "NONE"}), 200##表示穿戴检测线程还未检测完
        
        wearing_items_nums = redis_client.lrange("platform_wearing_items_nums", 0, -1)
        wearing_items_list = ['belt', 'helmet',  'shoes']
        json_array = []
        for num, item in zip(wearing_items_nums, wearing_items_list):
            json_object = {"name": item, "number": num}
            json_array.append(json_object)

        app.logger.info(json_array)
        image=redis_client.get("platform_wearing_detection_img")
        app.logger.info(image)

        return jsonify({"status": "SUCCESS","data":json_array,"image":image}), 200

               
@app.route('/end_wearing_exam', methods=['GET'])
def end_wearing_exam():
    init_wearing_detection()
    return jsonify({"status": "SUCCESS"}), 200

    

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
    app.run(debug=False, host='172.16.20.163', port=5003)
