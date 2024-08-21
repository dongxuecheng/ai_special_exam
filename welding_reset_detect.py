import cv2
import torch
import threading
from shapely.geometry import box, Polygon

from datetime import datetime
from ultralytics import YOLO

from config import WELDING_MODEL_PATHS,WELDING_VIDEO_SOURCES
from config import WELDING_CH1_RTSP,WELDING_CH2_RTSP,WELDING_CH3_RTSP,WELDING_CH4_RTSP,WELDING_CH5_RTSP
from config import SAVE_IMG_PATH,POST_IMG_PATH2,WELDING_REGION2,WELDING_REGION3
from globals import oil_barrel_flag,main_switch_flag,ground_wire_flag,welding_components_flag,welding_machine_switch_flag
from globals import lock,redis_client,stop_event


def init_rest_detection():
    redis_client.delete("welding_reset_post_path")#删除该列表welding_reset_post_path
    
    redis_client.set("welding_main_switch_save_img",'False')
    redis_client.set("welding_oil_barrel_save_img",'False')
    redis_client.set("welding_ground_wire_save_img",'False')
    redis_client.set("welding_components_save_img",'False')
    redis_client.set("welding_machine_switch_save_img",'False')


def start_reset_detection(start_events):
        # Create threads for each video stream and model
    threads = []
    for model_path, video_source in zip(WELDING_MODEL_PATHS, WELDING_VIDEO_SOURCES):
        event = threading.Event()
        start_events.append(event)
        thread = threading.Thread(target=process_video, args=(model_path, video_source,event))
        threads.append(thread)
        thread.daemon=True
        thread.start()


    # Wait for all threads to complete
    for thread in threads:
        thread.join()
        print("复位检测子线程结束")

# Function to process video with YOLO model
def process_video(model_path, video_source, start_event):
    # Load YOLO model
    model = YOLO(model_path)
    #results = model.predict(video_source,stream=True,verbose=False,conf=0.4,device='0')#这里的results是一个生成器
    cap = cv2.VideoCapture(video_source)
    # Loop through the video frames
    while cap.isOpened():
        if stop_event.is_set():#控制停止推理
            break
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 != 0:#跳帧检测，
                continue

            if video_source == WELDING_CH2_RTSP or video_source == WELDING_CH4_RTSP:#这两个视频流用的分类模型，因为分类模型预处理较慢，需要手动resize
                frame=cv2.resize(frame,(640,640))

            # Run YOLOv8 inference on the frame
            results = model.predict(frame,verbose=False,conf=0.4)

            global oil_barrel_flag,main_switch_flag,ground_wire_flag,welding_components_flag,welding_machine_switch_flag

    #with lock:
            for r in results:


                #因为搭铁线不好识别，所以当没有检测到时，ground_wire_flag=False
                if video_source == WELDING_CH2_RTSP:
                    # annotated_frame = results[0].plot()
                    # cv2.namedWindow('main_switch', cv2.WINDOW_NORMAL)
                    # cv2.imshow('main_switch', annotated_frame)
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #     break  

                    if r.probs.top1conf>0.6:
                        label=model.names[r.probs.top1]
                        
                        welding_components_flag=True if label == "component" else False
                    else:
                        continue

                if video_source == WELDING_CH4_RTSP:
                    if r.probs.top1conf>0.6:
                        label=model.names[r.probs.top1]#获取最大概率的类别的label
                        
                        main_switch_flag = True if label == "open" else False
                    else:
                        continue

                if video_source == WELDING_CH5_RTSP or video_source == WELDING_CH3_RTSP or video_source == WELDING_CH1_RTSP:
                    ##下面这些都是tensor类型
                    boxes = r.boxes.xyxy  # 提取所有检测到的边界框坐标
                    confidences = r.boxes.conf  # 提取所有检测到的置信度
                    classes = r.boxes.cls  # 提取所有检测到的类别索引

                    if video_source == WELDING_CH5_RTSP:
                        ground_wire_flag=False
                        #welding_components_flag=False
                    if video_source == WELDING_CH3_RTSP:
                        oil_barrel_flag=True

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
                            #print(is_inside)
                            
                            if is_inside>=0 :
                                oil_barrel_flag=False #表示油桶在危险区域
                            else:
                                oil_barrel_flag=True 

                        
                        if label== "open":#检测焊机开关
                            welding_machine_switch_flag = True



                        if label=="close":#检测焊机开关
                            welding_machine_switch_flag = False


                        if label=="grounding_wire" :
                            rect_shapely = box(x1,y1, x2, y2)#使用shapely库创建的矩形
                            WELDING_REGION3_shapely = Polygon(WELDING_REGION3.tolist()) #shapely计算矩形检测框和多边形的iou使用
                            intersection = rect_shapely.intersection(WELDING_REGION3_shapely)
                                    # 计算并集
                            union = rect_shapely.union(WELDING_REGION3_shapely)
                                    # 计算 IoU
                            iou = intersection.area / union.area


                            if iou>0 :
                                ground_wire_flag=True #表示搭铁线连接在焊台上
                            else:
                                ground_wire_flag=False #表示未连接上

                flag_count = sum([oil_barrel_flag, main_switch_flag, ground_wire_flag, welding_components_flag, welding_machine_switch_flag])
                redis_client.set("welding_reset_flag",flag_count)
                #print("主开关",main_switch_flag)

                if video_source == WELDING_CH4_RTSP :#检测到总开关
                        if main_switch_flag and redis_client.get("welding_main_switch_save_img")=='False':
                            print("当前总开关没有复位")##焊前检查只保存一次
                            redis_client.set("welding_main_switch_save_img",'True')
                            #save_time=datetime.now().strftime('%Y%m%d_%H%M')
                            save_time=datetime.now().strftime('%Y%m%d_%H')
                            imgpath = f"{SAVE_IMG_PATH}/welding_resetStep2_{save_time}.jpg"
                            post_path = f"{POST_IMG_PATH2}/welding_resetStep2_{save_time}.jpg"
                            redis_client.rpush("welding_reset_post_path",post_path)#welding_reset_post_path为一个列表，存储需要发送的图片路径，rpush为从右侧加入
                            annotated_frame = results[0].plot()
                            cv2.imwrite(imgpath, annotated_frame)



                                
                if video_source == WELDING_CH3_RTSP :##检测油桶{0: 'dump'},在安全区域时保存图片一张
                    if oil_barrel_flag and redis_client.get("welding_oil_barrel_save_img")=='False':
                        print("当前油桶没有复位")
                        redis_client.set("welding_oil_barrel_save_img",'True')
                        #save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        save_time=datetime.now().strftime('%Y%m%d_%H')
                        imgpath = f"{SAVE_IMG_PATH}/welding_resetStep1_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/welding_resetStep1_{save_time}.jpg"
                        redis_client.rpush("welding_reset_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(imgpath, annotated_frame)
                        #reset_post(welding_resetStep='1',path=post_path)
                        #post(step='2',path=post_path)

                if video_source == WELDING_CH1_RTSP:#检测到焊机开关
                        if welding_machine_switch_flag and redis_client.get("welding_machine_switch_save_img")=='False':
                            print("当前焊机开关没有复位")##焊前检查只保存一次
                            redis_client.set("welding_machine_switch_save_img",'True')
                            #save_time=datetime.now().strftime('%Y%m%d_%H%M')
                            save_time=datetime.now().strftime('%Y%m%d_%H')
                            imgpath = f"{SAVE_IMG_PATH}/welding_resetStep5_{save_time}.jpg"
                            post_path = f"{POST_IMG_PATH2}/welding_resetStep5_{save_time}.jpg"
                            redis_client.rpush("welding_reset_post_path",post_path)
                            #cv2.imwrite(imgpath, annotator.result())
                            annotated_frame = results[0].plot()
                            cv2.imwrite(imgpath, annotated_frame)
                            #reset_post(welding_resetStep='5',path=post_path)
                            #post(step='2',path=post_path)
            

                if video_source == WELDING_CH5_RTSP:
                    if ground_wire_flag and redis_client.get("welding_ground_wire_save_img")=='False':
                        print("搭铁线没有复位")
                        redis_client.set("welding_ground_wire_save_img",'True')
                        #save_time=datetime.now().strftime('%Y%m%d_%H%M')  
                        save_time=datetime.now().strftime('%Y%m%d_%H')
                        imgpath = f"{SAVE_IMG_PATH}/welding_resetStep3_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/welding_resetStep3_{save_time}.jpg"
                        redis_client.rpush("welding_reset_post_path",post_path)
                        # result_image = annotator.result()
                        # cv2.polylines(result_image, [WELDING_REGION3.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
                        # cv2.imwrite(imgpath, result_image)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(imgpath, annotated_frame)
                        #reset_post(welding_resetStep="3",path=post_path)
                        #time.sleep(1)
                        #post(step='4',path=post_path)

                if video_source == WELDING_CH2_RTSP:
                    if welding_components_flag and redis_client.get("welding_components_save_img")=='False':
                        print("焊件没有复位")
                        redis_client.set("welding_components_save_img",'True')
                        #save_time=datetime.now().strftime('%Y%m%d_%H%M')  
                        save_time=datetime.now().strftime('%Y%m%d_%H')
                        imgpath = f"{SAVE_IMG_PATH}/welding_resetStep4_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/welding_resetStep4_{save_time}.jpg"
                        redis_client.rpush("welding_reset_post_path",post_path)
                        # result_image = annotator.result()
                        # cv2.polylines(result_image, [REGION4.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=4)
                        # cv2.imwrite(imgpath, result_image)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(imgpath, annotated_frame)

                        #reset_post(welding_resetStep='4',path=post_path)
                        #post(step='4',path=post_path)

                #运行到这里表示一个线程检测完毕
                start_event.set()
            


        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    # 释放模型资源（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model
    #cv2.destroyAllWindows()


