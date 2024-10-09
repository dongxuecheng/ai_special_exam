import time
import torch
import cv2
import threading
from datetime import datetime
from ultralytics import YOLO
from globals import stop_event,redis_client
from config import SAVE_IMG_PATH,POST_IMG_PATH3,PLATFORM_WEARING_MODEL,PLATFORM_WEARING_VIDEO_SOURCES



def init_wearing_detection():
    redis_client.set("platform_wearing_human_in_postion",'False')
    redis_client.delete("platform_wearing_items_nums")
    redis_client.delete("platform_wearing_detection_img")
    redis_client.set("platform_wearing_detection_img_flag",'False')

def start_wearing_detection(start_events):
        # Create threads for each video stream and model
    threads = []
    for model_path in PLATFORM_WEARING_MODEL:
        event = threading.Event()
        start_events.append(event)
        thread = threading.Thread(target=process_video, args=(model_path,PLATFORM_WEARING_VIDEO_SOURCES,event))
        threads.append(thread)
        thread.daemon=True
        thread.start()


    # Wait for all threads to complete
    for thread in threads:
        thread.join()

def process_video(model_path, video_source, start_event):

    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
    # Read a frame from the video
        success, frame = cap.read()
        
        if stop_event.is_set():#控制停止推理
            break

        if success:
            
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 10 != 0:#跳帧检测，
                continue

            x, y, w, h = 920, 0, 587, 1436#剪裁画面的中心区域

            # Crop the frame to the ROI
            frame = frame[y:y+h, x:x+w]
            # Run YOLOv8 inference on the frame
            if model_path==PLATFORM_WEARING_MODEL[0]:#yolov8n，专门用来检测人
                #model.classes = [0]#设置只检测人一个类别
                results = model.predict(frame,conf=0.6,verbose=False,classes=[0])#这里的results是一个生成器

                for r in results:

                    ##下面这些都是tensor类型
                    boxes = r.boxes.xyxy  # 提取所有检测到的边界框坐标
                    confidences = r.boxes.conf  # 提取所有检测到的置信度
                    classes = r.boxes.cls  # 提取所有检测到的类别索引

                    
                    for i in range(len(boxes)):
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]
                        
                        if label=="person" and redis_client.get("platform_wearing_human_in_postion")=='False':
                            redis_client.set("platform_wearing_human_in_postion",'True')

                    start_event.set()  


            if model_path==PLATFORM_WEARING_MODEL[1]:
                results = model.predict(frame,conf=0.6,verbose=False)
                for r in results:
                    boxes=r.obb.xyxyxyxy
                    confidences=r.obb.conf
                    classes=r.obb.cls
                
                    wearing_items={"belt" :0,
                            'helmet': 0,
                            'shoes': 0
                    }

                    for i in range(len(boxes)):
                        confidence = confidences[i].item()
                        cls = int(classes[i].item())
                        label = model.names[cls]

                        wearing_items[label] += 1

                    
                    #因为安全带检测有四个标签，当检测到两个及以上的时候，就认为有安全带
                    wearing_items["belt"] = 1 if wearing_items["belt"] > 2 else 0

                    wearing_items_nums = [wearing_items["belt"],  wearing_items["helmet"], wearing_items["shoes"]]
                    if redis_client.exists("platform_wearing_items_nums"):
                        redis_client.delete("platform_wearing_items_nums")
                    redis_client.rpush("platform_wearing_items_nums", *wearing_items_nums)


                    if redis_client.get("platform_wearing_detection_img_flag")=='True' and not redis_client.exists("platform_wearing_detection_img"):
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        imgpath = f"{SAVE_IMG_PATH}/platform_wearing_detection_{save_time}.jpg"
                        post_path= f"{POST_IMG_PATH3}/platform_wearing_detection_{save_time}.jpg"
                        annotated_frame = results[0].plot()
                        cv2.imwrite(imgpath, annotated_frame)
                        redis_client.set("platform_wearing_detection_img",post_path)


                    start_event.set()    


        else:
            # Break the loop if the end of the video is reached
            break

        # Release the video capture object and close the display window
    cap.release()    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model


