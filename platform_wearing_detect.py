#import time
import torch
import cv2
#import threading
from datetime import datetime
from ultralytics import YOLO
#from globals import stop_event,redis_client
from config import SAVE_IMG_PATH,POST_IMG_PATH3,PLATFORM_WEARING_MODEL,PLATFORM_WEARING_VIDEO_SOURCES
from utils.tool import IoU


# def init_wearing_detection():
#     redis_client.set("platform_wearing_human_in_postion",'False')
#     redis_client.delete("platform_wearing_items_nums")
#     redis_client.delete("platform_wearing_detection_img")
#     redis_client.set("platform_wearing_detection_img_flag",'False')

# def start_wearing_detection(start_events):
#         # Create threads for each video stream and model
#     threads = []
#     for model_path in PLATFORM_WEARING_MODEL:
#         event = threading.Event()
#         start_events.append(event)
#         thread = threading.Thread(target=process_video, args=(model_path,PLATFORM_WEARING_VIDEO_SOURCES,event))
#         threads.append(thread)
#         thread.daemon=True
#         thread.start()


#     # Wait for all threads to complete
#     for thread in threads:
#         thread.join()

def video_decoder(rtsp_url, frame_queue_list,start_event, stop_event):
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        if stop_event.is_set():#控制停止推理
            print("视频流关闭")
            break
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 != 0:
            continue
        frame_queue_list[0].put(frame)
        frame_queue_list[1].put(frame)

        start_event.set()  
    cap.release()

def process_video(model_path, video_source, start_event, stop_event,platform_wearing_human_in_postion, platform_wearing_items_nums, platform_wearing_detection_img_flag, platform_wearing_detection_img):

    
    model = YOLO(model_path)
    #cap = cv2.VideoCapture(video_source)
    while True:
    # Read a frame from the video
        if stop_event.is_set():
            print("穿戴子线程关闭")
            break
        
        if video_source.empty():
        # 队列为空，跳过处理
            continue
        frame = video_source.get()


    # while cap.isOpened():
    # # Read a frame from the video
    #     success, frame = cap.read()
        
    #     if stop_event.is_set():#控制停止推理
    #         break

    #     if success:
            
    #         if cap.get(cv2.CAP_PROP_POS_FRAMES) % 10 != 0:#跳帧检测，
    #             continue

        # x, y, w, h = 800, 0, 789, 1439#剪裁画面的中心区域

        # # Crop the frame to the ROI
        # frame = frame[y:y+h, x:x+w]
        # Run YOLOv8 inference on the frame
        if model_path==PLATFORM_WEARING_MODEL[0]:#yolov8n，专门用来检测人
            #model.classes = [0]#设置只检测人一个类别
            results = model.predict(frame,conf=0.6,verbose=False,classes=[0],device='0')#这里的results是一个生成器
            platform_wearing_human_in_postion.value = False
            for r in results:

                ##下面这些都是tensor类型
                boxes = r.boxes.xyxy  # 提取所有检测到的边界框坐标
                confidences = r.boxes.conf  # 提取所有检测到的置信度
                classes = r.boxes.cls  # 提取所有检测到的类别索引
                
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    confidence = confidences[i].item()
                    cls = int(classes[i].item())
                    label = model.names[cls]
                    
                    # if label=="person" and redis_client.get("platform_wearing_human_in_postion")=='False':
                    #     redis_client.set("platform_wearing_human_in_postion",'True')

                    if label=="person" and not platform_wearing_human_in_postion.value:
                        if IoU([x1,y1,x2,y2],[800, 0, 1589, 1439]) > 0:
                            platform_wearing_human_in_postion.value = True

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
                wearing_items["belt"] = 1 if wearing_items["belt"] >= 2 else 0

                # wearing_items_nums = [wearing_items["belt"],  wearing_items["helmet"], wearing_items["shoes"]]
                # if redis_client.exists("platform_wearing_items_nums"):
                #     redis_client.delete("platform_wearing_items_nums")
                # redis_client.rpush("platform_wearing_items_nums", *wearing_items_nums)
                if platform_wearing_human_in_postion.value and not platform_wearing_detection_img_flag.value:
                    platform_wearing_items_nums[0] = max(platform_wearing_items_nums[0],wearing_items["belt"])

                    platform_wearing_items_nums[1] = max(platform_wearing_items_nums[1],wearing_items["helmet"])
                    platform_wearing_items_nums[2] = max(platform_wearing_items_nums[2],wearing_items["shoes"])
                    #platform_wearing_detection_img_flag.value = True


                # if redis_client.get("platform_wearing_detection_img_flag")=='True' and not redis_client.exists("platform_wearing_detection_img"):
                if platform_wearing_detection_img_flag.value and 'wearing_img' not in platform_wearing_detection_img:
                    save_time=datetime.now().strftime('%Y%m%d_%H%M')
                    imgpath = f"{SAVE_IMG_PATH}/platform_wearing_detection_{save_time}.jpg"
                    post_path= f"{POST_IMG_PATH3}/platform_wearing_detection_{save_time}.jpg"
                    annotated_frame = results[0].plot()
                    cv2.imwrite(imgpath, annotated_frame)
                    print("保存图片")
                    platform_wearing_detection_img['wearing_img']=post_path


                start_event.set()    

  
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model


