import torch
import cv2
from datetime import datetime
from ultralytics import YOLO
from config import SAVE_IMG_PATH,POST_IMG_PATH1,WELDING_WEARING_MODEL,WELDING_WEARING_VIDEO_SOURCES


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
        frame_queue_list[0].put_nowait(frame)
        frame_queue_list[1].put_nowait(frame)

        start_event.set()  
    cap.release()

def process_video(model_path, video_source, start_event, stop_event,welding_wearing_human_in_postion, welding_wearing_items_nums, welding_wearing_detection_img_flag, welding_wearing_detection_img):

    
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

        x, y, w, h = 480, 0, 733, 1082#剪裁画面的中心区域

        # Crop the frame to the ROI
        cropped_frame = frame[y:y+h, x:x+w]
        # Run YOLOv8 inference on the frame
        if model_path==WELDING_WEARING_MODEL[0]:#yolov8s，专门用来检测人
            #model.classes = [0]#设置只检测人一个类别
            results = model.predict(cropped_frame,conf=0.6,verbose=False,classes=[0])#这里的results是一个生成器
        else:
            results = model.predict(cropped_frame,conf=0.6,verbose=False)
        #while not stop_event.is_set():

        for r in results:

            ##下面这些都是tensor类型
            boxes = r.boxes.xyxy  # 提取所有检测到的边界框坐标
            confidences = r.boxes.conf  # 提取所有检测到的置信度
            classes = r.boxes.cls  # 提取所有检测到的类别索引
            ###劳保,不在函数外部定义是因为需要每一帧重新赋值
            wearing_items={"pants" :0,
                    'jacket': 0,
                    'helmet': 0,
                    'gloves': 0,
                    'shoes': 0
            }

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                confidence = confidences[i].item()
                cls = int(classes[i].item())
                label = model.names[cls]

                
                # if x1 < WEAR_DETECTION_AREA[0] or y1 < WEAR_DETECTION_AREA[1] or x2 > WEAR_DETECTION_AREA[2] or y2 > WEAR_DETECTION_AREA[3]:
                #     continue  # 跳过不在区域内的检测框
                
                if model_path==WELDING_WEARING_MODEL[0]:#yolov8s，专门用来检测人
                    # if label=="person" and redis_client.get("welding_wearing_human_in_postion")=='False':
                    #     redis_client.set("welding_wearing_human_in_postion",'True')
                    if label=="person" and not welding_wearing_human_in_postion.value:
                        welding_wearing_human_in_postion.value=True
                else:
                    wearing_items[label] += 1


            if model_path==WELDING_WEARING_MODEL[1]:

                if welding_wearing_human_in_postion.value and not welding_wearing_detection_img_flag.value:
                    welding_wearing_items_nums[0] = wearing_items["pants"]
                    welding_wearing_items_nums[1] = wearing_items["jacket"]
                    welding_wearing_items_nums[2] = wearing_items["helmet"]
                    welding_wearing_items_nums[3] = wearing_items["gloves"]
                    welding_wearing_items_nums[4] = wearing_items["shoes"]

                if welding_wearing_detection_img_flag.value and 'wearing_img' not in welding_wearing_detection_img:
                    save_time=datetime.now().strftime('%Y%m%d_%H%M')
                    imgpath = f"{SAVE_IMG_PATH}/welding_wearing_detection_{save_time}.jpg"
                    post_path= f"{POST_IMG_PATH1}/welding_wearing_detection_{save_time}.jpg"
                    annotated_frame = results[0].plot()
                    cv2.imwrite(imgpath, annotated_frame)
                    welding_wearing_detection_img['wearing_img']=post_path
                    #welding_wearing_detection_img_flag.value=False

            start_event.set()    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model


