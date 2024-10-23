import cv2
import torch
from shapely.geometry import box, Polygon

from datetime import datetime
from ultralytics import YOLO

from config import WELDING_MODEL_PATHS,WELDING_VIDEO_SOURCES
from config import SAVE_IMG_PATH,POST_IMG_PATH2,WELDING_REGION2,WELDING_REGION3

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
        if rtsp_url==WELDING_VIDEO_SOURCES[0]:
            frame_queue_list[0].put_nowait(frame)
        elif rtsp_url==WELDING_VIDEO_SOURCES[1]:
            frame_queue_list[1].put_nowait(frame)
        elif rtsp_url==WELDING_VIDEO_SOURCES[2]:
            frame_queue_list[2].put_nowait(frame)
        elif rtsp_url==WELDING_VIDEO_SOURCES[3]:
            frame_queue_list[3].put_nowait(frame)
        elif rtsp_url==WELDING_VIDEO_SOURCES[4]:
            frame_queue_list[4].put_nowait(frame)

        start_event.set()  
    cap.release()   

def save_image(welding_reset_imgs,results, step_name):

    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    imgpath = f"{SAVE_IMG_PATH}/{step_name}_{save_time}.jpg"
    postpath = f"{POST_IMG_PATH2}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()

    cv2.imwrite(imgpath, annotated_frame)
    welding_reset_imgs[step_name]=postpath

def process_video(model_path, video_source, start_event, stop_event,welding_reset_flag, welding_reset_imgs):
    # Load YOLO model
    model = YOLO(model_path)
    while True:      
        if stop_event.is_set():
            print("复位子线程关闭")
            break
        
        if video_source.empty():
        # 队列为空，跳过处理
            continue
        frame = video_source.get()
        #if video_source == WELDING_CH2_RTSP:#这两个视频流用的分类模型，因为分类模型预处理较慢，需要手动resize
        if model_path == WELDING_MODEL_PATHS[1]:
            frame=cv2.resize(frame,(640,640))

        # Run YOLOv8 inference on the frame
        results = model.predict(frame,verbose=False,conf=0.4)

        for r in results:


            if model_path == WELDING_MODEL_PATHS[1]:

                if r.probs.top1conf>0.6:
                    label=model.names[r.probs.top1]
                    
                    welding_reset_flag[3]=True if label == "component" else False
                else:
                    continue



            # if video_source == WELDING_CH5_RTSP or video_source == WELDING_CH3_RTSP or video_source == WELDING_CH1_RTSP or video_source == WELDING_CH4_RTSP:
            if model_path == WELDING_MODEL_PATHS[0] or model_path == WELDING_MODEL_PATHS[2] or model_path == WELDING_MODEL_PATHS[3] or model_path == WELDING_MODEL_PATHS[4]:
                ##下面这些都是tensor类型
                boxes = r.boxes.xyxy  # 提取所有检测到的边界框坐标
                confidences = r.boxes.conf  # 提取所有检测到的置信度
                classes = r.boxes.cls  # 提取所有检测到的类别索引


                if model_path == WELDING_MODEL_PATHS[4]:
                    welding_reset_flag[2]=False
                    #welding_components_flag=False

                    
                #if video_source==WELDING_CH3_RTSP:#当画面没有油桶时，给个初值为安全
                if model_path == WELDING_MODEL_PATHS[2]:
                    welding_reset_flag[0]=True

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
                            welding_reset_flag[0]=False #表示油桶在危险区域
                        else:
                            welding_reset_flag[0]=True 

                    if label=='turnon':
                        welding_reset_flag[1]=True
                    if label=='turnoff':
                        welding_reset_flag[1]=False

                    if label== "open":#检测焊机开关
                        welding_reset_flag[4] = True



                    if label=="close":#检测焊机开关
                        welding_reset_flag[4] = False


                    if label=="grounding_wire" :
                        rect_shapely = box(x1,y1, x2, y2)#使用shapely库创建的矩形
                        WELDING_REGION3_shapely = Polygon(WELDING_REGION3.tolist()) #shapely计算矩形检测框和多边形的iou使用
                        intersection = rect_shapely.intersection(WELDING_REGION3_shapely)
                                # 计算并集
                        union = rect_shapely.union(WELDING_REGION3_shapely)
                                # 计算 IoU
                        iou = intersection.area / union.area


                        if iou>0 :
                            welding_reset_flag[2]=True #表示搭铁线连接在焊台上
                        else:
                            welding_reset_flag[2]=False #表示未连接上


            if model_path == WELDING_MODEL_PATHS[3]:
                if welding_reset_flag[1] and 'reset_step_2' not in welding_reset_imgs:
                    print("当前总开关没有复位")
                    save_image(welding_reset_imgs,results, "reset_step_2")

            if model_path == WELDING_MODEL_PATHS[2]:
                if welding_reset_flag[0] and 'reset_step_1' not in welding_reset_imgs:
                    print("当前油桶没有复位")
                    save_image(welding_reset_imgs,results, "reset_step_1")

            if model_path == WELDING_MODEL_PATHS[0]:
                if welding_reset_flag[4] and 'reset_step_5' not in welding_reset_imgs:
                    print("当前焊机开关没有复位")
                    save_image(welding_reset_imgs,results, "reset_step_5")

            if model_path == WELDING_MODEL_PATHS[4]:
                if welding_reset_flag[2] and 'reset_step_3' not in welding_reset_imgs:
                    print("搭铁线没有复位")
                    save_image(welding_reset_imgs,results, "reset_step_3")

            if model_path == WELDING_MODEL_PATHS[1]:
                if welding_reset_flag[3] and 'reset_step_4' not in welding_reset_imgs:
                    print("当前焊件没有复位")
                    save_image(welding_reset_imgs,results, "reset_step_4")

            #运行到这里表示一个线程检测完毕
            start_event.set()
            
    # 释放模型资源（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



