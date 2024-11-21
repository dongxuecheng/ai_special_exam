import cv2
import torch
from shapely.geometry import box, Polygon
from datetime import datetime
from ultralytics import YOLO

from config import WELDING_MODEL_PATHS,WELDING_VIDEO_SOURCES
from utils.tool import IoU
from config import WELDING_CH1_RTSP,WELDING_CH2_RTSP,WELDING_CH3_RTSP,WELDING_CH4_RTSP,WELDING_CH5_RTSP
from config import  SAVE_IMG_PATH,POST_IMG_PATH2,WELDING_REGION1,WELDING_REGION2,WELDING_REGION3
from globals import oil_barrel,main_switch,grounding_wire,welding_machine_switch,welding_components,mask,welding,gloves,sweep,sweep_detect_num,welding_detect_num



def save_image(welding_exam_imgs,results, step_name,welding_exam_order):
    save_time = datetime.now().strftime('%Y%m%d_%H%M')
    imgpath = f"{SAVE_IMG_PATH}/{step_name}_{save_time}.jpg"
    postpath = f"{POST_IMG_PATH2}/{step_name}_{save_time}.jpg"
    annotated_frame = results[0].plot()
    cv2.imwrite(imgpath, annotated_frame)
    welding_exam_imgs[step_name]=postpath
    welding_exam_order.append(step_name)
    print(f"{step_name}完成")

# Function to process video with YOLO model
def process_video(model_path, video_source,start_event,stop_event,welding_exam_flag, welding_exam_imgs,welding_exam_order):
    # Load YOLO model
    model = YOLO(model_path)
    while True:
        
        if stop_event.is_set():
            print("考核子线程关闭")
            break
        
        if video_source.empty():
        # 队列为空，跳过处理
            #print("队列为空")
            continue
        
        frame = video_source.get()    

        #if video_source == WELDING_CH2_RTSP:#这两个视频流用的分类模型，因为分类模型预处理较慢，需要手动resize
        # if model_path==WELDING_MODEL_PATHS[1]:
        #     frame=cv2.resize(frame,(214,214))
        
        results = model.predict(frame,verbose=False,conf=0.4,device='1')
        # start_event.set()
        # print("焊接考核子线程运行中f{model_path}")   
        #global steps
        global oil_barrel,main_switch,grounding_wire,welding_machine_switch,welding_components,mask,welding,gloves,sweep
        global sweep_detect_num,welding_detect_num

        for r in results:

            #if video_source == WELDING_CH2_RTSP:#焊台
            if model_path == WELDING_MODEL_PATHS[1]:
                if r.probs.top1conf>0.8:
                    label=model.names[r.probs.top1]
                    if label=='component':
                        welding_components='in_position'#在焊台上
                    if label=='empty':
                        welding_components='not_in_position'#不在焊台上
                    if label=='welding':
                        if welding_detect_num<3:
                            welding_detect_num+=1
                        else:
                            welding_exam_flag[6]=True#表示有焊接
                    if label=='sweep' and welding_exam_flag[6]==True:
                        if sweep_detect_num<3:
                            sweep_detect_num+=1
                        else:
                            welding_exam_flag[11]=True#表示有打扫
                else:
                    continue


            #if video_source == WELDING_CH3_RTSP:#油桶
            # if video_source == WELDING_CH4_RTSP:#总开关
            #     if r.probs.top1conf>0.8:
            #         label=model.names[r.probs.top1]#获取最大概率的类别的label
            #         main_switch = "open" if label == "open" else "close"
            #     else:
            #         continue   


            
            #if video_source == WELDING_CH1_RTSP or video_source==WELDING_CH3_RTSP or video_source==WELDING_CH5_RTSP or video_source==WELDING_CH4_RTSP:#焊接操作，进行目标检测
            if model_path == WELDING_MODEL_PATHS[0] or model_path == WELDING_MODEL_PATHS[2] or model_path == WELDING_MODEL_PATHS[4] or model_path == WELDING_MODEL_PATHS[3]:
                ##下面这些都是tensor类型
                boxes = r.boxes.xyxy  # 提取所有检测到的边界框坐标
                confidences = r.boxes.conf  # 提取所有检测到的置信度
                classes = r.boxes.cls  # 提取所有检测到的类别索引


                # if video_source==WELDING_CH5_RTSP:
                #     grounding_wire=="disconnect"##单独每次设置为false，是为了防止没有检测到
                    #welding_components=False

                #if video_source==WELDING_CH3_RTSP:#当画面没有油桶时，给个初值为安全
                if model_path == WELDING_MODEL_PATHS[2]:
                    oil_barrel="safe"

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
                            oil_barrel='danger' #表示油桶在危险区域
                        else:
                            oil_barrel='safe' 

                    if label== "open" or "close":#检测焊机开关
                        if confidence>0.8:
                            welding_machine_switch = label

                    if label=="turnon" and confidence>0.8:
                        main_switch="open"
                    if label=="turnoff" and confidence>0.8:
                        main_switch="close"

                    if label=="grounding_wire" :
                        if confidence<0.6:
                            continue
                        rect_shapely = box(x1,y1, x2, y2)#使用shapely库创建的矩形
                        WELDING_REGION3_shapely = Polygon(WELDING_REGION3.tolist()) #shapely计算矩形检测框和多边形的iou使用
                        intersection = rect_shapely.intersection(WELDING_REGION3_shapely)
                        # 计算并集
                        union = rect_shapely.union(WELDING_REGION3_shapely)
                        # 计算 IoU
                        iou = intersection.area / union.area

                        grounding_wire="connect" if iou>0 else "disconnect" #表示搭铁线连接在焊台上    


                    
                    if label=="mask":
                        #mask=True #表示戴面罩
                        iou=IoU(boxes[i].tolist(),WELDING_REGION1)
                        welding_exam_flag[5]=True if iou>0 else False #表示戴面罩


                    if label=="gloves":
                        if confidence>0.5:
                            welding_exam_flag[7]=True#表示戴手套

            if model_path == WELDING_MODEL_PATHS[1]:
                if not start_event.is_set():
                    start_event.set()
                    print(f"焊接考核子线程运行中{model_path}")

                if welding_exam_flag[11]==True and 'welding_exam_12' not in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_12",welding_exam_order)

                if welding_components=='in_position' and 'welding_exam_4' not in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_4",welding_exam_order)
            
                if welding_components=='not_in_position' and 'welding_exam_11' not in welding_exam_imgs and 'welding_exam_4' in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_11",welding_exam_order)
                
                if welding_exam_flag[6]==True and 'welding_exam_7' not in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_7",welding_exam_order)

            if model_path == WELDING_MODEL_PATHS[2]:
                if not start_event.is_set():
                    start_event.set()
                    print(f"焊接考核子线程运行中{model_path}")
                if oil_barrel=="safe" and 'welding_exam_1' not in welding_exam_imgs:#排除危险源
                    save_image(welding_exam_imgs,results,"welding_exam_1",welding_exam_order)
                
            if model_path == WELDING_MODEL_PATHS[3]:
                if not start_event.is_set():
                    start_event.set()
                    print(f"焊接考核子线程运行中{model_path}")
                if main_switch=="open" and 'welding_exam_2' not in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_2",welding_exam_order)

                if main_switch=="close" and 'welding_exam_13' not in welding_exam_imgs and 'welding_exam_2' in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_13",welding_exam_order)
                
            if model_path == WELDING_MODEL_PATHS[0]:
                if not start_event.is_set():
                    start_event.set()
                    print(f"焊接考核子线程运行中{model_path}")

                if welding_machine_switch=="open" and 'welding_exam_5' not in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_5",welding_exam_order)
                

                if welding_machine_switch=="close" and 'welding_exam_9' not in welding_exam_imgs and 'welding_exam_5' in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_9",welding_exam_order)
                


            if model_path == WELDING_MODEL_PATHS[4]:
                if not start_event.is_set():
                    start_event.set()
                    print(f"焊接考核子线程运行中{model_path}")

                if welding_exam_flag[7]==True and 'welding_exam_8' not in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_8",welding_exam_order)
      
                if grounding_wire=="connect" and 'welding_exam_3' not in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_3",welding_exam_order)
                        
                if grounding_wire=="disconnect" and 'welding_exam_10' not in welding_exam_imgs and 'welding_exam_3' in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_10",welding_exam_order)
                
                if welding_exam_flag[5]==True and 'welding_exam_6' not in welding_exam_imgs:
                    save_image(welding_exam_imgs,results,"welding_exam_6",welding_exam_order)
        
                   

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



        

    