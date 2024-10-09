import cv2
import torch
from shapely.geometry import box, Polygon
import threading
from datetime import datetime
from ultralytics import YOLO

from config import WELDING_MODEL_PATHS,WELDING_VIDEO_SOURCES
from utils.tool import IoU
from globals import stop_event,redis_client,lock
from config import WELDING_CH1_RTSP,WELDING_CH2_RTSP,WELDING_CH3_RTSP,WELDING_CH4_RTSP,WELDING_CH5_RTSP
from config import  SAVE_IMG_PATH,POST_IMG_PATH2,WELDING_REGION1,WELDING_REGION2,WELDING_REGION3
from globals import steps
from globals import oil_barrel,main_switch,grounding_wire,welding_machine_switch,welding_components,mask,welding,gloves,sweep,sweep_detect_num,welding_detect_num


def init_welding_detection():
    global steps
    global oil_barrel,main_switch,grounding_wire,welding_machine_switch,welding_components,mask,welding,gloves,sweep
    global sweep_detect_num,welding_detect_num
    oil_barrel=None
    main_switch=None
    grounding_wire=None
    welding_machine_switch=None
    welding_components=None
    mask=None
    welding=None
    gloves=None
    sweep=None

    sweep_detect_num=0
    welding_detect_num=0
    
    steps = [False] * 13
    redis_client.delete("welding_post_path")

def start_welding_detection(start_events):
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
        print("焊接子线程运行结束")



# Function to process video with YOLO model
def process_video(model_path, video_source,start_event):
    # Load YOLO model
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        if stop_event.is_set():#控制停止推理
        #del model
            break
    # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 != 0:
                continue
            
            if video_source == WELDING_CH2_RTSP:#这两个视频流用的分类模型，因为分类模型预处理较慢，需要手动resize
                frame=cv2.resize(frame,(640,640))
            
            results = model.predict(frame,verbose=False,conf=0.4)

            global steps
            global oil_barrel,main_switch,grounding_wire,welding_machine_switch,welding_components,mask,welding,gloves,sweep
            global sweep_detect_num,welding_detect_num

            for r in results:

                if video_source == WELDING_CH2_RTSP:#焊台
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
                                welding=True#表示有焊接
                        if label=='sweep' and welding==True:
                            if sweep_detect_num<3:
                                sweep_detect_num+=1
                            else:
                                sweep=True#表示有打扫
                    else:
                        continue


                #if video_source == WELDING_CH3_RTSP:#油桶
                # if video_source == WELDING_CH4_RTSP:#总开关
                #     if r.probs.top1conf>0.8:
                #         label=model.names[r.probs.top1]#获取最大概率的类别的label
                #         main_switch = "open" if label == "open" else "close"
                #     else:
                #         continue   


                
                if video_source == WELDING_CH1_RTSP or video_source==WELDING_CH3_RTSP or video_source==WELDING_CH5_RTSP or video_source==WELDING_CH4_RTSP:#焊接操作，进行目标检测
                    ##下面这些都是tensor类型
                    boxes = r.boxes.xyxy  # 提取所有检测到的边界框坐标
                    confidences = r.boxes.conf  # 提取所有检测到的置信度
                    classes = r.boxes.cls  # 提取所有检测到的类别索引


                    # if video_source==WELDING_CH5_RTSP:
                    #     grounding_wire=="disconnect"##单独每次设置为false，是为了防止没有检测到
                        #welding_components=False

                    if video_source==WELDING_CH3_RTSP:#当画面没有油桶时，给个初值为安全
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
                            welding_machine_switch = label

                        if label=="turnon":
                            main_switch="open"
                        if label=="turnoff":
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

                        # if label=="welding_components" :
                        #     welding_components_xyxy=boxes[i].tolist()#实时检测焊件的位置
                        #     # 计算检测框的中心点
                        #     x_center = (x1 + x2) / 2
                        #     y_center = (y1 + y2) / 2
                        #     center_point = (int(x_center), int(y_center))
                        #     # 检查中心点是否在多边形内
                        #     is_inside = cv2.pointPolygonTest(REGION4.reshape((-1, 1, 2)), center_point, False)
                        #     welding_components=True if is_inside>=0 else False #表示在焊料在焊台上

                        
                        if label=="mask":
                            #mask=True #表示戴面罩
                            iou=IoU(boxes[i].tolist(),WELDING_REGION1)
                            mask=True if iou>0 else False #表示戴面罩


                        if label=="gloves":
                            #gloves_xyxy=boxes[i].tolist()#实时检测手套的位置
                            if confidence>0.5:
                                gloves=True#表示戴手套
                            else:
                                gloves=False

                        # if label=="welding" :
                        #     iou1=IoU(gloves_xyxy,boxes[i].tolist())#自定义的矩形iou方法,焊枪跟手套进行iou计算
                        #     iou2=IoU(welding_components_xyxy,boxes[i].tolist())#自定义的矩形iou方法,焊枪跟焊件进行iou计算
                        #     if iou1>0 and iou2>0:
                        #         gloves=True#表示有手套焊接
                        #     if iou1<=0 and iou2>0:
                        #         welding=True#表示无手套焊接
                        
                        # if label=="sweep" :
                        #     # 计算检测框的中心点
                        #     x_center = (x1 + x2) / 2
                        #     y_center = (y1 + y2) / 2
                        #     center_point = (int(x_center), int(y_center))
                        #     # 检查中心点是否在多边形内
                        #     is_inside = cv2.pointPolygonTest(REGION4.reshape((-1, 1, 2)), center_point, False)
                        #     sweep=True if is_inside>=0 else False #表示是否打扫
                
                if video_source ==WELDING_CH3_RTSP:
                    if oil_barrel=="safe" and steps[0]==False:#排除危险源
                        steps[0]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step1_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step1_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step1完成")
                        #post("1",post_path)
                    
                if video_source==WELDING_CH4_RTSP:
                    if main_switch=="open" and steps[1]==False:
                        steps[1]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step2_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step2_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step2完成")
                        #post("2",post_path)

                    if main_switch=="close" and steps[12]==False and steps[1]:
                        steps[12]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step13_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step13_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step13完成")
                        #post("13",post_path)
                    
                if video_source==WELDING_CH1_RTSP:
                    if welding_machine_switch=="open" and steps[4]==False:
                        steps[4]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step5_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step5_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step5完成")
                        #post("5",post_path)
                    
                    if welding_machine_switch=="close" and steps[8]==False and steps[4]:
                        steps[8]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step9_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step9_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step9完成")
                        #post("9",post_path)
                    
                if video_source==WELDING_CH2_RTSP:
                    if sweep==True and steps[11]==False:#打扫
                        steps[11]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step12_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step12_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step12完成")
                        #post("12",post_path)
                    
                    if welding_components=='in_position' and steps[3]==False:
                        steps[3]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step4_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step4_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step4完成")
                        #post("4",post_path)

                    if welding_components=='not_in_position' and steps[10]==False and steps[3]:
                        steps[10]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step11_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step11_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step11完成")
                        #post("11",post_path)
                    
                    if welding==True and steps[6]==False:
                        steps[6]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step7_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step7_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step7完成")
                        #post("8",post_path)


                if video_source==WELDING_CH5_RTSP:
                    if gloves==True and steps[7]==False:
                        steps[7]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step8_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step8_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step8完成")
                        #post("7",post_path)

                    if grounding_wire=="connect" and steps[2]==False:
                        steps[2]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step3_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step3_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step3完成")
                        #post("3",post_path)
                            
                    if grounding_wire=="disconnect" and steps[9]==False and steps[2]:
                        steps[9]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step10_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step10_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step10完成")
                        #post("10",post_path)
                    
                            
                    if mask==True and steps[5]==False:
                        steps[5]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        #save_time=datetime.now().strftime('%Y%m%d_%H')
                        img_path = f"{SAVE_IMG_PATH}/step6_{save_time}.jpg"
                        post_path = f"{POST_IMG_PATH2}/step6_{save_time}.jpg"
                        redis_client.rpush("welding_post_path",post_path)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(img_path, annotated_frame)
                        print("step6完成")
                        #post("6",post_path)
            
            
                start_event.set()          
                # Display the annotated frame
            # cv2.imshow("YOLOv8 Inference", results[0].plot())

            # # Break the loop if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



        

    