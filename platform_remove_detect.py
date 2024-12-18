import cv2
import torch
import time
import math
from shapely.geometry import box, Polygon
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from utils.tool import IoU
from globals import stop_event,redis_client
from config import SAVE_IMG_PATH,POST_IMG_PATH4,PLATFORM_SETUP_MODEL,PLATFORM_SETUP_VIDEO_SOURCES
from globals import platform_remove_steps_detect_num,platform_remove_final_result,platform_remove_steps_img,remove_detection_status,remove_detection_timers

def update_detection_status(platform_remove_steps):
    global remove_detection_status,remove_detection_timers
    current_time = time.time()
    
    for i,nums in enumerate(platform_remove_steps):
        if nums>0:
            remove_detection_timers[i] = current_time  # 更新物体的最后检测时间
            remove_detection_status[i] = False  # 重置状态为 False



def check_timeout():
    current_time = time.time()
    
    for i in range(14):
        if current_time - remove_detection_timers[i] > 10:  # 如果超过10秒未检测到
            remove_detection_status[i] = True  # 将状态置为 True
    #         return True
    # return False



def init_platform_remove_detection():
    global remove_detection_timers,remove_detection_status
    remove_detection_timers = [time.time()] * 14  # 初始化计时器
    remove_detection_status = [False]*14 # 初始化检
    redis_client.delete("platform_remove_order")
    for i in range(1, 14):
        redis_client.delete(f"platform_remove_{i}_img")


def start_platform_remove_detection(start_events):
    threads = []
    for video_source in PLATFORM_SETUP_VIDEO_SOURCES:
        event = threading.Event()
        start_events.append(event)
        thread = threading.Thread(target=process_video, args=(PLATFORM_SETUP_MODEL,video_source,event))
        threads.append(thread)
        thread.daemon=True
        thread.start()


    # Wait for all threads to complete
    for thread in threads:
        thread.join()
        print("平台拆除子线程启动")



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
            
            results = model.predict(frame,conf=0.6,verbose=False,task='obb')
            LEVLEL0_REGION = np.array([[1167, 908], [931, 1153], [1962, 1187], [2034, 936]], np.int32)
            LEVLEL1_REGION = np.array([[1163, 574], [859, 818], [1969, 828], [2060, 588]], np.int32)
            LEVLEL2_REGION = np.array([[1147, 263], [778, 438], [1953, 442], [2044, 248]], np.int32)
            LEVLEL3_REGION = np.array([[1142, 34], [793, 163], [1945, 112], [2050, 17]], np.int32)


            DIAGONAL_REGION = np.array([[838, 147], [845, 1145], [1935, 1147], [1943, 147]], np.int32)
            global platform_remove_steps_detect_num,remove_detection_status,platform_remove_steps_img
            
            # 每十秒归零 platform_remove_steps_detect_num
            # current_time = time.time()
            # if not hasattr(process_video, 'last_reset_time'):
            #     process_video.last_reset_time = current_time
            
            # if current_time - process_video.last_reset_time >= 10:
            #     platform_remove_steps_detect_num = [0] * 14
            #     process_video.last_reset_time = current_time


            for r in results:
                boxes=r.obb.xyxyxyxy
                boxes1=r.obb.xywhr
                confidences=r.obb.conf
                classes=r.obb.cls
                
                # 0: montant
                # 1: diagonal
                # 2: wheel
                # 3: vertical_bar
                # 4: horizontal_bar
                # 5: ladder
                # 6: toe_board
                # 7: scaffold


                # 1=轮子
                # 2=立杆
                # 3=纵向扫地杆
                # 4=横向扫地杆
                # 5=纵向水平杆1
                # 6=横向水平杆1
                # 7=纵向水平杆2
                # 8=横向水平杆2
                # 9=斜撑
                # 10=爬梯
                # 11=脚手板
                # 12=挡脚板
                # 13=纵向水平杆3
                # 14=横向水平杆3
                platform_remove_steps = [0] * 14
                hengxianggan=0
                zongxinaggan=0
                for i in range(len(boxes)):
                    confidence = confidences[i].item()
                    cls = int(classes[i].item())
                    label = model.names[cls]

                    #print(boxes[i].tolist())
                    box_coords = boxes[i].tolist()
                    x_center = (box_coords[0][0] + box_coords[1][0]+box_coords[2][0]+box_coords[3][0]) / 4
                    y_center=(box_coords[0][1] + box_coords[1][1]+box_coords[2][1]+box_coords[3][1]) / 4
                    center_point = (int(x_center), int(y_center))
                    rotation= math.degrees(boxes1[i][4])

                    if label=="wheel":#轮子
                        # platform_remove_steps_detect_num[0]+=1
                        # if platform_remove_steps_detect_num[0]>3:
                        platform_remove_steps[0]+=1
                    elif label=="bar":#立杆
                        # platform_setup_steps_detect_num[1]+=1
                        # if platform_setup_steps_detect_num[1]>3:
                        #     platform_setup_steps[1]+=1
                        if video_source == PLATFORM_SETUP_VIDEO_SOURCES[0]:
                            if 70 < rotation < 110:
                                platform_remove_steps[1] += 1  # 立杆
                            elif rotation < 10 or rotation > 170:
                                hengxianggan += 1  # 横向杆
                            elif 135 < rotation < 150:
                                platform_remove_steps[8] += 1  # 斜杆
                            else:
                                zongxinaggan += 1  # 纵向杆
                        elif video_source == PLATFORM_SETUP_VIDEO_SOURCES[1]:
                            if 70 < rotation < 110:
                                platform_remove_steps[1] += 1  # 立杆
                            elif 5 < rotation < 25:
                                hengxianggan += 1  # 横向杆
                            elif 40 < rotation < 60:
                                platform_remove_steps[8] += 1  # 斜杆
                            else:
                                zongxinaggan += 1  # 纵向杆
                    # elif label=="montant":#立杆
                    #     platform_remove_steps_detect_num[1]+=1
                    #     if platform_remove_steps_detect_num[1]>3:
                    #         platform_remove_steps[1]+=1
                    # elif label=="vertical_bar":#水平横杆,纵杆
                        

                    #     is_inside0 = cv2.pointPolygonTest(LEVLEL0_REGION.reshape((-1, 1, 2)), center_point, False)
                    #     is_inside1 = cv2.pointPolygonTest(LEVLEL1_REGION.reshape((-1, 1, 2)), center_point, False)
                    #     is_inside2 = cv2.pointPolygonTest(LEVLEL2_REGION.reshape((-1, 1, 2)), center_point, False)
                    #     is_inside3 = cv2.pointPolygonTest(LEVLEL3_REGION.reshape((-1, 1, 2)), center_point, False)
                    #     #print(is_inside)
                    #     if is_inside0>=0 :
                    #         platform_remove_steps_detect_num[2]+=1
                    #         if platform_remove_steps_detect_num[2]>3:
                    #             platform_remove_steps[2]+=1 #表示纵向扫地杆
                    #     elif is_inside1>=0:
                    #         platform_remove_steps_detect_num[4]+=1
                    #         if platform_remove_steps_detect_num[4]>3:
                    #             platform_remove_steps[4]+=1#5=纵向水平杆1
                    #     elif is_inside2>=0:
                    #         platform_remove_steps_detect_num[6]+=1
                    #         if platform_remove_steps_detect_num[6]>3:
                    #             platform_remove_steps[6]+=1#7=纵向水平杆2
                    #     elif is_inside3>=0:
                    #         platform_remove_steps_detect_num[12]+=1
                    #         if platform_remove_steps_detect_num[12]>3:
                    #             platform_remove_steps[12]+=1#13=纵向水平杆3
                        

                    # elif label=="horizontal_bar":#水平纵杆

                    #     is_inside0 = cv2.pointPolygonTest(LEVLEL0_REGION.reshape((-1, 1, 2)), center_point, False)
                    #     is_inside1 = cv2.pointPolygonTest(LEVLEL1_REGION.reshape((-1, 1, 2)), center_point, False)
                    #     is_inside2 = cv2.pointPolygonTest(LEVLEL2_REGION.reshape((-1, 1, 2)), center_point, False)
                    #     is_inside3 = cv2.pointPolygonTest(LEVLEL3_REGION.reshape((-1, 1, 2)), center_point, False)
                    #     #print(is_inside)
                    #     if is_inside0>=0 :
                    #         platform_remove_steps_detect_num[3]+=1
                    #         if platform_remove_steps_detect_num[3]>3:
                    #             platform_remove_steps[3]+=1#4=横向扫地杆
                    #     elif is_inside1>=0:
                    #         platform_remove_steps_detect_num[5]+=1
                    #         if platform_remove_steps_detect_num[5]>3:
                    #             platform_remove_steps[5]+=1#6=横向水平杆1
                    #     elif is_inside2>=0:
                    #         platform_remove_steps_detect_num[7]+=1
                    #         if platform_remove_steps_detect_num[7]>3:
                    #             platform_remove_steps[7]+=1# 8=横向水平杆2
                    #     elif is_inside3>=0:
                    #         platform_remove_steps_detect_num[13]+=1
                    #         if platform_remove_steps_detect_num[13]>3:
                    #             platform_remove_steps[13]+=1#14=横向水平杆3

                    # elif label=="diagonal":#斜撑

                    #     is_inside = cv2.pointPolygonTest(DIAGONAL_REGION.reshape((-1, 1, 2)), center_point, False)
                    #     if is_inside>=0:
                    #         platform_remove_steps_detect_num[8]+=1
                    #         if platform_remove_steps_detect_num[8]>3:
                    #             platform_remove_steps[8]+=1# 9=斜撑
                    

                    elif label=="ladder":#梯子
                        #10=爬梯
                        # platform_remove_steps_detect_num[9]+=1
                        # if platform_remove_steps_detect_num[9]>3:
                        platform_remove_steps[9]+=1
                        
                    elif label=="scaffold":#脚手板
                        #11=脚手板
                        # platform_remove_steps_detect_num[10]+=1
                        # if platform_remove_steps_detect_num[10]>3:
                        platform_remove_steps[10]+=1


                    elif label=="toe_board":#档脚板
                        #12=挡脚板
                        # platform_remove_steps_detect_num[11]+=1
                        # if platform_remove_steps_detect_num[11]>3:
                        platform_remove_steps[11]+=1

                if hengxianggan >= 2:
                    platform_remove_steps[3] = 2
                if hengxianggan >= 4:
                    platform_remove_steps[5] = 2
                if hengxianggan >= 6:
                    platform_remove_steps[7] = 2
                if hengxianggan >= 8:
                    platform_remove_steps[13] = 2
                
                if zongxinaggan>=2:
                    platform_remove_steps[2]=2
                if zongxinaggan>=4:
                    platform_remove_steps[4]=2
                if zongxinaggan>=6:
                    platform_remove_steps[6]=2
                if zongxinaggan>=8:
                    platform_remove_steps[12]=2


                update_detection_status(platform_remove_steps)
                check_timeout()#返回True表示10秒内未检测到某物体，表示该物体拆除完成

                for i, status in reversed(list(enumerate(remove_detection_status))):
                    if status and platform_remove_steps_img[i]==False:
                        redis_client.rpush("platform_remove_order", i+1)
                        platform_remove_steps_img[i]=True
                        save_time=datetime.now().strftime('%Y%m%d_%H%M')
                        imgpath = f"{SAVE_IMG_PATH}/platform_remove{i+1}_{save_time}.jpg"
                        post_path= f"{POST_IMG_PATH4}/platform_remove{i+1}_{save_time}.jpg"
                        annotated_frame = results[0].plot()
                        cv2.imwrite(imgpath, annotated_frame)
                        redis_client.set(f"platform_remove_{i+1}_img",post_path)
                        #redis_client.set(f"platform_remove_{14-i-1}", "1")






                start_event.set()          


        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model



        

    