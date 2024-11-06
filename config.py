import numpy as np


#CLIENT_URL = 'http://172.16.20.23:8081/'

SAVE_IMG_PATH = '/mnt/xcd/code/ai_special_exam/static/images'  # 图片保存在服务器的实际位置

POST_IMG_PATH1 = 'http://172.16.20.163:5001/images'  # 通过端口映射post发送能够访问的位置 焊接考核科目1
POST_IMG_PATH2 = 'http://172.16.20.163:5002/images' #焊接考核科目2
POST_IMG_PATH3 = 'http://172.16.20.163:5003/images' #平台搭设科目1，劳保穿戴
POST_IMG_PATH4 = 'http://172.16.20.163:5004/images' #平台搭设科目2，搭建和拆除
POST_IMG_PATH5 = 'http://172.16.20.163:5005/images'#吊篮清洗
POST_IMG_PATH6 = 'http://172.16.20.163:5006/images'#吊具清洗

#焊接考核视频流
# Define paths to RTSP streams
WELDING_CH1_RTSP = 'rtsp://admin:yaoan1234@172.16.22.230/cam/realmonitor?channel=1&subtype=0'#焊机开关视频
WELDING_CH2_RTSP = 'rtsp://admin:yaoan1234@172.16.22.231/cam/realmonitor?channel=1&subtype=0'#焊台视频
WELDING_CH3_RTSP = 'rtsp://admin:yaoan1234@172.16.22.247/cam/realmonitor?channel=1&subtype=0'#检查油桶视频
WELDING_CH4_RTSP = 'rtsp://admin:yaoan1234@172.16.22.232/cam/realmonitor?channel=1&subtype=0'#检查总开关视频
WELDING_CH5_RTSP = 'rtsp://admin:yaoan1234@172.16.22.234/cam/realmonitor?channel=1&subtype=0'#检查面具手套接地线视频
WELDING_CH6_RTSP = 'rtsp://admin:yaoan1234@172.16.22.235/cam/realmonitor?channel=1&subtype=0'#劳保用品穿戴视频

# WELDING_CH1_MODEL="/mnt/xcd/code/ai_special_exam/weights/ch1_welding_switch_813.pt"
# WELDING_CH2_MODEL="/mnt/xcd/code/ai_special_exam/weights/ch2_welding_desk_cls_815_m.pt"
# WELDING_CH3_MODEL="/mnt/xcd/code/ai_special_exam/weights/ch3_oil_barrel_detect_815_m.pt"
# WELDING_CH4_MODEL="/mnt/xcd/code/ai_special_exam/weights/ch4_main_switch_detect_910_m.pt"
# WELDING_CH5_MODEL="/mnt/xcd/code/ai_special_exam/weights/ch5_mask_gloves_wire_detect_910_s_p2.pt"
# WELDING_CH6_MODEL='/mnt/xcd/code/ai_special_exam/weights/ch6_wearing_detect_815_m.pt'

WELDING_CH1_MODEL="/mnt/xcd/code/ai_special_exam/weights/11n/ch1_welding_switch_11n_detect_11_5.pt"
WELDING_CH2_MODEL="/mnt/xcd/code/ai_special_exam/weights/ch2_welding_desk_cls_815_m.pt"
WELDING_CH3_MODEL="/mnt/xcd/code/ai_special_exam/weights/11n/ch3_oil_barrel_11n_detect_115.pt"
WELDING_CH4_MODEL="/mnt/xcd/code/ai_special_exam/weights/11n/ch4_main_switch_11n_detect_115.pt"
WELDING_CH5_MODEL="/mnt/xcd/code/ai_special_exam/weights/11n/ch5_mask_gloves_wire_11n_detect_115.pt"
WELDING_CH6_MODEL='/mnt/xcd/code/ai_special_exam/weights/11n/ch6_wearing_detect_115.pt'

HUMAN_DETECTION_MODEL="/mnt/xcd/code/ai_special_exam/weights/11n/yolo11n.pt"#人体检测模型

# Define paths to models
WELDING_MODEL_PATHS = [
    WELDING_CH1_MODEL,
    WELDING_CH2_MODEL,
    WELDING_CH3_MODEL,
    WELDING_CH4_MODEL,
    WELDING_CH5_MODEL
]

WELDING_VIDEO_SOURCES = [
    WELDING_CH1_RTSP,
    WELDING_CH2_RTSP,
    WELDING_CH3_RTSP,
    WELDING_CH4_RTSP,
    WELDING_CH5_RTSP
]



WELDING_WEARING_MODEL=[
    HUMAN_DETECTION_MODEL,
    WELDING_CH6_MODEL
]

WELDING_WEARING_VIDEO_SOURCES= WELDING_CH6_RTSP

#WEAR_DETECTION_VIDEO_SOURCES= "/home/dxc/special_test/ch1.mp4"

# 劳保用品 指定要检测的区域 (xmin, ymin, xmax, ymax)
WEAR_DETECTION_AREA = (350, 0, 1400, 1080)


# 头盔检测区域(xmin, ymin, xmax, ymax)

WELDING_REGION1=(1499,339,1839,723)
# 油桶危险区域（多边形）

WELDING_REGION2 = np.array([[341, 0], [443, 399], [329, 856], [7, 247], [26, 1420],[2555, 1428], [2550, 10]], np.int32)

# 搭铁夹连接焊台位置

WELDING_REGION3 = np.array([[1534, 627], [1664, 1054], [2000, 1015], [1844, 588]], np.int32)


####平台搭设视频流
PLATFORM_CH1_RTSP='rtsp://admin:yaoan1234@172.16.22.241/cam/realmonitor?channel=1&subtype=0'#检测穿戴

PLATFORM_CH2_RTSP='rtsp://admin:yaoan1234@172.16.22.240/cam/realmonitor?channel=1&subtype=0'#脚手架搭建
# PLATFORM_CH3_RTSP='rtsp://admin:yaoan1234@172.16.22.243/cam/realmonitor?channel=1&subtype=0'#脚手架搭建
PLATFORM_CH4_RTSP='rtsp://admin:yaoan1234@172.16.22.233/cam/realmonitor?channel=1&subtype=0'#脚手架搭建

PLATFORM_CH1_MODEL='/mnt/xcd/code/ai_special_exam/weights/platform_ch1_wearing_9_3.pt'

PLATFORM_SETUP_MODEL='/mnt/xcd/code/ai_special_exam/weights/obb_9_2.pt'
PLATFORM_SETUP_VIDEO_SOURCES=[PLATFORM_CH2_RTSP,
                              #PLATFORM_CH3_RTSP,
                              PLATFORM_CH4_RTSP
]

PLATFORM_WEARING_MODEL=[
    HUMAN_DETECTION_MODEL,
    PLATFORM_CH1_MODEL    
]

PLATFORM_WEARING_VIDEO_SOURCES=PLATFORM_CH1_RTSP

#PLATFORM_CH2_MODEL='/mnt/xcd/code/ai_special_exam/weights/high_work_obb_final.pt'

# Define paths to input videos


#焊接劳保检测相关参数

#################平台搭设检测相关参数

# PLATFORM_SETUP_VIDEO_SOURCES=PLATFORM_CH2_RTSP
# PLATFORM_SETUP_MODEL=PLATFORM_SETUP_MODEL

#吊篮清洗


BASKET_CLEANING_CH4_POSE_MODEL='/mnt/xcd/code/ai_special_exam/weights/yolov8s-pose1.pt'
BASKET_CLEANING_CH5_DETECT_MODEL='/mnt/xcd/code/ai_special_exam/weights/ch6detect_basker_9_14_a_1.pt'
BASKET_CLEANING_CH6_POSE_MODEL='/mnt/xcd/code/yolov8/yolo11m-pose.pt'
BASKET_CLEANING_CH6_DETECT_MODEL='/mnt/xcd/code/ai_special_exam/weights/ch6detect_basker_9_14_a.pt'
BASKET_CLEANING_CH6_SEG_MODEL='/mnt/xcd/code/ai_special_exam/weights/basket_seg_9_13.pt'
BASKET_CLEANING_CH6_SAFETY_BELT_MODEL='/mnt/xcd/code/yolov8/runs/detect/train51/weights/last.pt'

BASKET_CLEANING_CH4_RTSP='rtsp://admin:yaoan1234@172.16.22.237/cam/realmonitor?channel=1&subtype=0'
BASKET_CLEANING_CH5_RTSP='rtsp://admin:yaoan1234@172.16.22.239/cam/realmonitor?channel=1&subtype=0'
BASKET_CLEANING_CH6_RTSP='rtsp://admin:yaoan1234@172.16.22.242/cam/realmonitor?channel=1&subtype=0'

BASKET_CLEANING_VIDEO_SOURCES=[BASKET_CLEANING_CH4_RTSP,
                               BASKET_CLEANING_CH5_RTSP,
                               BASKET_CLEANING_CH6_RTSP]
                            #    BASKET_CLEANING_CH6_RTSP,
                            #    BASKET_CLEANING_CH6_RTSP,
                            #    BASKET_CLEANING_CH6_RTSP]

BASKET_CLEANING_MODEL_SOURCES=[BASKET_CLEANING_CH4_POSE_MODEL,
                               BASKET_CLEANING_CH5_DETECT_MODEL,
                               BASKET_CLEANING_CH6_POSE_MODEL,
                               BASKET_CLEANING_CH6_DETECT_MODEL,
                               BASKET_CLEANING_CH6_SEG_MODEL,
                               BASKET_CLEANING_CH6_SAFETY_BELT_MODEL]

#

#悬挂机构区域，分为四个区域 D4
BASKET_SUSPENSION_REGION = np.array([
    [[668, 310], [800, 310], [800, 1070], [668, 1070]],
    [[1690, 310], [1750, 310], [1750, 710], [1690, 710]],
    [[1350, 340], [1405, 340], [1405, 720], [1350, 720]],
    [[550, 385], [635, 385], [635, 880], [550, 880]]
], np.int32)

BASKET_STEEL_WIRE_REGION = np.array([
    [(374, 846), (601, 970), (630, 900), (441, 786)],  # 右一多边形区域
    [(1518, 736), (1649, 945), (2005, 917), (1888, 677)]  # 右二多边形区域
    # [(1293, 0), (1867, 935), (1904, 906), (1354, 9)],  # 左边多边形区域
], np.int32)#钢丝绳区域，暂时没有钢丝绳的区域

BASKET_PLATFORM_REGION = np.array([], np.int32)
BASKET_LIFTING_REGION_L = np.array([]
,np.int32)
BASKET_LIFTING_REGION_R = np.array([]
,np.int32)
BASKET_ELECTRICAL_SYSTEM_REGION = np.array([], np.int32)

BASKET_SAFETY_LOCK_REGION = np.array([
    [[1635, 813], [1742, 927], [1955, 910], [1906, 747]],
    [[650, 944], [800, 1000], [800, 923], [680, 872]]
    ], np.int32)

BASKET_CLEANING_OPERATION_REGION = np.array([[6, 954], [4, 1437], [1054, 1436], [1051, 954]], np.int32)
BASKET_EMPTY_LOAD_REGION = np.array([(752, 855), (712, 969), (836, 1020), (896, 918)], np.int32)
BASKET_WARNING_ZONE_REGION=np.array([(1250, 0), (1256, 469), (2048, 492), (2048, 0)], np.int32)

#单人吊具
EQUIPMENT_CLEANING_CH3_RTSP='rtsp://admin:yaoan1234@172.16.22.238/cam/realmonitor?channel=1&subtype=0'
EQUIPMENT_CLEANING_CH8_RTSP='rtsp://admin:yaoan1234@172.16.22.44/cam/realmonitor?channel=1&subtype=0'
#EQUIPMENT_CLEANING_CH10_RTSP='rtsp://admin:yaoan1234@172.16.22.44/cam/realmonitor?channel=1&subtype=0'


EQUIPMENT_CLEANING_CH3_DETECT_MODEL1='/mnt/xcd/code/yolov8/runs/detect/train52/weights/last.pt'#增加了座板的标签的模型
EQUIPMENT_CLEANING_CH3_DETECT_MODEL2='/mnt/xcd/code/ai_special_exam/weights/yolov8s.pt'
#EQUIPMENT_CLEANING_CH10_DETECT_MODEL1='/mnt/xcd/code/ai_special_exam/weights/ch6detect_basket_9_6_1.pt'
EQUIPMENT_CLEANING_CH8_POSE_MODEL='/mnt/xcd/code/ai_special_exam/weights/yolov8s-pose2.pt'
EQUIPMENT_CLEANING_CH8_DETECT_MODEL='/mnt/xcd/code/yolov8/runs/detect/train50/weights/last1.pt'

#检测安全带/自锁器
EQUIPMENT_CLEANING_CH8_SAFETY_BELT_DETECT_MODEL='/mnt/xcd/code/yolov8/runs/detect/train51/weights/last.pt'

EQUIPMENT_CLEANING_VIDEO_SOURCES=[EQUIPMENT_CLEANING_CH3_RTSP,
                                  #EQUIPMENT_CLEANING_CH3_RTSP,
                                  #EQUIPMENT_CLEANING_CH10_RTSP,
                                  EQUIPMENT_CLEANING_CH8_RTSP,
                                  #EQUIPMENT_CLEANING_CH8_RTSP,
                                  #EQUIPMENT_CLEANING_CH8_RTSP
]

EQUIPMENT_CLEANING_MODEL_SOURCES=[EQUIPMENT_CLEANING_CH3_DETECT_MODEL1,
                                  EQUIPMENT_CLEANING_CH3_DETECT_MODEL2,
                                  #EQUIPMENT_CLEANING_CH10_DETECT_MODEL1,
                                  EQUIPMENT_CLEANING_CH8_POSE_MODEL,
                                  EQUIPMENT_CLEANING_CH8_DETECT_MODEL,
                                  EQUIPMENT_CLEANING_CH8_SAFETY_BELT_DETECT_MODEL
]

EQUIPMENT_WARNING_ZONE_REGION = np.array([
    [[813, 835], [815, 1435], [1682, 1435], [1681, 835]],
], np.int32)

EQUIPMENT_ANCHOR_DEVICE_REGION = np.array([
    [[855, 0], [855, 536], [1288, 536], [1288, 0]],
], np.int32)
EQUIPMENT_WORK_ROPE_REGION = np.array([
    [[368, 1193], [1546, 1400], [1803, 1178], [1378, 1126]],
], np.int32)
EQUIPMENT_SAFETY_ROPE_REGION = np.array([
    [[368, 1193], [1546, 1400], [1803, 1178], [1378, 1126]],
], np.int32)
EQUIPMENT_SELF_LOCKING_DEVICE_REGION = np.array([
    [],
], np.int32)

EQUIPMENT_CLEANING_OPERATION_REGION=np.array([
    [[405, 693], [405, 1422], [1827, 1422], [1827, 685]],
    ], np.int32)