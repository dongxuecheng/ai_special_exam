import numpy as np


#CLIENT_URL = 'http://172.16.20.23:8081/'

SAVE_IMG_PATH = '/mnt/xcd/code/ai_test/static/images'  # 图片保存在服务器的实际位置

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
WELDING_CH3_RTSP = 'rtsp://admin:yaoan1234@172.16.22.233/cam/realmonitor?channel=1&subtype=0'#检查油桶视频
WELDING_CH4_RTSP = 'rtsp://admin:yaoan1234@172.16.22.232/cam/realmonitor?channel=1&subtype=0'#检查总开关视频
WELDING_CH5_RTSP = 'rtsp://admin:yaoan1234@172.16.22.234/cam/realmonitor?channel=1&subtype=0'#检查面具手套接地线视频
WELDING_CH6_RTSP = 'rtsp://admin:yaoan1234@172.16.22.235/cam/realmonitor?channel=1&subtype=0'#劳保用品穿戴视频

WELDING_CH1_MODEL="/mnt/xcd/code/ai_test/weights/ch1_welding_switch_813.pt"
WELDING_CH2_MODEL="/mnt/xcd/code/ai_test/weights/ch2_welding_desk_cls_813.pt"
WELDING_CH3_MODEL="/mnt/xcd/code/ai_test/weights/ch3_oil_barrel_detect_813.pt"
WELDING_CH4_MODEL="/mnt/xcd/code/ai_test/weights/ch4_main_switch_cls_813.pt"
WELDING_CH5_MODEL="/mnt/xcd/code/ai_test/weights/ch5_mask_gloves_wire_detect_813.pt"
WELDING_CH6_MODEL='/mnt/xcd/code/ai_test/weights/ch6_wearing_detect_813.pt'

HUMAN_DETECTION_MODEL="/mnt/xcd/code/ai_test/weights/yolov8n.pt"#人体检测模型

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

WELDING_REGION2 = np.array([[607, 555], [454, 0], [2560, 0], [2560, 1440], [430, 1440]], np.int32)

# 搭铁夹连接焊台位置

WELDING_REGION3 = np.array([[1613, 627], [1601, 658], [1697, 987], [1710, 962]], np.int32)


####平台搭设视频流
PLATFORM_CH1_RTSP='rtsp://admin:yaoan1234@172.16.22.241/cam/realmonitor?channel=1&subtype=0'#检测穿戴

PLATFORM_CH2_RTSP='rtsp://admin:yaoan1234@172.16.22.240/cam/realmonitor?channel=1&subtype=0'#脚手架搭建
# PLATFORM_CH3_RTSP='rtsp://admin:yaoan1234@172.16.22.243/cam/realmonitor?channel=1&subtype=0'#脚手架搭建
PLATFORM_CH4_RTSP='rtsp://admin:yaoan1234@172.16.22.233/cam/realmonitor?channel=1&subtype=0'#脚手架搭建

PLATFORM_CH1_MODEL='/mnt/xcd/code/ai_test/weights/platform_ch1_wearing.pt'

PLATFORM_SETUP_MODEL='/mnt/xcd/code/ai_test/weights/obb_9_2.pt'
PLATFORM_SETUP_VIDEO_SOURCES=[PLATFORM_CH2_RTSP,
                              #PLATFORM_CH3_RTSP,
                              PLATFORM_CH4_RTSP
]

PLATFORM_WEARING_MODEL=[
    HUMAN_DETECTION_MODEL,
    PLATFORM_CH1_MODEL    
]

PLATFORM_WEARING_VIDEO_SOURCES=PLATFORM_CH1_RTSP

#PLATFORM_CH2_MODEL='/mnt/xcd/code/ai_test/weights/high_work_obb_final.pt'

# Define paths to input videos


#焊接劳保检测相关参数

#################平台搭设检测相关参数

# PLATFORM_SETUP_VIDEO_SOURCES=PLATFORM_CH2_RTSP
# PLATFORM_SETUP_MODEL=PLATFORM_SETUP_MODEL

#吊篮清洗


BASKET_CLEANING_CH4_POSE_MODEL='/mnt/xcd/code/ai_test/weights/yolov8s-pose1.pt'
BASKET_CLEANING_CH5_DETECT_MODEL='/mnt/xcd/code/ai_test/weights/yolov8s1.pt'
BASKET_CLEANING_CH6_POSE_MODEL='/mnt/xcd/code/ai_test/weights/yolov8s-pose2.pt'
BASKET_CLEANING_CH6_DETECT_MODEL='/mnt/xcd/code/ai_test/weights/ch6detect_basket.pt'
BASKET_CLEANING_CH6_SEG_MODEL='/mnt/xcd/code/ai_test/weights/basket_seg.pt'

BASKET_CLEANING_CH4_RTSP='rtsp://admin:yaoan1234@172.16.22.237/cam/realmonitor?channel=1&subtype=0'
#BASKET_CLEANING_CH5_RTSP='rtsp://admin:yaoan1234@172.16.22.239/cam/realmonitor?channel=1&subtype=0'
BASKET_CLEANING_CH6_RTSP='rtsp://admin:yaoan1234@172.16.22.242/cam/realmonitor?channel=1&subtype=0'

BASKET_CLEANING_VIDEO_SOURCES=[BASKET_CLEANING_CH4_RTSP,
                               #BASKET_CLEANING_CH5_RTSP,
                               BASKET_CLEANING_CH6_RTSP,
                               BASKET_CLEANING_CH6_RTSP,
                               BASKET_CLEANING_CH6_RTSP]

BASKET_CLEANING_MODEL_SOURCES=[BASKET_CLEANING_CH4_POSE_MODEL,
                               #BASKET_CLEANING_CH5_DETECT_MODEL,
                               BASKET_CLEANING_CH6_POSE_MODEL,
                               BASKET_CLEANING_CH6_DETECT_MODEL,
                               BASKET_CLEANING_CH6_SEG_MODEL]

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
BASKET_LIFTING_REGION = np.array([]
,np.int32)

BASKET_SAFETY_LOCK_REGION = np.array([
    [[1635, 813], [1742, 927], [1955, 910], [1906, 747]],
    [[650, 944], [800, 1000], [800, 923], [680, 872]]
    ], np.int32)
BASKET_ELECTRICAL_SYSTEM_REGION = np.array([], np.int32)
BASKET_CLEANING_OPERATION_REGION = np.array([[8, 1038], [14, 1423], [1910, 1432], [1894, 1129]], np.int32)
BASKET_EMPTY_LOAD_REGION = np.array([(752, 855), (712, 969), (836, 1020), (896, 918)], np.int32)


#单人吊具
EQUIPMENT_CLEANING_CH3_RTSP='rtsp://admin:yaoan1234@172.16.22.238/cam/realmonitor?channel=1&subtype=0'
EQUIPMENT_CLEANING_CH8_RTSP='rtsp://admin:yaoan1234@172.16.22.44/cam/realmonitor?channel=1&subtype=0'


EQUIPMENT_CLEANING_CH3_DETECT_MODEL='/mnt/xcd/code/ai_test/weights/ch6detect_basket.pt'
EQUIPMENT_CLEANING_CH8_POSE_MODEL='/mnt/xcd/code/ai_test/weights/yolov8s-pose2.pt'
EQUIPMENT_CLEANING_CH8_DETECT_MODEL='/mnt/xcd/code/ai_test/weights/ch6detect_basket.pt'

EQUIPMENT_CLEANING_VIDEO_SOURCES=[EQUIPMENT_CLEANING_CH3_RTSP,
                                  EQUIPMENT_CLEANING_CH8_RTSP,
                                  EQUIPMENT_CLEANING_CH8_RTSP
]

EQUIPMENT_CLEANING_MODEL_SOURCES=[EQUIPMENT_CLEANING_CH3_DETECT_MODEL,
                                  EQUIPMENT_CLEANING_CH8_POSE_MODEL,
                                  EQUIPMENT_CLEANING_CH8_DETECT_MODEL
]

EQUIPMENT_ANCHOR_DEVICE_REGION = np.array([
    [[913, 6], [914, 520], [1350, 523], [1351, 9]],
], np.int32)
EQUIPMENT_WORK_ROPE_REGION = np.array([
    [[1466, 1187], [1416, 1248], [2500, 1162], [2502, 1063]],
], np.int32)
EQUIPMENT_SAFETY_ROPE_REGION = np.array([
    [[1466, 1187], [1416, 1248], [2500, 1162], [2502, 1063]],
], np.int32)
EQUIPMENT_CLEANING_OPERATION_REGION=np.array([[1697, 930], [1055, 1253], [2000, 1420], [2450, 1000]], np.int32)