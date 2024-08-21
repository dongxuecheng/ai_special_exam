import numpy as np


#CLIENT_URL = 'http://172.16.20.23:8081/'

SAVE_IMG_PATH = '/mnt/xcd/code/ai_test/static/images'  # 图片保存在服务器的实际位置

POST_IMG_PATH1 = 'http://172.16.20.163:5001/images'  # 通过端口映射post发送能够访问的位置 焊接考核科目1
POST_IMG_PATH2 = 'http://172.16.20.163:5002/images' #焊接考核科目2
POST_IMG_PATH3 = 'http://172.16.20.163:5003/images' #平台搭设科目1，劳保穿戴
POST_IMG_PATH4 = 'http://172.16.20.163:5004/images' #平台搭设科目2，搭建和拆除

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

PLATFORM_CH3_RTSP='rtsp://admin:yaoan1234@172.16.22.243/cam/realmonitor?channel=1&subtype=0'#脚手架搭建

PLATFORM_CH1_MODEL='/mnt/xcd/code/ai_test/weights/platform_ch1_wearing.pt'
PLATFORM_CH2_MODEL='/mnt/xcd/code/ai_test/weights/high_work_obb_final.pt'

# Define paths to input videos


#焊接劳保检测相关参数

#################平台搭设检测相关参数
PLATFORM_WEARING_MODEL=[
    HUMAN_DETECTION_MODEL,
    PLATFORM_CH1_MODEL    
]

PLATFORM_WEARING_VIDEO_SOURCES=PLATFORM_CH1_RTSP
PLATFORM_SETUP_VIDEO_SOURCES=PLATFORM_CH2_RTSP
PLATFORM_SETUP_MODEL=PLATFORM_CH2_MODEL

#吊篮清洗


BASKET_CLEANING_CH4_POSE_MODEL='/mnt/xcd/code/ai_test/weights/yolov8s-pose.pt'
BASKET_CLEANING_CH5_DETECT_MODEL='/mnt/xcd/code/ai_test/weights/detect.pt'

BASKET_CLEANING_CH6_POSE_MODEL='/mnt/xcd/code/ai_test/weights/yolov8s-pose.pt'
BASKET_CLEANING_CH6_DETECT_MODEL='/mnt/xcd/code/ai_test/weights/detect.pt'

BASKET_CLEANING_CH4_RTSP='rtsp://admin:yaoan1234@172.16.22.237/cam/realmonitor?channel=1&subtype=0'
BASKET_CLEANING_CH5_RTSP='rtsp://admin:yaoan1234@172.16.22.239/cam/realmonitor?channel=1&subtype=0'
BASKET_CLEANING_CH6_RTSP='rtsp://admin:yaoan1234@172.16.22.242/cam/realmonitor?channel=1&subtype=0'

BASKET_CLEANING_VIDEO_SOURCES=[BASKET_CLEANING_CH4_RTSP,
                               BASKET_CLEANING_CH5_RTSP,
                               BASKET_CLEANING_CH6_RTSP,
                               BASKET_CLEANING_CH6_RTSP]

BASKET_CLEANING_MODEL_SOURCES=[BASKET_CLEANING_CH4_POSE_MODEL,
                               BASKET_CLEANING_CH5_DETECT_MODEL,
                               BASKET_CLEANING_CH6_POSE_MODEL,
                               BASKET_CLEANING_CH6_DETECT_MODEL]