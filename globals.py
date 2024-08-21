import threading
# 全局变量来控制推理线程
import redis
import time
# 连接到 Redis 服务器
redis_client = redis.StrictRedis(host='localhost', port=5050, db=0,decode_responses=True)

inference_thread = None
stop_event = threading.Event()
lock=threading.Lock()

#condition = threading.Condition()
###############焊接考核
#为True时，表示某一步骤完成,并保存图片post
step1=False #危险源排除
step2=False
step3=False
step4=False
step5=False
step6=False #危险源排除
step7=False
step8=False
step9=False
step10=False
step11=False
step12=False
step13=False

steps = [False] * 13

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
###############




###########检测物品是否复位
oil_barrel_flag=False
main_switch_flag=False
ground_wire_flag=False
welding_components_flag=False
welding_machine_switch_flag=False

oil_barrel_save_img=False
main_switch_save_img=False
ground_wire_save_img=False
welding_components_save_img=False
welding_machine_switch_save_img=False


reset_all=None
log_in_flag=False#登录标志，如前端未登录，不允许保存图片并post
###############################
###############平台搭设考核
platform_setup_steps_detect_num=[0]*14
platform_setup_final_result=[0]*14
platform_setup_steps_img=[False]*14
################平台拆除考核
platform_remove_steps_detect_num=[0]*14
platform_remove_final_result=[0]*14
platform_remove_steps_img=[False]*14

remove_detection_timers = [time.time()] * 14  # 初始化计时器
remove_detection_status = [False]*14 # 初始化检