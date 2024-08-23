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

#吊篮清洗
basket_person_flag=False#吊篮区域是否存在人员
basket_suspension_flag=False#吊篮悬挂机构
basket_warning_zone_flag=False#吊篮警戒区
basket_steel_wire_flag=False#吊篮钢丝绳
basket_platform_flag=False#吊篮平台
basket_lifting_flag=False#吊篮升降机构
basket_safety_lock_flag=False#吊篮安全锁
basket_electrical_system_flag=False#吊篮电气系统
basket_empty_load_flag=False#吊篮空载
basket_safety_belt_flag=False#吊篮安全带挂设

basket_cleaning_operation_flag=False#吊篮清洗操作,检查刷子是否在指定区域
basket_cleaning_up_flag=False#吊篮清理现场