from shapely.geometry import box, Polygon



def IoU(box1, box2):
    '''
    计算两个矩形框的交并比
    :param box1: list,第一个矩形框的左上角和右下角坐标
    :param box2: list,第二个矩形框的左上角和右下角坐标
    :return: 两个矩形框的交并比iou
    '''
    x1 = max(box1[0], box2[0])   # 交集左上角x
    x2 = min(box1[2], box2[2])   # 交集右下角x
    y1 = max(box1[1], box2[1])   # 交集左上角y
    y2 = min(box1[3], box2[3])   # 交集右下角y

    overlap = max(0., x2-x1) * max(0., y2-y1)
    union = (box1[2]-box1[0]) * (box1[3]-box1[1]) \
            + (box2[2]-box2[0]) * (box2[3]-box2[1]) \
            - overlap

    if union == 0:
        return 0.0

    return overlap / union

def IoU_polygon(box1, polygon_points):
    '''
    计算矩形框与多边形的交并比
    :param box1: list, 矩形框的左上角和右下角坐标
    :param polygon_points: list, 多边形的顶点坐标
    :return: 矩形框与多边形的交并比iou
    '''
    rect_shapely = box(box1[0], box1[1], box1[2], box1[3])  # 使用shapely库创建的矩形
    polygon_shapely = Polygon(polygon_points)  # 使用shapely库创建的多边形

    intersection = rect_shapely.intersection(polygon_shapely)
    union = rect_shapely.union(polygon_shapely)

    if union.area == 0:
        return 0.0

    iou = intersection.area / union.area
    return iou