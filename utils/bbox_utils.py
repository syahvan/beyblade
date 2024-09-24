def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def is_overlapping(bbox1, bbox2):
    if bbox1 and bbox2:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        width_inter = max(0, x_inter_max - x_inter_min)
        height_inter = max(0, y_inter_max - y_inter_min)
        area_inter = width_inter * height_inter

        if area_inter == 0:
            return False

        area_obj1 = (x1_max - x1_min) * (y1_max - y1_min)
        area_obj2 = (x2_max - x2_min) * (y2_max - y2_min)
        area_union = area_obj1 + area_obj2 - area_inter

        IoU = area_inter / area_union

        return IoU > 0 
    else:
        return False
