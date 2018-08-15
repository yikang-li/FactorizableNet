# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import pdb

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def unary_nms(dets, classes, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where((ovr <= thresh) | (classes[i] != classes[order[1:]]))[0]
        order = order[inds + 1]

    return keep

def triplet_nms(sub_ids, obj_ids, pred_ids, sub_boxes, obj_boxes, thresh):
    #print('before: {}'.format(len(sub_ids))),
    sub_x1 = sub_boxes[:, 0]
    sub_y1 = sub_boxes[:, 1]
    sub_x2 = sub_boxes[:, 2]
    sub_y2 = sub_boxes[:, 3]
    obj_x1 = obj_boxes[:, 0]
    obj_y1 = obj_boxes[:, 1]
    obj_x2 = obj_boxes[:, 2]
    obj_y2 = obj_boxes[:, 3]


    sub_areas = (sub_x2 - sub_x1 + 1) * (sub_y2 - sub_y1 + 1)
    obj_areas = (obj_x2 - obj_x1 + 1) * (obj_y2 - obj_y1 + 1)
    order = np.array(range(len(sub_ids)))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        sub_xx1 = np.maximum(sub_x1[i], sub_x1[order[1:]])
        sub_yy1 = np.maximum(sub_y1[i], sub_y1[order[1:]])
        sub_xx2 = np.minimum(sub_x2[i], sub_x2[order[1:]])
        sub_yy2 = np.minimum(sub_y2[i], sub_y2[order[1:]])
        sub_id = sub_ids[i]
        obj_xx1 = np.maximum(obj_x1[i], obj_x1[order[1:]])
        obj_yy1 = np.maximum(obj_y1[i], obj_y1[order[1:]])
        obj_xx2 = np.minimum(obj_x2[i], obj_x2[order[1:]])
        obj_yy2 = np.minimum(obj_y2[i], obj_y2[order[1:]])
        obj_id = obj_ids[i]
        pred_id = pred_ids[i]

        w = np.maximum(0.0, sub_xx2 - sub_xx1 + 1)
        h = np.maximum(0.0, sub_yy2 - sub_yy1 + 1)
        inter = w * h
        sub_ovr = inter / (sub_areas[i] + sub_areas[order[1:]] - inter)

        w = np.maximum(0.0, obj_xx2 - obj_xx1 + 1)
        h = np.maximum(0.0, obj_yy2 - obj_yy1 + 1)
        inter = w * h
        obj_ovr = inter / (obj_areas[i] + obj_areas[order[1:]] - inter)
        inds = np.where( (sub_ovr <= thresh) |
                                    (obj_ovr <= thresh) |
                                    (sub_ids[order[1:]] != sub_id) |
                                    (obj_ids[order[1:]] != obj_id) |
                                    (pred_ids[order[1:]] != pred_id) )[0]
        order = order[inds + 1]
    #print(' After: {}'.format(len(keep)))
    return sub_ids[keep], obj_ids[keep], pred_ids[keep], sub_boxes[keep], obj_boxes[keep], keep
