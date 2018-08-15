import numpy as np
import pdb
from ..utils.cython_bbox import bbox_overlaps

def _compute_gt_target(pred_boxes, gt_boxes):
    """
    compute which gt gets mapped to each predicted box
    [Modified from Danfei's implementation. 
    Directly use top-1-score boxes. 
    In Danfei's implementation, per-class-boxes 
    are used.]
    """

    num_boxes = pred_boxes.shape[0]
    # map predicted boxes to ground-truth
    gt_targets = np.zeros(num_boxes).astype(np.int32)
    gt_target_iou = np.zeros(num_boxes)
    gt_target_iou.fill(-1)

    for j in xrange(num_boxes):
        # prepare inputs
        bb = pred_boxes[j].astype(float)
        # # compute max IoU over classes
        # # for c in xrange(1, num_classes):
        # for c in xrange(pred_class_scores.shape[1]):
        #     bb = bbox[4*c:4*(c+1)]
        if gt_boxes.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], bb[0])
            iymin = np.maximum(gt_boxes[:, 1], bb[1])
            ixmax = np.minimum(gt_boxes[:, 2], bb[2])
            iymax = np.minimum(gt_boxes[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                    (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            max_iou_class = np.max(overlaps)
            max_k_class = np.argmax(overlaps)

            # select max over classes
            if max_iou_class > gt_target_iou[j]:
                gt_target_iou[j] = max_iou_class
                gt_targets[j] = max_k_class

    return gt_targets, gt_target_iou


def ground_predictions(boxes, gt_boxes, ovthresh=0.5):
    """
    ground graph predictions onto ground truth annotations
    boxes: predicted boxes
    """

    # get predictions
    num_boxes = boxes.shape[0]

    
    # compute which gt index each roi gets mapped to
    gt_targets, gt_target_iou = _compute_gt_target(boxes, gt_boxes)

    # filter out predictions with low IoUs
    filter_inds = np.where(gt_target_iou > ovthresh)[0]

    # make sure each gt box is referenced only once
    # if referenced more than once, use the one that
    # has the maximum IoU
    gt_to_pred = {} # {gt_ind: pred_ind}
    for j in xrange(num_boxes):
        gti = gt_targets[j] # referenced gt ind
        if gti in gt_to_pred:
            pred_ind = gt_to_pred[gti]
            if gt_target_iou[j] > gt_target_iou[pred_ind]:
                gt_to_pred[gti] = j
        elif j in filter_inds: # also must survive filtering
            gt_to_pred[gti] = j

    return gt_to_pred

def expand_relationships_mat(objects, relationships):
    rel_sub_idx, rel_obj_idx = np.where(relationships > 0) # ground truth number
    sub = objects[rel_sub_idx, :5]
    obj = objects[rel_obj_idx, :5]
    rel = relationships[rel_sub_idx, rel_obj_idx]
    return sub, obj, rel, rel_sub_idx, rel_obj_idx
def expand_relationships_list(objects, relationships):
    relationships = np.array(relationships, dtype=np.int)
    sub = objects[relationships[:, 0]][:, :5]
    obj = objects[relationships[:, 1]][:, :5]
    rel = relationships[:, 2]
    return sub, obj, rel 


def check_recalled_graph(gt_objects, gt_relationships, 
        pred_objects, pred_relationships, thres=0.5):
    # rearrange the ground truth
    gt_sub, gt_obj, gt_rel,gt_sub_assign, gt_obj_assign  = expand_relationships_mat(gt_objects, gt_relationships)
    pred_sub, pred_obj, pred_rel,_, _  = expand_relationships_mat(pred_objects, pred_relationships)
    rec_rel = np.zeros_like(gt_relationships)
    # compute the overlap
    try:
        sub_overlaps = bbox_overlaps(
            np.ascontiguousarray(pred_sub[:, :4], dtype=np.float),
            np.ascontiguousarray(gt_sub[:, :4], dtype=np.float))
        obj_overlaps = bbox_overlaps(
            np.ascontiguousarray(pred_obj[:, :4], dtype=np.float),
            np.ascontiguousarray(gt_obj[:, :4], dtype=np.float))
    except:
        print('[Warning] No relationship remaining.')
        return gt_objects, gt_relationships


    for gt_id in xrange(gt_sub.shape[0]):
        fg_candidate = np.where(np.logical_and(
            sub_overlaps[:, gt_id] >= thres, 
            obj_overlaps[:, gt_id] >= thres))[0]
        
        for candidate_id in fg_candidate:
            if pred_rel[candidate_id] == gt_rel[gt_id] and \
                   pred_sub[candidate_id, 4] == gt_sub[gt_id, 4] and \
                   pred_obj[candidate_id, 4] == gt_obj[gt_id, 4]:
                
                rec_rel[gt_sub_assign[gt_id], gt_obj_assign[gt_id]] = gt_rel[gt_id]
                break

    rec_sub, rec_obj = np.where(rec_rel > 0)    
    rec_objects = np.union1d(rec_sub, rec_obj)

    return gt_objects[rec_objects], rec_rel[rec_objects][:,rec_objects]
