import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from lib.visualize_graph.vis_utils import expand_relationships_mat, expand_relationships_list
from .cython_bbox import bbox_overlaps, bbox_intersections

def recall(rois, gt_objects, top_N, thres):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[:top_N, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))

    overlap_gt = np.amax(overlaps, axis=0)
    correct_cnt = np.sum(overlap_gt >= thres)
    total_cnt = overlap_gt.size 
    return correct_cnt, total_cnt

def check_recall(rois, gt_objects, top_N, thres=0.5):

    rois = rois.cpu().data.numpy()
    if isinstance(gt_objects, list):
        correct_cnt, total_cnt = 0, 0
        for i, gt in enumerate(gt_objects):
            im_rois = rois[np.where(rois[:, 0] == i)[0]]
            r = recall(im_rois, gt, top_N, thres)
            correct_cnt += r[0]
            total_cnt += r[1]
        return correct_cnt, total_cnt
    else:
        return recall(rois, gt_objects, top_N, thres)

def get_phrase_boxes(sub_boxes, obj_boxes):
    phrase_boxes = [np.minimum(sub_boxes[:, 0], obj_boxes[:, 0]), 
                    np.minimum(sub_boxes[:, 1], obj_boxes[:, 1]),
                    np.maximum(sub_boxes[:, 2], obj_boxes[:, 2]),
                    np.maximum(sub_boxes[:, 3], obj_boxes[:, 3])]
    phrase_boxes = np.stack(phrase_boxes, axis=1)
    return phrase_boxes

def check_phrase_recall(gt_objects, gt_relationships, 
        subject_inds, object_inds, predicate_inds, 
        subject_boxes, object_boxes, top_Ns, thres=0.5):
    # rearrange the ground truth
    gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0) # ground truth number
    gt_sub = gt_objects[gt_rel_sub_idx, :5]
    gt_obj = gt_objects[gt_rel_obj_idx, :5]
    gt_rel = gt_relationships[gt_rel_sub_idx, gt_rel_obj_idx]

    rel_cnt = len(gt_rel)
    rel_correct_cnt = np.zeros(len(top_Ns))
    max_topN = max(top_Ns)

    # compute the overlap
    try:
        phrase_overlaps = bbox_overlaps(
            np.ascontiguousarray(
                get_phrase_boxes(subject_boxes[:max_topN], object_boxes[:max_topN]), dtype=np.float),
            np.ascontiguousarray(
                get_phrase_boxes(gt_sub[:, :4], gt_obj[:, :4]), dtype=np.float))
    except:
        print('[Warning] No relationship remaining.')
        return rel_cnt, rel_correct_cnt


    for idx, top_N in enumerate(top_Ns):
        for gt_id in xrange(rel_cnt):
            fg_candidate = np.where(phrase_overlaps[:top_N, gt_id] >= thres)[0]
            
            for candidate_id in fg_candidate:
                if predicate_inds[candidate_id] == gt_rel[gt_id] and \
                        subject_inds[candidate_id] == gt_sub[gt_id, 4] and \
                        object_inds[candidate_id] == gt_obj[gt_id, 4]:
                    rel_correct_cnt[idx] += 1 
                    break
    return rel_cnt, rel_correct_cnt


def check_relationship_recall(gt_objects, gt_relationships, 
        subject_inds, object_inds, predicate_inds, 
        subject_boxes, object_boxes, top_Ns, thres=0.5):
    # rearrange the ground truth
    gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0) # ground truth number
    gt_sub = gt_objects[gt_rel_sub_idx, :5]
    gt_obj = gt_objects[gt_rel_obj_idx, :5]
    gt_rel = gt_relationships[gt_rel_sub_idx, gt_rel_obj_idx]

    rel_cnt = len(gt_rel)
    pred_correct_cnt = np.zeros(len(top_Ns))
    rel_correct_cnt = np.zeros(len(top_Ns))
    max_topN = max(top_Ns)

    # compute the overlap
    try:
        sub_overlaps = bbox_overlaps(
            np.ascontiguousarray(subject_boxes[:max_topN], dtype=np.float),
            np.ascontiguousarray(gt_sub[:, :4], dtype=np.float))
        obj_overlaps = bbox_overlaps(
            np.ascontiguousarray(object_boxes[:max_topN], dtype=np.float),
            np.ascontiguousarray(gt_obj[:, :4], dtype=np.float))
    except:
        print('[Warning] No relationship remaining.')
        return rel_cnt, rel_correct_cnt, pred_correct_cnt


    for idx, top_N in enumerate(top_Ns):
        for gt_id in xrange(rel_cnt):
            fg_candidate = np.where(np.logical_and(
                sub_overlaps[:top_N, gt_id] >= thres, 
                obj_overlaps[:top_N, gt_id] >= thres))[0]
            
            pred_flag = 1
            for candidate_id in fg_candidate:
                if predicate_inds[candidate_id] == gt_rel[gt_id]:
                    pred_correct_cnt[idx] += pred_flag
                    pred_flag = 0 # only add once
                    if subject_inds[candidate_id] == gt_sub[gt_id, 4] and \
                            object_inds[candidate_id] == gt_obj[gt_id, 4]:
                        
                        rel_correct_cnt[idx] += 1 
                        break
    return rel_cnt, rel_correct_cnt, pred_correct_cnt


def check_hit_detections(gt_objects, gt_relationships, 
        pred_objects, pred_relationships, thres=0.5):

    
    # rearrange the ground truth
    gt_sub, gt_obj, gt_rel,_, _  = expand_relationships_mat(gt_objects, gt_relationships)
    pred_sub, pred_obj, pred_rel  = expand_relationships_list(pred_objects, pred_relationships)
    hit_pred = np.zeros_like(pred_rel)
    assigned_gt = np.ones_like(gt_rel)
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
        return hit_pred


    
    for pred_id in xrange(pred_rel.shape[0]):

        fg_candidate = np.where(np.logical_and(
            sub_overlaps[pred_id] >= thres, 
            obj_overlaps[pred_id] >= thres))[0]
        for candidate_id in fg_candidate:
            if pred_rel[pred_id] == gt_rel[candidate_id] and \
                   pred_sub[pred_id, 4] == gt_sub[candidate_id, 4] and \
                   pred_obj[pred_id, 4] == gt_obj[candidate_id, 4] and assigned_gt[candidate_id]:
                
                hit_pred[pred_id] = 1
                assigned_gt[candidate_id] = 0
                break

    return hit_pred

