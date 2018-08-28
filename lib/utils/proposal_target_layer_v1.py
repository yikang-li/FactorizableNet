# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
import pdb
import time

from ..utils.cython_bbox import bbox_overlaps, bbox_intersections

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from options.config_FN import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

from ..fast_rcnn.nms_wrapper import nms
from models.HDN_v2.utils import nms_detections

# <<<< obsolete

DEBUG = False


def merge_gt_rois(object_rois, gt_rois, thresh=0.5):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(object_rois, dtype=np.float),
        np.ascontiguousarray(gt_rois, dtype=np.float))
    max_overlaps = overlaps.max(axis=1)
    keep_inds = np.where(max_overlaps < thresh)[0]
    rois = np.concatenate((gt_rois, object_rois[keep_inds]), 0)
    rois = rois[:len(object_rois)]
    return rois

def graph_construction(object_rois, gt_rois=None): # if use GT boxes, we merge the GT boxes with generated ROIs

    # pdb.set_trace()
    #     object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    #     region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    object_roi_num = min(cfg.TEST.BBOX_NUM, object_rois.shape[0])
    object_rois = object_rois[:object_roi_num]

    if gt_rois is not None:
        object_rois = merge_gt_rois(object_rois, gt_rois) # to make the message passing more likely to training
        sub_assignment, obj_assignment, _ = _generate_pairs(range(len(gt_rois)))
    else:
        sub_assignment=None
        obj_assignment=None
    object_rois, region_rois, mat_object, mat_relationship, mat_region = _setup_connection(object_rois,
            nms_thres=cfg.TEST.REGION_NMS_THRES,
            sub_assignment_select=sub_assignment,
            obj_assignment_select=obj_assignment)

    if DEBUG:
        print ('[Testing] Region ROI num: {0:d}'.format(region_rois.shape[0]))

    return object_rois, region_rois, mat_object, mat_relationship, mat_region


def proposal_target_layer(object_rois, gt_objects, gt_relationships, num_classes):

    #     object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    #     region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    #     gt_objects:   (G_obj, 5) [x1 ,y1 ,x2, y2, obj_class] float
    #     gt_relationships: (G_obj, G_obj) [pred_class] int (-1 for no relationship)
    #     gt_regions:   (G_region, 4+40) [x1, y1, x2, y2, word_index] (imdb.eos for padding)
    #     # gt_ishard: (G_region, 4+40) {0 | 1} 1 indicates hard
    #     # dontcare_areas: (D, 4) [ x1, y1, x2, y2]
    #     n_classes_obj
    #     n_classes_pred
    #     is_training to indicate whether in training scheme

    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    #####################
    ## sample object ROIs ##
    #####################
    num_images = 1
    object_rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    object_keep_inds, object_gt_assignment, object_fg_indicator, object_fg_duplicate = \
            _sample_rois(object_rois[:, 1:5], gt_objects[:, :4], object_rois_per_image, cfg.TRAIN.FG_FRACTION)
    object_labels = gt_objects[object_gt_assignment, 4]
    # Clamp labels for the background RoIs to 0
    object_labels[np.logical_not(object_fg_indicator)] = 0
    object_selected_rois = object_rois[object_keep_inds]


    object_bbox_targets_temp = bbox_transform(object_selected_rois[:, 1:5], gt_objects[object_gt_assignment, :4])
    object_bbox_target_data = np.hstack(
        (object_labels[:, np.newaxis], object_bbox_targets_temp)).astype(np.float32, copy=False)
    object_bbox_targets, object_bbox_inside_weights = \
        _get_bbox_regression_labels(object_bbox_target_data, num_classes)

    ######################
    ## sample relationships ##
    ######################


    rel_per_image = int(cfg.TRAIN.BATCH_SIZE_RELATIONSHIP / num_images)
    rel_bg_num = rel_per_image
    object_fg_inds = object_keep_inds[object_fg_indicator]
    if object_fg_inds.size > 0:
        id_i, id_j = np.meshgrid(xrange(object_fg_inds.size), xrange(object_fg_inds.size), indexing='ij') # Grouping the input object rois
        id_i = id_i.reshape(-1)
        id_j = id_j.reshape(-1)
        pair_labels = gt_relationships[object_gt_assignment[id_i], object_gt_assignment[id_j]]
        fg_id_rel = np.where(pair_labels > 0)[0]
        rel_fg_num = fg_id_rel.size
        rel_fg_num = int(min(np.round(rel_per_image * cfg.TRAIN.FG_FRACTION_RELATIONSHIP), rel_fg_num))
        # print 'rel_fg_num'
        # print rel_fg_num
        if rel_fg_num > 0:
            fg_id_rel = npr.choice(fg_id_rel, size=rel_fg_num, replace=False)
        else:
            fg_id_rel = np.empty(0, dtype=int)
        rel_labels_fg = pair_labels[fg_id_rel]
        sub_assignment_fg = id_i[fg_id_rel]
        obj_assignment_fg = id_j[fg_id_rel]
        sub_list_fg = object_fg_inds[sub_assignment_fg]
        obj_list_fg = object_fg_inds[obj_assignment_fg]
        rel_bg_num = rel_per_image - rel_fg_num

    phrase_labels = np.zeros(rel_bg_num, dtype=np.float)
    sub_assignment = npr.choice(xrange(object_keep_inds.size), size=rel_bg_num, replace=True)
    obj_assignment = npr.choice(xrange(object_keep_inds.size), size=rel_bg_num, replace=True)
    if (sub_assignment == obj_assignment).any(): # an ugly hack for the issue
        obj_assignment[sub_assignment == obj_assignment] = (obj_assignment[sub_assignment == obj_assignment] + 1) % object_keep_inds.size



    if object_fg_inds.size > 0:
        phrase_labels = np.append(rel_labels_fg, phrase_labels, )
        sub_assignment = np.append(sub_assignment_fg, sub_assignment,)
        obj_assignment = np.append(obj_assignment_fg, obj_assignment, )

    object_selected_rois, region_selected_rois, mat_object, mat_relationship, mat_region = \
            _setup_connection(object_selected_rois,  nms_thres=cfg.TRAIN.REGION_NMS_THRES,
                                                sub_assignment_select = sub_assignment,
                                                obj_assignment_select = obj_assignment)
    #print '[Training] Region ROI num: {0:d}'.format(region_selected_rois.shape[0])

    # assert region_labels.shape[1] == cfg.TRAIN.LANGUAGE_MAX_LENGTH
    object_labels = object_labels.reshape(-1, 1)
    phrase_labels = phrase_labels.reshape(-1, 1)
    object_fg_duplicate = np.stack([object_fg_indicator, object_fg_duplicate], axis=1)

    return (object_labels, object_selected_rois, object_bbox_targets, object_bbox_inside_weights, mat_object, object_fg_duplicate), \
              (phrase_labels, mat_relationship), \
              (region_selected_rois[:, :5], mat_region) \





def _sample_rois(rois, gt_rois, rois_per_image, fg_frac):
    '''
    include_last_one: set to make sure the last one ROI is included. Set for region ROI sampling
    to make sure the entire image is included
    '''

    assert rois.shape[1] == 4, 'Shape mis-match: [{}, {}] v.s. [:, 4]'.format(rois.shape[0], rois.shape[1])
    fg_rois_per_image = np.round(fg_frac * rois_per_image)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois, dtype=np.float),
        np.ascontiguousarray(gt_rois, dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    if bg_inds.size == 0:
        bg_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH_HI)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # fg_rois_per_this_image = int(min(bg_inds.size, fg_inds.size))
    # Sample foreground regions without replacement
    fg_inds_original = fg_inds[:]
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # bg_rois_per_this_image = fg_rois_per_this_image
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    if len(fg_inds) > 0:
        fg_overlap = max_overlaps[fg_inds]
        fg_sorted = np.argsort(fg_overlap)[::-1]
        fg_inds = fg_inds[fg_sorted]

    keep_inds = np.append(fg_inds, bg_inds)
    gt_assignment = gt_assignment[keep_inds]
    fg_indicator = np.zeros(len(keep_inds), dtype=np.bool)
    fg_indicator[:len(fg_inds)] = True

    # for duplicate removal
    _, highest_inds = np.unique(gt_assignment[:len(fg_inds)], return_index=True)
    fg_duplicate = np.zeros(len(keep_inds))
    fg_duplicate[highest_inds] = 1


    return keep_inds, gt_assignment, fg_indicator, fg_duplicate


def _setup_connection(object_rois,  nms_thres=0.6, sub_assignment_select = None, obj_assignment_select = None):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    # t_start = time.time()
    sub_assignment, obj_assignment, rel_assignment = _generate_pairs(range(object_rois.shape[0]), sub_assignment_select, obj_assignment_select)
    region_rois = box_union(object_rois[sub_assignment], object_rois[obj_assignment])
    mapping = nms(region_rois[:, 1:].astype(np.float32), nms_thres, retain_all=True)

    keep, keep_inverse = np.unique(mapping, return_inverse=True)
    selected_region_rois = region_rois[keep, :5]

    mat_region = np.zeros((len(keep), object_rois.shape[0]), dtype=np.int64)
    mat_relationship = np.zeros((len(rel_assignment), 3), dtype=np.int64)
    mat_relationship[:, 0] = sub_assignment[rel_assignment]
    mat_relationship[:, 1] = obj_assignment[rel_assignment]
    mat_relationship[:, 2] = keep_inverse[rel_assignment]

    for relationship_id, region_id in enumerate(keep_inverse):
        mat_region[region_id, sub_assignment[relationship_id]] +=1
        mat_region[region_id, obj_assignment[relationship_id]] +=1

    mat_region = mat_region.astype(np.bool, copy=False)
    mat_object = mat_region.transpose()


    return object_rois[:, :5], selected_region_rois, mat_object, mat_relationship, mat_region

def box_union(box1, box2):
    return np.concatenate((
        np.minimum(box1[:, :3], box2[:, :3]), # id, x1, y1
        np.maximum(box1[:, 3:5], box2[:, 3:5]), # id, x2, y2
        box1[:, [5]] * box2[:, [5]]
        ), 1)




def _generate_pairs(ids, sub_assignment_select = None, obj_assignment_select = None):
    id_i, id_j = np.meshgrid(ids, ids, indexing='ij') # Grouping the input object rois
    id_i = id_i.reshape(-1)
    id_j = id_j.reshape(-1)
    # removing diagonal items
    # VG-DR-Net has self-relations: e.g. A-relation-A, so we comment these codes
    # id_num = len(ids)
    # diagonal_items = np.array(range(id_num))
    # diagonal_items = diagonal_items * id_num + diagonal_items
    # all_id = range(len(id_i))
    # selected_id = np.setdiff1d(all_id, diagonal_items)
    # id_i = id_i[selected_id]
    # id_j = id_j[selected_id]
    if sub_assignment_select is not None and obj_assignment_select is not None:
        # For diagnoal-items-removed
        # rel_assignment = sub_assignment_select * (len(ids) - 1) + obj_assignment_select - (obj_assignment_select > sub_assignment_select).astype(np.int)
        rel_assignment = sub_assignment_select * len(ids) + obj_assignment_select
    else:
        rel_assignment = range(len(id_i))

    return id_i, id_j, rel_assignment


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = [1.0, 1.0, 1.0, 1.0]
    return bbox_targets, bbox_inside_weights
