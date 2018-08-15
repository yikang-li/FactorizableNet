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
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = False



def graph_construction(object_rois, region_rois):

    #     object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    #     region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    object_roi_num = min(cfg.TEST.BBOX_NUM, object_rois.shape[0])
    object_rois = object_rois[:object_roi_num]

    region_roi_entire = np.concatenate((np.amin(object_rois[:, 0:3], 0), np.amax(object_rois[:, 3:5], 0)), 0)
    region_rois = np.vstack((region_roi_entire, region_rois))
    region_roi_num = min(cfg.TEST.REGION_NUM, region_rois.shape[0])
    region_rois = region_rois[:region_roi_num]
    
    sub_assignment, obj_assignment = _generate_pairs(range(object_rois.shape[0]))

    mat_object, mat_relationship, mat_region = _setup_connection(object_rois, region_rois, 
                sub_assignment, obj_assignment, coverage_thres=0.7)

    return object_rois, region_rois, mat_object, mat_relationship, mat_region


def proposal_target_layer(object_rois, region_rois, gt_objects, gt_relationships, 
                gt_regions, num_classes, voc_eos):

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
    ########################
    ## sample object ROIs ##
    ########################
    num_images = 1
    object_rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    object_keep_inds, object_gt_assignment, object_fg_indicator = \
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

    ########################
    ## sample region ROIs ##
    ########################
    # To include an extra ROI containing all object ROIs
    region_roi_entire = np.concatenate((np.amin(object_rois[:, 0:3], 0), np.amax(object_rois[:, 3:5], 0)), 0)
    region_rois = np.vstack((region_roi_entire, region_rois))

    region_rois_per_image = cfg.TRAIN.BATCH_SIZE_REGION / num_images
    region_keep_inds, region_gt_assignment, region_fg_indicator = \
            _sample_rois(region_rois[:, 1:5], gt_regions[:, :4], region_rois_per_image, cfg.TRAIN.FG_FRACTION_REGION, 
                include_first_one=True)

    region_labels = np.zeros((len(region_keep_inds), gt_regions.shape[1] - 4), dtype=np.int64)
    region_labels = gt_regions[region_gt_assignment, 4:]
    # add start label to background and padding them with <end> sign
    region_labels[np.logical_not(region_fg_indicator), 0].fill(voc_eos['start'])
    region_labels[np.logical_not(region_fg_indicator), 1:].fill(voc_eos['end'])
    region_selected_rois = region_rois[region_keep_inds]
    region_bbox_targets = bbox_transform(
            region_selected_rois[:, 1:5], gt_regions[region_gt_assignment, :4])

    region_bbox_inside_weights = np.zeros(region_bbox_targets.shape, dtype=np.float32)
    region_bbox_inside_weights[region_fg_indicator] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    
    ##########################
    ## sample relationships ##
    ##########################

    
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
    

    if object_fg_inds.size > 0:
        phrase_labels = np.append(rel_labels_fg, phrase_labels, )
        sub_assignment = np.append(sub_assignment_fg, sub_assignment,)
        obj_assignment = np.append(obj_assignment_fg, obj_assignment, )

    mat_object, mat_relationship, mat_region = _setup_connection(object_selected_rois, region_selected_rois, 
                sub_assignment, obj_assignment, coverage_thres=0.7)


    # assert region_labels.shape[1] == cfg.TRAIN.LANGUAGE_MAX_LENGTH
    object_labels = object_labels.reshape(-1, 1)
    phrase_labels = phrase_labels.reshape(-1, 1)

    assert object_rois.shape[1] == 5

    return object_labels, object_selected_rois, object_bbox_targets, object_bbox_inside_weights, mat_object, \
            phrase_labels, mat_relationship, \
            region_labels, region_selected_rois, \
            region_bbox_targets, region_bbox_inside_weights, mat_region \



    

def _sample_rois(rois, gt_rois, rois_per_image, fg_frac, include_first_one=False):
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
    keep_inds = np.append(fg_inds, bg_inds)
    if include_first_one:
        if 0 not in fg_inds:
            if 0 in fg_inds_original:
                keep_inds[0] = 0
            elif 0 not in bg_inds and bg_inds.size:
                keep_inds[-1] = 0
    gt_assignment = gt_assignment[keep_inds]
    fg_indicator = np.zeros(len(keep_inds), dtype=np.bool)
    fg_indicator[:len(fg_inds)] = True
    return keep_inds, gt_assignment, fg_indicator


def _setup_connection(object_rois, region_rois, sub_assignment, obj_assignment, 
                        coverage_thres=0.5, fast_mode=False):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    # t_start = time.time()
    region_object_coverage = bbox_intersections(
        np.ascontiguousarray(region_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(object_rois[:, 1:5], dtype=np.float))

    mat_region = (region_object_coverage > coverage_thres).astype(np.int64)
    mat_object = mat_region.transpose()

    region_relationship_coverage = np.logical_and(
        bbox_intersections(
            np.ascontiguousarray(region_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(object_rois[sub_assignment, 1:5], dtype=np.float)) > coverage_thres, 
        bbox_intersections(
            np.ascontiguousarray(region_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(object_rois[obj_assignment, 1:5], dtype=np.float)) > coverage_thres

        )
    # t_middle = time.time()
    mat_relationship = np.zeros((len(sub_assignment), 3), dtype=np.int64)
    mat_relationship[:, 0] = sub_assignment
    mat_relationship[:, 1] = obj_assignment

    if fast_mode:
        for i in range(len(sub_assignment)):
            # find the largest region to inference relationship
            mat_relationship[i, 2] = np.random.choice(np.where(region_relationship_coverage[:, i])[0], 1)[0]
    else:
        for i in range(len(sub_assignment)):
            # find the largest region to inference relationship
            region_ids = np.where(region_relationship_coverage[:, i])
            region_areas = (region_rois[region_ids, 4] - region_rois[region_ids, 2]) * \
                                (region_rois[region_ids, 3] - region_rois[region_ids, 1])
            mat_relationship[i, 2] = np.argmin(region_areas)



    # t_end = time.time()
    # print 'pre: {0:.3f}; post: {1:.3f}'.format(t_middle - t_start, t_end - t_middle)

    return mat_object, mat_relationship, mat_region

def box_union(box1, box2):
    return np.concatenate((np.minimum(box1[:, :3], box2[:, :3]), np.maximum(box1[:, 3:], box2[:, 3:])), 1)




def _generate_pairs(ids):
    id_i, id_j = np.meshgrid(ids, ids, indexing='ij') # Grouping the input object rois
    id_i = id_i.reshape(-1) 
    id_j = id_j.reshape(-1)
    # remove the diagonal items
    id_num = len(ids)
    diagonal_items = np.array(range(id_num))
    diagonal_items = diagonal_items * id_num + diagonal_items
    all_id = range(len(id_i))
    selected_id = np.setdiff1d(all_id, diagonal_items)
    id_i = id_i[selected_id]
    id_j = id_j[selected_id]

    return id_i, id_j


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
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights