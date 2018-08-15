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
from .proposal_target_layer_v1 import _sample_rois, _setup_connection, _get_bbox_regression_labels, graph_construction
 

# <<<< obsolete

DEBUG = False


def proposal_target_layer(object_rois, gt_objects, gt_relationships, gt_regions,
                            num_classes, voc_sign):

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
    region_labels, bbox_targets_region, bbox_inside_weight_region, region_assignments = \
                    _sample_regions(region_selected_rois, gt_regions, voc_sign)

    # assert region_labels.shape[1] == cfg.TRAIN.LANGUAGE_MAX_LENGTH
    object_labels = object_labels.reshape(-1, 1)
    phrase_labels = phrase_labels.reshape(-1, 1)
    object_fg_duplicate = np.stack([object_fg_indicator, object_fg_duplicate], axis=1)

    return (object_labels, object_selected_rois, object_bbox_targets, object_bbox_inside_weights, mat_object, object_fg_duplicate), \
            (phrase_labels, mat_relationship), \
            (region_selected_rois[:, :5], mat_region), \
            (region_labels, bbox_targets_region, bbox_inside_weight_region, region_assignments)





def _sample_regions(region_rois, gt_regions, voc_sign):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_regions)
    overlaps_gt = bbox_overlaps(
        np.ascontiguousarray(region_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_regions[:, :4], dtype=np.float))
    # gt_assignment = overlaps_gt.argmax(axis=1)
    max_overlaps_gt = overlaps_gt.max(axis=1)
    # labels = gt_regions[gt_assignment, 4:]
    fg_inds = np.where(max_overlaps_gt >= cfg.TRAIN.FG_THRESH_REGION)[0]
    bg_inds = np.where(
        (max_overlaps_gt < cfg.TRAIN.BG_THRESH_HI_REGION) & (max_overlaps_gt >= cfg.TRAIN.BG_THRESH_LO_REGION))[0]

    # ## Debug Codes
    # print('fg: {} v.s. bg:{}'.format(len(fg_inds), len(bg_inds)))
    # gt_hit_overlap = overlaps_gt.max(axis=0)
    # hit_ids = np.unique(np.where(gt_hit_overlap >= cfg.TRAIN.FG_THRESH_REGION)[0])
    # print('Recall: {} ({}/{})'.format(
    #     float(len(hit_ids)) / len(gt_regions), len(hit_ids), len(gt_regions)))
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = np.ones((len(keep_inds), gt_regions.shape[1] - 4), dtype=np.int64) * voc_sign['end']
    # Here we randomly select regions overlapped with proposed ROI more than 0.7
    gt_assignment = np.zeros(len(fg_inds), dtype=np.int64)
    for i in range(len(fg_inds)):
        gt_assignment[i] = npr.choice(np.where(overlaps_gt[fg_inds[i]] > cfg.TRAIN.FG_THRESH_REGION)[0], size=1)
        labels[i] = gt_regions[gt_assignment[i], 4:]

    # add start label to background and padding them with <end> sign
    labels[len(fg_inds):, 0] = voc_sign['start']
    rois = region_rois[keep_inds]

    targets_fg = bbox_transform(rois[:len(fg_inds), 1:5], gt_regions[gt_assignment, :4])
    bbox_inside_weights_fg = np.ones(targets_fg.shape, dtype=np.float32) * cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    targets_bg = np.zeros((bg_inds.size, targets_fg.shape[1]), dtype=np.float32)
    bbox_inside_weight_bg = np.zeros(targets_bg.shape, dtype=np.float32)
    bbox_targets = np.vstack([targets_fg, targets_bg])
    bbox_inside_weight = np.vstack([bbox_inside_weights_fg, bbox_inside_weight_bg])

    return labels, bbox_targets, bbox_inside_weight, keep_inds


