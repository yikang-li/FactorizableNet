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
from .proposal_target_layer_v1 import merge_gt_rois, _sample_rois, _get_bbox_regression_labels, _generate_pairs, box_union

# <<<< obsolete

DEBUG = False

def graph_construction(object_rois, caption_rois, gt_rois=None): # if use GT boxes, we merge the GT boxes with generated ROIs

    #     object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    #     region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    object_roi_num = min(cfg.TEST.BBOX_NUM, object_rois.shape[0])
    object_rois = object_rois[:object_roi_num]
    caption_roi_num = min(cfg.TEST.REGION_NUM, caption_rois.shape[0])
    caption_rois = caption_rois[:caption_roi_num]

    if gt_rois is not None:
        object_rois = merge_gt_rois(object_rois, gt_rois)
        sub_assignment, obj_assignment, _ = _generate_pairs(range(len(gt_rois)))
    else:
        sub_assignment=None
        obj_assignment=None
    object_rois, region_rois, mat_object, mat_relationship, \
        mat_region, mat_caption_object, mat_caption_phrase = _setup_connection(
            object_rois, caption_rois,
            nms_thres=cfg.TEST.REGION_NMS_THRES,
            coverage_thres=cfg.TEST.CAPTION_COVERAGE_THRES,
            sub_assignment_select=sub_assignment,
            obj_assignment_select=obj_assignment)

    return object_rois, region_rois, caption_rois, \
            mat_object, mat_relationship, mat_region, mat_caption_object, mat_caption_phrase


def proposal_target_layer(object_rois, caption_rois,
                            gt_objects, gt_relationships, gt_regions,
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

    ########################
    ## sample region ROIs ##
    ########################
    # To include an extra ROI containing all object ROIs
    caption_rois_per_image = cfg.TRAIN.BATCH_SIZE_CAPTION / num_images
    caption_keep_inds, caption_gt_assignment, caption_fg_indicator = \
            _sample_rois(caption_rois[:, 1:5], gt_regions[:, :4],
                caption_rois_per_image, cfg.TRAIN.FG_FRACTION_CAPTION)

    caption_labels = gt_regions[caption_gt_assignment, 4:]
    # Clamp labels for the background RoIs to 0
    caption_labels[np.logical_not(caption_fg_indicator)] = voc_sign['end']
    caption_labels[np.logical_not(caption_fg_indicator), 0] = voc_sign['start']
    caption_selected_rois = caption_rois[caption_keep_inds]
    caption_bbox_targets = bbox_transform(
        caption_selected_rois[:, 1:5],
        gt_regions[caption_gt_assignment, :4])
    caption_bbox_inside_weights = np.ones(caption_bbox_targets.shape, dtype=np.float32) * cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    caption_bbox_inside_weights[np.logical_not(caption_fg_indicator)] = 0

    object_selected_rois, region_selected_rois, \
        mat_object, mat_relationship, mat_region, \
        mat_caption_object, mat_caption_phrase = \
            _setup_connection(object_selected_rois,  caption_selected_rois,
                                nms_thres=cfg.TRAIN.REGION_NMS_THRES,
                                coverage_thres=cfg.TRAIN.CAPTION_COVERAGE_THRES,
                                sub_assignment_select = sub_assignment,
                                obj_assignment_select = obj_assignment)

    # assert region_labels.shape[1] == cfg.TRAIN.LANGUAGE_MAX_LENGTH
    object_labels = object_labels.reshape(-1, 1)
    phrase_labels = phrase_labels.reshape(-1, 1)
    object_fg_duplicate = np.stack([object_fg_indicator, object_fg_duplicate], axis=1)

    return (object_labels, object_selected_rois, object_bbox_targets, object_bbox_inside_weights, mat_object, object_fg_duplicate), \
            (phrase_labels, mat_relationship), \
            (region_selected_rois[:, :5], mat_region), \
            (caption_selected_rois, caption_labels, caption_bbox_targets, caption_bbox_inside_weights, mat_caption_object, mat_caption_phrase)






def _setup_connection(object_rois,  caption_rois,
            nms_thres=0.6, coverage_thres=0.8,
            sub_assignment_select = None,
            obj_assignment_select = None):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    # t_start = time.time()
    sub_assignment, obj_assignment, rel_assignment = _generate_pairs(
        range(object_rois.shape[0]), sub_assignment_select, obj_assignment_select)
    region_rois = box_union(object_rois[sub_assignment], object_rois[obj_assignment])
    mapping = nms(region_rois[:, 1:], nms_thres, retain_all=True)

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

    caption_coverage_region = bbox_intersections(
        np.ascontiguousarray(caption_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(selected_region_rois[:, 1:5], dtype=np.float))
    mat_caption_phrase = (caption_coverage_region > coverage_thres).astype(np.int64)

    caption_coverage_object = bbox_intersections(
        np.ascontiguousarray(caption_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(object_rois[:, 1:5], dtype=np.float))
    mat_caption_object = (caption_coverage_object > coverage_thres).astype(np.int64)


    return object_rois[:, :5], selected_region_rois, \
                mat_object, mat_relationship, mat_region, \
                mat_caption_object, mat_caption_phrase

