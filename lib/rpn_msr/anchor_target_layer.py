# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
import numpy as np
import numpy.random as npr

from .generate_anchors import generate_anchors
from ..utils.cython_bbox import bbox_overlaps, bbox_intersections

import pdb

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from lib.fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = False
_fg_sum = 0.
_bg_sum = 0.
_count = 0.


def anchor_target_layer(img, gt_boxes, im_info, _feat_stride, 
            rpn_opts, mappings):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    ----------
    rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
    gt_boxes: (G, 4) vstack of [x1, y1, x2, y2]
    #gt_ishard: (G, 1), 1 or 0 indicates difficult or not
    #dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    _anchors = generate_anchors(scales=rpn_opts['anchor_scales'], ratios=rpn_opts['anchor_ratios'])
    opts = rpn_opts['train']
    _num_anchors = _anchors.shape[0]
    full_height, full_width = mappings[int(img.size(1))], mappings[int(img.size(2))]

    # allow boxes to sit over the edge by a small amount
    _allowed_border = opts['allowed_border']

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, full_width) * _feat_stride
    shift_y = np.arange(0, full_height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    
    # map of shape (..., H, W)
    # height, width = rpn_cls_score.shape[1:3]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    # map of shape (..., H, W)
    # pytorch (bs, c, h, w)
    height = mappings[int(im_info[0])]
    width = mappings[int(im_info[1])]
    valid_mask =np.zeros((full_height, full_width, A), dtype=np.bool)
    valid_mask[:height, :width] = True
    valid_ids = valid_mask.reshape(K*A)
    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border) &  # height
        valid_ids # remove the useless points
    )[0]
    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(np.logical_and(overlaps == gt_max_overlaps, overlaps > 0.))[0] # avoid zero overlap

    if not opts['clobber_positives']:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < opts['negative_overlap']] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= opts['positive_overlap']] = 1
    if opts['clobber_positives']:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < opts['negative_overlap']] = 0

    # preclude dontcare areas

    # subsample positive labels if we have too many
    num_fg = int(opts['fg_fraction'] * opts['batch_size'])
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    else:
        num_fg = len(fg_inds)
        
    # subsample negative labels if we have too many
    num_bg = opts['batch_size'] - num_fg
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) < num_bg:
        disable_inds = npr.choice(
            disable_inds, size=(len(disable_inds) - num_bg + len(bg_inds)), replace=False)
    try:
        labels[disable_inds] = -1
    except Exception:
        pass


    #  if DEBUG:
        #  print 'gt_max_overlaps', gt_max_overlaps
        #  print 'gt_boxes.shape[0]', gt_boxes.shape[0]
        #  recall = 1 - len(np.setdiff1d(range(gt_boxes.shape[0]), argmax_overlaps)) / float(gt_boxes.shape[0])
        #  print 'recall: %.3f%%' % (recall * 100)



    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        # print "was %s inds, disabling %s, now %s inds" % (
        # len(bg_inds), len(disable_inds), np.sum(labels == 0))

    # bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # _compute_targets(anchors[fg_inds], gt_boxes[argmax_overlaps[fg_inds], :])
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])


    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(opts['BBOX_INSIDE_WEIGHTS'])

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if opts['POSITIVE_WEIGHT'] < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((opts['POSITIVE_WEIGHT'] > 0) &
                (opts['POSITIVE_WEIGHT'] < 1))
        positive_weights = (opts['POSITIVE_WEIGHT'] /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - opts['POSITIVE_WEIGHT']) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights


    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        global _fg_sum, _bg_sum, _count
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # labels
    labels = labels.reshape((full_height, full_width, A)).transpose(2, 0, 1).reshape(-1)

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((full_height, full_width, A * 4)).transpose(2, 0, 1)

    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((full_height, full_width, A * 4)).transpose(2, 0, 1)
    # assert bbox_inside_weights.shape[2] == height
    # assert bbox_inside_weights.shape[3] == width

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((full_height, full_width, A * 4)).transpose(2, 0, 1)
    # assert bbox_outside_weights.shape[2] == height
    # assert bbox_outside_weights.shape[3] == width

    result = [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]

    return result




# def anchor_target_layer(rpn_cls_score, gt_boxes, dontcare_areas, 
#                         im_infos, _feat_stride, opts, anchor_scales, 
#                         anchor_ratios, mappings):
#     """
#     Assign anchors to ground-truth targets. Produces anchor classification
#     labels and bounding-box regression targets.
#     ----------
#     rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
#     gt_boxes: (G, 4) vstack of [x1, y1, x2, y2]
#     #gt_ishard: (G, 1), 1 or 0 indicates difficult or not
#     #dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
#     im_info: a list of [image_height, image_width, scale_ratios]
#     _feat_stride: the downsampling ratio of feature map to the original input image
#     anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
#     ----------
#     Returns
#     ----------
#     rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
#     rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
#                             that are the regression objectives
#     rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
#     rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
#                             beacuse the numbers of bgs and fgs mays significiantly different
#     """
#     _anchors = generate_anchors(scales=anchor_scales, ratios=anchor_ratios)
#     _num_anchors = _anchors.shape[0]
#     batch_size = rpn_cls_score.shape[0]
#     full_height, full_width = rpn_cls_score.shape[2:4]

#     # allow boxes to sit over the edge by a small amount
#     _allowed_border = opts['allowed_border']

#     # 1. Generate proposals from bbox deltas and shifted anchors
#     shift_x = np.arange(0, full_width) * _feat_stride
#     shift_y = np.arange(0, full_height) * _feat_stride
#     shift_x, shift_y = np.meshgrid(shift_x, shift_y)
#     shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
#                         shift_x.ravel(), shift_y.ravel())).transpose()

#     # add A anchors (1, A, 4) to
#     # cell K shifts (K, 1, 4) to get
#     # shift anchors (K, A, 4)
#     # reshape to (K*A, 4) shifted anchors
#     A = _num_anchors
#     K = shifts.shape[0]
#     all_anchors = (_anchors.reshape((1, A, 4)) +
#                    shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
#     all_anchors = all_anchors.reshape((K * A, 4))
#     total_anchors = int(K * A)

    
#     # map of shape (..., H, W)
#     # height, width = rpn_cls_score.shape[1:3]

#     # Algorithm:
#     #
#     # for each (H, W) location i
#     #   generate 9 anchor boxes centered on cell i
#     #   apply predicted bbox deltas at cell i to each of the 9 anchors
#     # filter out-of-image anchors
#     # measure GT overlap

#     # map of shape (..., H, W)
#     # pytorch (bs, c, h, w)
#     rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = [], [], [], []
#     for i in range(batch_size):
#         im_info = im_infos[i]
#         height = mappings[int(im_info[0])]
#         width = mappings[int(im_info[1])]
#         valid_mask =np.zeros((full_height, full_width, A), dtype=np.bool)
#         valid_mask[:height, :width] = True
#         valid_ids = valid_mask.reshape(K*A)
#         # only keep anchors inside the image
#         inds_inside = np.where(
#             (all_anchors[:, 0] >= -_allowed_border) &
#             (all_anchors[:, 1] >= -_allowed_border) &
#             (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
#             (all_anchors[:, 3] < im_info[0] + _allowed_border) &  # height
#             valid_ids # remove the useless points
#         )[0]
#         # keep only inside anchors
#         anchors = all_anchors[inds_inside, :]

#         # label: 1 is positive, 0 is negative, -1 is dont care
#         labels = np.empty((len(inds_inside),), dtype=np.float32)
#         labels.fill(-1)

#         # overlaps between the anchors and the gt boxes
#         # overlaps (ex, gt)
#         overlaps = bbox_overlaps(
#             np.ascontiguousarray(anchors, dtype=np.float),
#             np.ascontiguousarray(gt_boxes[i], dtype=np.float))
#         argmax_overlaps = overlaps.argmax(axis=1)
#         max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
#         gt_argmax_overlaps = overlaps.argmax(axis=0)
#         gt_max_overlaps = overlaps[gt_argmax_overlaps,
#                                    np.arange(overlaps.shape[1])]
#         gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

#         if not opts['clobber_positives']:
#             # assign bg labels first so that positive labels can clobber them
#             labels[max_overlaps < opts['negative_overlap']] = 0

#         # fg label: for each gt, anchor with highest overlap
#         labels[gt_argmax_overlaps] = 1

#         # fg label: above threshold IOU
#         labels[max_overlaps >= opts['positive_overlap']] = 1
#         if opts['clobber_positives']:
#             # assign bg labels last so that negative labels can clobber positives
#             labels[max_overlaps < opts['negative_overlap']] = 0

#         # preclude dontcare areas
#         if dontcare_areas is not None and dontcare_areas.shape[0] > 0:
#             # intersec shape is D x A
#             intersecs = bbox_intersections(
#                 np.ascontiguousarray(dontcare_areas, dtype=np.float), # D x 4
#                 np.ascontiguousarray(anchors, dtype=np.float) # A x 4
#             )
#             intersecs_ = intersecs.sum(axis=0) # A x 1
#             labels[intersecs_ > opts['dontcare_area_intersection_hi']] = -1

#         # subsample positive labels if we have too many
#         num_fg = int(opts['fg_fraction'] * opts['batch_size'])
#         fg_inds = np.where(labels == 1)[0]
#         if len(fg_inds) > num_fg:
#             disable_inds = npr.choice(
#                 fg_inds, size=(len(fg_inds) - num_fg), replace=False)
#         else:
#             num_fg = len(fg_inds)
            
#         # subsample negative labels if we have too many
#         num_bg = opts['batch_size'] - num_fg
#         bg_inds = np.where(labels == 0)[0]
#         if len(bg_inds) < num_bg:
#             disable_inds = npr.choice(
#                 disable_inds, size=(len(disable_inds) - num_bg + len(bg_inds)), replace=False)
#         try:
#             labels[disable_inds] = -1
#         except Exception:
#             pass


#         #  if DEBUG:
#             #  print 'gt_max_overlaps', gt_max_overlaps
#             #  print 'gt_boxes.shape[0]', gt_boxes.shape[0]
#             #  recall = 1 - len(np.setdiff1d(range(gt_boxes.shape[0]), argmax_overlaps)) / float(gt_boxes.shape[0])
#             #  print 'recall: %.3f%%' % (recall * 100)



#         if len(bg_inds) > num_bg:
#             disable_inds = npr.choice(
#                 bg_inds, size=(len(bg_inds) - num_bg), replace=False)
#             labels[disable_inds] = -1
#             # print "was %s inds, disabling %s, now %s inds" % (
#             # len(bg_inds), len(disable_inds), np.sum(labels == 0))

#         # bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
#         # _compute_targets(anchors[fg_inds], gt_boxes[argmax_overlaps[fg_inds], :])
#         bbox_targets = _compute_targets(anchors, gt_boxes[i][argmax_overlaps, :])


#         bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
#         bbox_inside_weights[labels == 1, :] = np.array(opts['BBOX_INSIDE_WEIGHTS'])

#         bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
#         if opts['POSITIVE_WEIGHT'] < 0:
#             # uniform weighting of examples (given non-uniform sampling)
#             num_examples = np.sum(labels >= 0)
#             positive_weights = np.ones((1, 4)) * 1.0 / num_examples
#             negative_weights = np.ones((1, 4)) * 1.0 / num_examples
#         else:
#             assert ((opts['POSITIVE_WEIGHT'] > 0) &
#                     (opts['POSITIVE_WEIGHT'] < 1))
#             positive_weights = (opts['POSITIVE_WEIGHT'] /
#                                 np.sum(labels == 1))
#             negative_weights = ((1.0 - opts['POSITIVE_WEIGHT']) /
#                                 np.sum(labels == 0))
#         bbox_outside_weights[labels == 1, :] = positive_weights
#         bbox_outside_weights[labels == 0, :] = negative_weights


#         # map up to original set of anchors
#         labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
#         bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
#         bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
#         bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

#         if DEBUG:
#             print 'rpn: max max_overlap', np.max(max_overlaps)
#             print 'rpn: num_positive', np.sum(labels == 1)
#             print 'rpn: num_negative', np.sum(labels == 0)
#             _fg_sum += np.sum(labels == 1)
#             _bg_sum += np.sum(labels == 0)
#             _count += 1
#             print 'rpn: num_positive avg', _fg_sum / _count
#             print 'rpn: num_negative avg', _bg_sum / _count

#         # labels
#         labels = labels.reshape((1, full_height, full_width, A)).transpose(0, 3, 1, 2)
#         labels = labels.reshape((1, 1, A * full_height, full_width))
#         rpn_labels.append(labels.transpose(0, 2, 3, 1).reshape(-1))

#         # bbox_targets
#         bbox_targets = bbox_targets \
#             .reshape((1, full_height, full_width, A * 4)).transpose(0, 3, 1, 2)

#         rpn_bbox_targets.append(bbox_targets)
#         # bbox_inside_weights
#         bbox_inside_weights = bbox_inside_weights \
#             .reshape((1, full_height, full_width, A * 4)).transpose(0, 3, 1, 2)
#         # assert bbox_inside_weights.shape[2] == height
#         # assert bbox_inside_weights.shape[3] == width

#         rpn_bbox_inside_weights.append(bbox_inside_weights)

#         # bbox_outside_weights
#         bbox_outside_weights = bbox_outside_weights \
#             .reshape((1, full_height, full_width, A * 4)).transpose(0, 3, 1, 2)
#         # assert bbox_outside_weights.shape[2] == height
#         # assert bbox_outside_weights.shape[3] == width

#         rpn_bbox_outside_weights.append(bbox_outside_weights)


#     # print 'rpn_labels', rpn_labels
#     # print 'rpn_bbox_targets', rpn_bbox_targets
#     rpn_labels = np.concatenate(rpn_labels, axis=0)
#     rpn_bbox_targets = np.concatenate(rpn_bbox_inside_weights, axis=0)
#     rpn_bbox_inside_weights = np.concatenate(rpn_bbox_inside_weights, axis=0)
#     rpn_bbox_outside_weights = np.concatenate(rpn_bbox_outside_weights, axis=0)

#     return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] >= 4

    targets = bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

    return targets
