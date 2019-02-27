# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
from lib.fast_rcnn.nms_wrapper import nms

from lib.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from generate_anchors import generate_anchors

import pdb


DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""


def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_infos, 
                    _feat_stride, opts, anchor_scales, anchor_ratios,
                    mappings):
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    # layer_params = yaml.load(self.param_str_)
    batch_size = rpn_cls_prob_reshape.shape[0]
    _anchors = generate_anchors(scales=anchor_scales, ratios=anchor_ratios)
    _num_anchors = _anchors.shape[0]
    pre_nms_topN = opts['num_box_pre_NMS']
    post_nms_topN = opts['num_box_post_NMS']
    nms_thres = opts['nms_thres']
    min_size = opts['min_size']

    blob = []
    
    for i in range(batch_size):
        im_info = im_infos[i]
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        height = mappings[int(im_info[0])]
        width = mappings[int(im_info[1])]
        scores = rpn_cls_prob_reshape[i, _num_anchors:, :height, :width]
        bbox_deltas = rpn_bbox_pred[i, :, :height, :width]

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * _feat_stride
        shift_y = np.arange(0, height) * _feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = _num_anchors
        K = shifts.shape[0]
        anchors = _anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((1, 2, 0)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        if opts['dropout_box_runoff_image']:
            _allowed_border = 16
            inds_inside = np.where(
                (proposals[:, 0] >= -_allowed_border) &
                (proposals[:, 1] >= -_allowed_border) &
                (proposals[:, 2] < im_info[1] + _allowed_border) &  # width
                (proposals[:, 3] < im_info[0] + _allowed_border)  # height
            )[0]
            proposals = proposals[inds_inside, :]
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        # print 'proposals', proposals
        # print 'scores', scores
        keep = nms(np.hstack((proposals, scores)).astype(np.float32), nms_thres)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.ones((proposals.shape[0], 1), dtype=np.float32) * i
        blob.append(np.hstack((batch_inds, proposals.astype(np.float32, copy=False), scores.astype(np.float32, copy=False))))

    return np.concatenate(blob, axis=0)


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
