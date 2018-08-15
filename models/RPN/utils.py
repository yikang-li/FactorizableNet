import cv2
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import shutil

from lib.fast_rcnn.nms_wrapper import nms
from lib import network

import pdb

def save_checkpoint(filename, model, epoch, is_best):
    model_name = '{}_epoch_{}.h5'.format(filename, epoch)
    model_name_best = '{}_best.h5'.format(filename)
    info_name = '{}_epoch_{}_info.json'.format(filename, epoch)
    info_name_best = '{}_best_info.json'.format(filename)
    network.save_net(model_name, model)
    with open(info_name, 'w') as f:
        json.dump(model.opts, f)
    print('save model: {}'.format(model_name))
    if is_best:
        shutil.copyfile(model_name, model_name_best)
        shutil.copyfile(info_name, info_name_best)

def load_checkpoint(filename, model):
    model_name = '{}.h5'.format(filename)
    info_name = '{}_info.json'.format(filename)
    network.load_net(model_name, model)
    if False:  # disable info loading #osp.isfile(info_name):
        with open(info_name, 'r') as f:
            model.opts = json.load(f)
    else:
        print('Info file missed, using the default options')



def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )

        return x
        # x = x.permute(0, 2, 3, 1)

def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]

def build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
    # classification loss
    rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
    rpn_label = rpn_data[0].view(-1)
    # print rpn_label.size(), rpn_cls_score.size()
    rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze())
    rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
    rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

    fg_cnt = torch.sum(rpn_label.data.ne(0))
    bg_cnt = rpn_label.data.numel() - fg_cnt

    _, predict = torch.max(rpn_cls_score.data, 1)
    error = torch.sum(torch.abs(predict - rpn_label.data))
    #  try:
    if fg_cnt == 0:
        tp = 0.
        tf = tf = torch.sum(predict.eq(rpn_label.data))
    else:
        tp = torch.sum(predict[:fg_cnt].eq(rpn_label.data[:fg_cnt]))
        tf = torch.sum(predict[fg_cnt:].eq(rpn_label.data[fg_cnt:]))
    fg_cnt = fg_cnt
    bg_cnt = bg_cnt
    # print 'accuracy: %2.2f%%' % ((self.tp + self.tf) / float(fg_cnt + bg_cnt) * 100)
    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
    # print rpn_cross_entropy

    # box loss
    rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
    rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
    rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)
    rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) /  (fg_cnt + 1e-4)

    return rpn_cross_entropy, rpn_loss_box, (tp, tf, fg_cnt, bg_cnt)


def generate_output_mapping(mapping_file, conv_layers, min_size=16, max_size=1001):
    if osp.isfile(mapping_file):
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
            
        mappings = {int(k):int(v) for k,v in mappings.items()}
        return mappings
    else:
        conv_layers.cuda()
        print('Generating input/output size mappings')
        mappings = {}
        for i in range(min_size, max_size):
            t_in = Variable(torch.zeros(1, 3, i, i).cuda())
            t_out = conv_layers(t_in)
            mappings[i] = t_out.size(2)

        with open(mapping_file, 'w') as f:
            json.dump(mappings, f)
    print('Done')
    return mappings
