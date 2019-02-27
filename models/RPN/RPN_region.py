import cv2
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.utils.timer import Timer
from lib.utils.blob import im_list_to_blob
from lib.rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from lib.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from lib.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

from lib import network
from lib.network import Conv2d, FC
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math
import json
import yaml
import pdb

from .utils import nms_detections, build_loss, reshape_layer, generate_output_mapping

DEBUG = False



class RPN(nn.Module):
    _feat_stride = [16, ]

    anchor_scales_normal = [2, 4, 8, 16, 32, 64]
    anchor_ratios_normal = [0.25, 0.5, 1, 2, 4]
    anchor_scales_normal_region = [4, 8, 16, 32, 64]
    anchor_ratios_normal_region = [0.25, 0.5, 1, 2, 4]

    def __init__(self, opts):
        super(RPN, self).__init__()

        self.opts = opts
        if self.opts['kmeans_anchors']:
            # Loading k-means anchors
            kmeans_anchors_file = osp.join(self.opts['anchor_dir'], 'kmeans_anchors.json')
            print 'using k-means anchors: {}'.format(kmeans_anchors_file)
            anchors = json.load(open(kmeans_anchors_file))
            if 'scale' not in self.opts:
                print('No RPN scale is given, default [600] is set')
            self.opts['object']['anchor_scales'] = list(np.array(anchors['anchor_scales_kmeans']) / 600.0 * self.opts.get('scale', 600.))
            self.opts['object']['anchor_ratios'] = anchors['anchor_ratios_kmeans']
            self.opts['region']['anchor_scales'] = list(np.array(anchors['anchor_scales_kmeans_region']) / 600.0 * self.opts.get('scale', 600.))
            self.opts['region']['anchor_ratios'] = anchors['anchor_ratios_kmeans_region']
        else:
            print 'using normal anchors'
            anchor_scales, anchor_ratios = \
                np.meshgrid(self.anchor_scales_normal, self.anchor_ratios_normal, indexing='ij')
            self.opts['object']['anchor_scales'] = anchor_scales.reshape(-1)
            self.opts['object']['anchor_ratios'] = anchor_ratios.reshape(-1)
            anchor_scales, anchor_ratios = \
                np.meshgrid(self.anchor_scales_normal_region, self.anchor_ratios_normal_region, indexing='ij')
            self.opts['region']['anchor_scales'] = anchor_scales.reshape(-1)
            self.opts['region']['anchor_ratios'] = anchor_ratios.reshape(-1)

        self.anchor_num = len(self.opts['object']['anchor_scales'])
        self.anchor_num_region = len(self.opts['region']['anchor_scales'])

        self.features = models.vgg16(pretrained=True).features
        self.features.__delattr__('30') # to delete the max pooling
        # by default, fix the first four layers
        network.set_trainable_param(list(self.features.parameters())[:8], requires_grad=False)

        # self.features = models.vgg16().features
        self.conv1 = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv = Conv2d(512, self.anchor_num * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, self.anchor_num * 4, 1, relu=False, same_padding=False)

        self.conv1_region = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv_region = Conv2d(512, self.anchor_num_region * 2, 1, relu=False, same_padding=False)
        self.bbox_conv_region = Conv2d(512, self.anchor_num_region * 4, 1, relu=False, same_padding=False)

        # initialize the parameters
        self.initialize_parameters()
        self.opts['mappings'] = generate_output_mapping(osp.join(self.opts['anchor_dir'], 'vgg16_mappings.json'),
                                                        self.features)

    def initialize_parameters(self, normal_method='normal'):


        if normal_method == 'normal':
            normal_fun = network.weights_normal_init
        elif normal_method == 'MSRA':
            normal_fun = network.weights_MSRA_init
        else:
            raise(Exception('Cannot recognize the normal method:'.format(normal_method)))

        normal_fun(self.conv1, 0.025)
        normal_fun(self.score_conv, 0.025)
        normal_fun(self.bbox_conv, 0.01)
        normal_fun(self.conv1_region, 0.025)
        normal_fun(self.score_conv_region, 0.025)
        normal_fun(self.bbox_conv_region, 0.01)

    @property
    def loss(self):
        return self.loss_cls_obj + self.loss_box_obj * 0.5 + self.loss_cls_region + self.loss_box_region * 0.5

    def forward(self, im_data, im_info, rpn_data_obj=None, rpn_data_region=None):

        features = self.features(im_data)
        # print 'features.std()', features.data.std()
        rpn_conv1 = self.conv1(features)
        # print 'rpn_conv1.std()', rpn_conv1.data.std()
        # object proposal score
        rpn_cls_score = self.score_conv(rpn_conv1)
        # print 'rpn_cls_score.std()', rpn_cls_score.data.std()
        rpn_cls_score_reshape = reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob_reshape = reshape_layer(rpn_cls_prob, self.anchor_num*2)
        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)
        # print 'rpn_bbox_pred.std()', rpn_bbox_pred.data.std() * 4

        rpn_conv1_region = self.conv1_region(features)
        # print 'rpn_conv1_region.std()', rpn_conv1_region.data.std()
        # object proposal score
        rpn_cls_score_region = self.score_conv(rpn_conv1_region)
        # print 'rpn_cls_score_region.std()', rpn_cls_score_region.data.std()
        rpn_cls_score_region_reshape = reshape_layer(rpn_cls_score_region, 2)
        rpn_cls_prob_region = F.softmax(rpn_cls_score_region_reshape, dim=1)
        rpn_cls_prob_region_reshape = reshape_layer(rpn_cls_prob_region, self.anchor_num*2)
        # rpn boxes
        rpn_bbox_pred_region = self.bbox_conv(rpn_conv1_region)
        # print 'rpn_bbox_pred_region.std()', rpn_bbox_pred_region.data.std() * 4
        # proposal layer
        cfg_key = 'train' if self.training else 'test'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                                   self._feat_stride, self.opts['object'][cfg_key],
                                   self.opts['object']['anchor_scales'],
                                   self.opts['object']['anchor_ratios'],
                                   mappings=self.opts['mappings'])
        region_rois = self.proposal_layer(rpn_cls_prob_region_reshape, rpn_bbox_pred_region, im_info,
                                   self._feat_stride, self.opts['region'][cfg_key],
                                   self.opts['region']['anchor_scales'],
                                   self.opts['region']['anchor_ratios'],
                                   mappings=self.opts['mappings'])

        # generating training labels and build the rpn loss
        if self.training and rpn_data_obj is not None:
            self.loss_cls_obj, self.loss_box_obj, accs = build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data_obj)
            self.tp, self.tf, self.fg_cnt, self.bg_cnt = accs

            self.loss_cls_region, self.loss_box_region, accs = build_loss(
                            rpn_cls_score_region_reshape, rpn_bbox_pred_region, rpn_data_region)
            self.tp_region, self.tf_region, self.fg_cnt_region, self.bg_cnt_region = accs

        return features, rois, region_rois

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                    _feat_stride, opts, anchor_scales, anchor_ratios, mappings):
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                    _feat_stride, opts, anchor_scales, anchor_ratios, mappings)
        x = network.np_to_variable(x, is_cuda=True)
        return x.view(-1, 6)
