import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import os.path as osp
import yaml
from .factorizable_network_v4 import Factorizable_network as FN_v4
from models.modules.factor_updating_structure_v3r import factor_updating_structure
import torch.nn as nn
from lib.network import GroupDropout
import lib.network as network
import torch.nn.functional as F
from .utils import nms_detections, build_loss_bbox, build_loss_cls, interpret_relationships, interpret_objects


from lib.utils.timer import Timer


DEBUG = False
TIME_IT = False


class Factorizable_network(FN_v4):

    def __init__(self, trainset, opts = None):

        super(Factorizable_network, self).__init__(trainset, opts)
        self.mps_list = nn.ModuleList(
                [factor_updating_structure(opts) for i in range(self.MPS_iter)])
        # self.mps_list.apply(network.weight_init_fun_kaiming)
        network.weights_normal_init(self.mps_list, 0.01)

    def forward(self, im_data, im_info, gt_objects=None, gt_relationships=None, rpn_anchor_targets_obj=None):
        # timing the process
        base_timer = Timer()
        mps_timer = Timer()
        infer_timer = Timer()
        assert im_data.size(0) == 1, "Only support Batch Size equals 1"
        base_timer.tic()
        # Currently, RPN support batch but not for MSDN
        features, object_rois, rpn_losses = self.rpn(im_data, im_info, rpn_data=rpn_anchor_targets_obj)
        if self.training:
            roi_data_object, roi_data_predicate, roi_data_region, mat_object, mat_phrase, mat_region = \
                self.proposal_target_layer(object_rois, gt_objects[0], gt_relationships[0], self.n_classes_obj)
            object_rois = roi_data_object[1]
            region_rois = roi_data_region[1]
        else:
            object_rois, region_rois, mat_object, mat_phrase, mat_region = self.graph_construction(object_rois,)
        # roi pool
        pooled_object_features = self.roi_pool_object(features, object_rois).view(len(object_rois), -1)
        pooled_object_features = self.fc_obj(pooled_object_features)
        # print 'fc7_object.std', pooled_object_features.data.std()

        pooled_region_features = self.roi_pool_region(features, region_rois)
        pooled_region_features = self.fc_region(pooled_region_features)

        bbox_object = self.bbox_obj(F.relu(pooled_object_features))
        base_timer.toc()

        mps_timer.tic()
        for i, mps in enumerate(self.mps_list):
            pooled_object_features, pooled_region_features = \
                mps(pooled_object_features, pooled_region_features, mat_object, mat_region, object_rois, region_rois)

        mps_timer.toc()

        infer_timer.tic()
        pooled_phrase_features = self.phrase_inference(pooled_object_features, pooled_region_features, mat_phrase)
        infer_timer.toc()

        cls_score_object = self.score_obj(F.relu(pooled_object_features))
        cls_prob_object = F.softmax(cls_score_object, dim=1)
        cls_score_predicate = self.score_pred(F.relu(pooled_phrase_features))
        cls_prob_predicate = F.softmax(cls_score_predicate, dim=1)

        if TIME_IT:
            print('TIMING:')
            print('[CNN]:\t{0:.3f} s'.format(base_timer.average_time))
            print('[MPS]:\t{0:.3f} s'.format(mps_timer.average_time))
            print('[INF]:\t{0:.3f} s'.format(infer_timer.average_time))


        # object cls loss
        loss_cls_obj, (tp, tf, fg_cnt, bg_cnt) = \
                build_loss_cls(cls_score_object, roi_data_object[0], 
                    loss_weight=self.object_loss_weight.to(cls_score_object.get_device()))
        # object regression loss
        loss_reg_obj= build_loss_bbox(bbox_object, roi_data_object, fg_cnt)
        # predicate cls loss
        loss_cls_rel,  (tp_pred, tf_pred, fg_cnt_pred, bg_cnt_pred)= \
                build_loss_cls(cls_score_predicate, roi_data_predicate[0], 
                    loss_weight=self.predicate_loss_weight.to(cls_score_predicate.get_device()))
        losses = {
            'rpn': rpn_losses,
            'loss_cls_obj': loss_cls_obj, 
            'loss_reg_obj': torch.zeros_like(loss_reg_obj) if torch.isnan(loss_reg_obj) else loss_reg_obj,
            'loss_cls_rel': loss_cls_rel,
            'tf': tf,
            'tp': tp,
            'fg_cnt': fg_cnt,
            'bg_cnt': bg_cnt,
            'tp_pred': tp_pred,
            'tf_pred': tf_pred,
            'fg_cnt_pred': fg_cnt_pred,
            'bg_cnt_pred': bg_cnt_pred,
        }
        # loss for NMS
        if self.learnable_nms:
            duplicate_labels = roi_data_object[4][:, 1:2]
            duplicate_weights = roi_data_object[4][:, 0:1]
            if duplicate_weights.data.sum() == 0:
                loss_nms = loss_cls_rel * 0 # Guarentee the data type
            else:
                mask = torch.zeros_like(cls_prob_object).byte()
                for i in range(duplicate_labels.size(0)):
                    mask[i, roi_data_object[0].data[i][0]] = 1
                selected_prob = torch.masked_select(cls_prob_object, mask)
                reranked_score = self.nms(pooled_object_features, selected_prob, roi_data_object[1])
                selected_prob = selected_prob.unsqueeze(1) * reranked_score
                loss_nms = F.binary_cross_entropy(selected_prob, duplicate_labels,
                                    weight=duplicate_weights,
                                    size_average=False) / (duplicate_weights.data.sum() + 1e-10)
            losses["loss_nms"] = loss_nms

        losses['loss'] = self.loss(losses)

        return losses

    def forward_eval(self, im_data, im_info, gt_objects=None):
        # Currently, RPN support batch but not for MSDN
        features, object_rois, _ = self.rpn(im_data, im_info)
        if gt_objects is not None:
            gt_rois = np.concatenate([np.zeros((gt_objects.shape[0], 1)),
                                      gt_objects[:, :4],
                                      np.ones((gt_objects.shape[0], 1))], 1)
        else:
            gt_rois = None
        object_rois, region_rois, mat_object, mat_phrase, mat_region = self.graph_construction(object_rois, gt_rois=gt_rois)
        # roi pool
        pooled_object_features = self.roi_pool_object(features, object_rois).view(len(object_rois), -1)
        pooled_object_features = self.fc_obj(pooled_object_features)
        pooled_region_features = self.roi_pool_region(features, region_rois)
        pooled_region_features = self.fc_region(pooled_region_features)
        bbox_object = self.bbox_obj(F.relu(pooled_object_features))

        for i, mps in enumerate(self.mps_list):
            pooled_object_features, pooled_region_features = \
                mps(pooled_object_features, pooled_region_features, mat_object, mat_region, object_rois, region_rois)

        pooled_phrase_features = self.phrase_inference(pooled_object_features, pooled_region_features, mat_phrase)
        pooled_object_features = F.relu(pooled_object_features)
        pooled_phrase_features = F.relu(pooled_phrase_features)

        cls_score_object = self.score_obj(pooled_object_features)
        cls_prob_object = F.softmax(cls_score_object, dim=1)
        cls_score_predicate = self.score_pred(pooled_phrase_features)
        cls_prob_predicate = F.softmax(cls_score_predicate, dim=1)



        if self.learnable_nms:
            selected_prob, _ = cls_prob_object[:, 1:].max(dim=1, keepdim=False)
            reranked_score = self.nms(pooled_object_features, selected_prob, object_rois)
        else:
            reranked_score = None


        return (cls_prob_object, bbox_object, object_rois, reranked_score), \
                (cls_prob_predicate, mat_phrase, region_rois.size(0)),

if __name__ == '__main__':
    Factorizable_network(None, None)
