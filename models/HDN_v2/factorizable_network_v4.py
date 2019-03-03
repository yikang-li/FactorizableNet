import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os.path as osp
import yaml

from lib.utils.timer import Timer
from lib.utils.metrics import check_relationship_recall, check_phrase_recall

from lib.utils.proposal_target_layer_v1 import proposal_target_layer as proposal_target_layer_py
from lib.utils.proposal_target_layer_v1 import graph_construction as graph_construction_py
from lib.fast_rcnn.bbox_transform import bbox_transform_inv_hdn, clip_boxes

# modules
import models.modules.phrase_inference_structure as fusion_inference
from models.modules import factor_updating_structure_v3 as factor_updating_structure
from models.RPN import RPN
#from models.language_model.Language_Model import Language_Model
from lib.network import GroupDropout
from models.modules import Dumplicate_Removal


from lib.utils.cython_bbox import bbox_overlaps
from .utils import nms_detections, build_loss_bbox, build_loss_cls, interpret_relationships, interpret_objects
import lib.network as network
from lib.layer_utils.roi_layers.roi_align import ROIAlign
import pdb

import engines_v1 as engines

DEBUG = False
TIME_IT = False


class Factorizable_network(nn.Module):
    _feat_stride = 16

    def __init__(self, trainset, opts = None):

        super(Factorizable_network, self).__init__()
        # network settings
        self.n_classes_obj = trainset.num_object_classes
        self.n_classes_pred = trainset.num_predicate_classes
        self.MPS_iter = opts['MPS_iter']
        # loss weight
        ce_weights_obj = np.sqrt(trainset.inverse_weight_object)
        ce_weights_obj[0] = 1.
        ce_weights_pred = np.sqrt(trainset.inverse_weight_predicate)
        ce_weights_pred[0] = 1.
        self.object_loss_weight = ce_weights_obj if opts.get('use_loss_weight', False) else None
        self.predicate_loss_weight = ce_weights_pred if opts.get('use_loss_weight', False) else None
        self.opts = opts

        # loss
        self.loss_cls_obj = None
        self.loss_cls_rel = None
        self.loss_reg_obj = None

        with open(opts['rpn_opts'], 'r') as f:
            self.rpn_opts = yaml.load(f)
            assert len(trainset.opts['test']['SCALES']) == 1, "Currently only support single testing scale."
            self.rpn_opts['scale'] = trainset.opts['test']['SCALES'][0]

        self.rpn = RPN(self.rpn_opts)
        pool_size = self.opts.get('pool_size', 7)
        self.roi_pool_object = ROIAlign((pool_size, pool_size), 1.0/16, 0)
        self.roi_pool_region = ROIAlign((pool_size, pool_size), 1.0/16, 0)
        self.fc_obj = nn.Sequential(
                        nn.Linear(512 * pool_size * pool_size, opts['dim_ho']),
                        GroupDropout(p=opts['dropout'], inplace=True),
                        nn.ReLU(inplace=True),
                        nn.Linear(opts['dim_ho'], opts['dim_ho']),
                        GroupDropout(p=opts['dropout'], inplace=True),)

        self.fc_region = nn.Sequential(
                        nn.Conv2d(512, opts['dim_hr'], 3, stride=1, padding=1),
                        GroupDropout(p=opts['dropout'], inplace=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(opts['dim_hr'], opts['dim_hr'], 3, stride=1, padding=1),
                        GroupDropout(p=opts['dropout'], inplace=True),)

        self.fc_obj.apply(network.weight_init_fun_kaiming)
        self.fc_region.apply(network.weight_init_fun_kaiming)
        # network.weights_normal_init(self.fc_obj, 0.01)
        # network.weights_normal_init(self.fc_region, 0.01)

        # the hierarchical message passing structure
        print('{} MPS modules are used.'.format(self.MPS_iter))
        self.mps_list = nn.ModuleList(
                [factor_updating_structure(opts) for i in range(self.MPS_iter)])
        # self.mps_list.apply(network.weight_init_fun_kaiming)
        network.weights_normal_init(self.mps_list, 0.01)

        self.phrase_inference = getattr(fusion_inference, self.opts['fusion'])(opts)
        # self.phrase_inference.apply(network.weight_init_fun_kaiming)
        network.weights_normal_init(self.phrase_inference, 0.01)

        self.score_obj = nn.Linear(opts['dim_ho'], self.n_classes_obj)
        self.bbox_obj = nn.Linear(opts['dim_ho'], self.n_classes_obj * 4)
        self.score_pred = nn.Linear(opts['dim_hp'], self.n_classes_pred)
        self.learnable_nms = self.opts.get('nms', 1.) > 0
        self.nms = Dumplicate_Removal(opts)


        network.weights_normal_init(self.score_obj, 0.01)
        network.weights_normal_init(self.bbox_obj, 0.005)
        network.weights_normal_init(self.score_pred, 0.01)

        self.engines = engines

        





    def loss(self, losses):
        # loss weights are defined in option files.
        if self.learnable_nms:
            return losses['loss_cls_obj'] * self.opts['cls_obj'] + \
                    losses['loss_reg_obj'] * self.opts['reg_obj']+ \
                    losses['loss_cls_rel'] * self.opts['cls_pred'] + \
                    losses['loss_nms'] * self.opts.get('nms', 1.)

        else:
            return losses['loss_cls_obj'] * self.opts['cls_obj'] + \
                    losses['loss_reg_obj'] * self.opts['reg_obj']+ \
                    losses['loss_cls_rel'] * self.opts['cls_pred']


    def forward(self, im_data, im_info, gt_objects=None, gt_relationships=None, rpn_anchor_targets_obj=None):

        assert im_data.size(0) == 1, "Only support Batch Size equals 1"
        base_timer = Timer()
        mps_timer = Timer()
        infer_timer = Timer()
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
                mps(pooled_object_features, pooled_region_features, mat_object, mat_region)
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
            print('[CNN]:\t{0:.3f} s'.format(self.base_timer.average_time))
            print('[MPS]:\t{0:.3f} s'.format(self.mps_timer.average_time))
            print('[INF]:\t{0:.3f} s'.format(self.infer_timer.average_time))


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
                mps(pooled_object_features, pooled_region_features, mat_object, mat_region)

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

    def evaluate(self, im_data, im_info, gt_objects, gt_relationships,
        thr=0.5, nms=-1., triplet_nms=-1., top_Ns = [100],
        use_gt_boxes=False):
        gt_objects = gt_objects[0]
        gt_relationships = gt_relationships[0]
        if use_gt_boxes:
            object_result, predicate_result = self.forward_eval(im_data, im_info,
                            gt_objects=gt_objects)
        else:
            object_result, predicate_result = self.forward_eval(im_data, im_info,)

        cls_prob_object, bbox_object, object_rois, reranked_score = object_result[:4]
        cls_prob_predicate, mat_phrase = predicate_result[:2]
        region_rois_num = predicate_result[2]

        # interpret the model output
        obj_boxes, obj_scores, obj_cls, subject_inds, object_inds, \
            subject_boxes, object_boxes, predicate_inds, \
            sub_assignment, obj_assignment, total_score = \
                interpret_relationships(cls_prob_object, bbox_object, object_rois,
                            cls_prob_predicate, mat_phrase, im_info,
                            nms=nms, top_N=max(top_Ns),
                            use_gt_boxes=use_gt_boxes,
                            triplet_nms=triplet_nms,
                            reranked_score=reranked_score)

        gt_objects[:, :4] /= im_info[0][2]
        rel_cnt, rel_correct_cnt, pred_correct_cnt = check_relationship_recall(gt_objects, gt_relationships,
                                        subject_inds, object_inds, predicate_inds,
                                        subject_boxes, object_boxes, top_Ns, thres=thr)
        _, phrase_correct_cnt = check_phrase_recall(gt_objects, gt_relationships,
                                        subject_inds, object_inds, predicate_inds,
                                        subject_boxes, object_boxes, top_Ns, thres=thr)

        result = {'objects': {
                            'bbox': obj_boxes,
                            'scores': obj_scores,
                            'class': obj_cls,},
                  'relationships': zip(sub_assignment, obj_assignment, predicate_inds, total_score),
                  'rel_recall': [float(v) / rel_cnt for v in rel_correct_cnt], 
                  'phr_recall': [float(v) / rel_cnt for v in phrase_correct_cnt], 
                  'pred_recall': [float(v) / rel_cnt for v in pred_correct_cnt],
                 }


        return rel_cnt, (rel_correct_cnt, phrase_correct_cnt, pred_correct_cnt, region_rois_num), result

    def evaluate_object_detection(self, im_data, im_info, gt_objects,thr=0.5, nms=-1., use_gt_boxes=False):
        gt_objects = gt_objects[0]
        if use_gt_boxes:
            object_result, predicate_result = self.forward_eval(im_data, im_info,
                            gt_objects=gt_objects)
        else:
            object_result, predicate_result = self.forward_eval(im_data, im_info,)

        cls_prob_object, bbox_object, object_rois = object_result[:3]

        # interpret the model output
        all_boxes = interpret_objects(cls_prob_object, bbox_object, object_rois, im_info,
                            nms_thres=nms, use_gt_boxes=use_gt_boxes)

        return all_boxes

    @staticmethod
    def graph_construction(object_rois, gt_rois=None):
        if isinstance(object_rois, torch.Tensor):
            object_rois = object_rois.cpu().numpy()
        object_rois, region_rois, mat_object, mat_phrase, mat_region = graph_construction_py(object_rois, gt_rois)
        object_rois = network.np_to_variable(object_rois, is_cuda=True)
        region_rois = network.np_to_variable(region_rois, is_cuda=True)

        return object_rois, region_rois, mat_object, mat_phrase, mat_region


    @staticmethod
    def proposal_target_layer(object_rois, gt_objects, gt_relationships, n_classes_obj):

        """
        ----------
        object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_objects:   (G_obj, 5) [x1 ,y1 ,x2, y2, obj_class] int
        gt_relationships: (G_obj, G_obj) [pred_class] int (-1 for no relationship)
        gt_regions:   (G_region, 4+40) [x1, y1, x2, y2, word_index] (-1 for padding)
        # gt_ishard: (G_region, 4+40) {0 | 1} 1 indicates hard
        # dontcare_areas: (D, 4) [ x1, y1, x2, y2]
        n_classes_obj
        n_classes_pred
        is_training to indicate whether in training scheme
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """

        object_rois = object_rois.data.cpu().numpy()

        targets_object, targets_phrase, targets_region = proposal_target_layer_py(object_rois, gt_objects, gt_relationships, n_classes_obj)
        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        object_labels, object_rois, bbox_targets, bbox_inside_weights, mat_object, object_fg_duplicate= targets_object[:6]
        phrase_labels, mat_phrase = targets_phrase[:2]
        region_rois, mat_region = targets_region[:2]


        object_rois = network.np_to_variable(object_rois, is_cuda=True)
        region_rois = network.np_to_variable(region_rois, is_cuda=True)
        object_labels = network.np_to_variable(object_labels, is_cuda=True, dtype=torch.LongTensor)
        bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
        bbox_inside_weights = network.np_to_variable(bbox_inside_weights, is_cuda=True)
        phrase_labels = network.np_to_variable(phrase_labels, is_cuda=True, dtype=torch.LongTensor)
        duplicate_labels = network.np_to_variable(object_fg_duplicate, is_cuda=True)

        return tuple([object_labels, object_rois, bbox_targets, bbox_inside_weights, duplicate_labels]), \
                tuple([phrase_labels]), \
                tuple([None, region_rois]), \
                mat_object, mat_phrase, mat_region




if __name__ == '__main__':
    Factorizable_network(None, None)
