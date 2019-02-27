import numpy as np
import os
import os.path as osp
import cPickle
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.fast_rcnn.nms_wrapper import nms
from lib import network
from lib.fast_rcnn.bbox_transform import bbox_transform_inv_hdn, clip_boxes
import lib.utils.logger as logger
from lib.utils.nms import triplet_nms as triplet_nms_py
from lib.utils.nms import unary_nms
from lib.visualize_graph.vis_utils import expand_relationships_mat, expand_relationships_list

import pdb

# def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
#     keep = range(scores.shape[0])
#     keep, scores, pred_boxes = zip(*sorted(zip(keep, scores, pred_boxes), key=lambda x: x[1])[::-1])
#     keep, scores, pred_boxes = np.array(keep), np.array(scores), np.array(pred_boxes)
#     dets = np.hstack((pred_boxes, scores[:, np.newaxis])).astype(np.float32)
#     keep_keep = nms(dets, nms_thresh)
#     keep_keep = keep_keep[:min(100, len(keep_keep))]
#     keep = keep[keep_keep]
#     if inds is None:
#         return pred_boxes[keep_keep], scores[keep_keep], keep
#     return pred_boxes[keep_keep], scores[keep_keep], inds[keep], keep


def nms_detections(pred_boxes, scores, nms_thresh, inds):

    dets = np.hstack((pred_boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = unary_nms(dets, inds, nms_thresh)
    # print('NMS: [{}] --> [{}]'.format(scores.shape[0], len(keep)))
    keep = keep[:min(100, len(keep))]
    return pred_boxes[keep], scores[keep], inds[keep], keep



def save_checkpoint(info, model, optim, dir_logs, save_all_from=None, is_best=True):
    os.system('mkdir -p ' + dir_logs)
    if save_all_from is None:
        path_ckpt_info  = os.path.join(dir_logs, 'ckpt_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_model.h5')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_optim.pth.tar')
        path_best_info  = os.path.join(dir_logs, 'best_info.pth.tar')
        path_best_model = os.path.join(dir_logs, 'best_model.h5')
        path_best_optim = os.path.join(dir_logs, 'best_optim.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info)
        if is_best:
            shutil.copyfile(path_ckpt_info, path_best_info)
        # save model state & optim state
        network.save_net(path_ckpt_model, model)
        torch.save(optim, path_ckpt_optim)
        if is_best:
            shutil.copyfile(path_ckpt_model, path_best_model)
            shutil.copyfile(path_ckpt_optim, path_best_optim)
    elif info['epoch'] >= save_all_from:
        is_best = False # because we don't know the test accuracy
        path_ckpt_info  = os.path.join(dir_logs, 'ckpt_epoch,{}_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_epoch,{}_model.h5')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_epoch,{}_optim.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info.format(info['epoch']))
        network.save_net(path_ckpt_model.format(info['epoch']), model)
        torch.save(optim, path_ckpt_optim.format(info['epoch']))

def load_checkpoint(model, optimizer, path_ckpt):
    path_ckpt_info  = path_ckpt + '_info.pth.tar'
    path_ckpt_model = path_ckpt + '_model.h5'
    path_ckpt_optim = path_ckpt + '_optim.pth.tar'
    start_epoch = 0
    best_recall   = 0
    exp_logger  = None
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        if 'epoch' in info:
            start_epoch = info['epoch']
        else:
            print('Warning train.py: no epoch to resume')
        if 'best_recall' in info:
            best_recall = info['best_recall']
        else:
            print('Warning train.py: no best_recall to resume')
        if 'exp_logger' in info:
            exp_logger = info['exp_logger']
        else:
            print('Warning train.py: no exp_logger to resume')
    else:
        print("Warning train.py: no info checkpoint found at '{}'".format(path_ckpt_info))
    if os.path.isfile(path_ckpt_model):
        network.load_net(path_ckpt_model, model)
    else:
        print("Warning train.py: no model checkpoint found at '{}'".format(path_ckpt_model))
    #  if os.path.isfile(path_ckpt_optim):
    #      optim_state = torch.load(path_ckpt_optim)
    #      optimizer.load_state_dict(optim_state)
    #  else:
    #      print("Warning train.py: no optim checkpoint found at '{}'".format(path_ckpt_optim))
    print("=> loaded checkpoint '{}' (epoch {}, best_recall {})"
              .format(path_ckpt, start_epoch, best_recall * 100))
    return start_epoch, best_recall, exp_logger


def save_results(results, epoch, dir_logs, is_testing = True):
    if is_testing:
        subfolder_name = 'evaluate'
    else:
        subfolder_name = 'epoch_' + str(epoch)

    dir_epoch = os.path.join(dir_logs, subfolder_name)
    path_rslt = os.path.join(dir_epoch, 'testing_result.pkl')
    os.system('mkdir -p ' + dir_epoch)
    with open(path_rslt, 'wb') as f:
        cPickle.dump(results, f, cPickle.HIGHEST_PROTOCOL)

def save_detections(results, epoch, dir_logs, is_testing = True):
    if is_testing:
        subfolder_name = 'evaluate_object_detection'
    else:
        subfolder_name = 'epoch_' + str(epoch) + '_object_detection'

    dir_epoch = os.path.join(dir_logs, subfolder_name)
    os.system('mkdir -p ' + dir_epoch)
    temp_class_name = None
    for obj_class in results:
        if len(results[obj_class]) == 0:
            continue
        temp_class_name = obj_class
        with open(osp.join(dir_epoch, obj_class + '.txt'), 'wt') as f:
            for filename in results[obj_class]:
                for det in results[obj_class][filename]:
                    det = [str(i) for i in det]
                    f.write(filename + ' ' + det[-1] + ' ' + det[0] + ' ' + det[1] + ' ' + det[2] + ' ' + det[3] + '\n')

    with open(osp.join(dir_epoch, 'imageset.txt'), 'wt') as f:
        for img in results[temp_class_name].keys():
            f.write(img + '\n')

    with open(osp.join(dir_epoch, 'classes.txt'), 'wt') as f:
        for obj_class in results.keys():
            f.write(obj_class + '\n')

    print('Done dumping to: {}'.format(dir_epoch))
    return dir_epoch


def build_loss_cls(cls_score, labels, loss_weight=None):

        labels = labels.squeeze()
        fg_cnt = torch.sum(labels.data.ne(0))
        bg_cnt = labels.data.numel() - fg_cnt
        cross_entropy = F.cross_entropy(cls_score, labels, weight=loss_weight)
        maxv, predict = cls_score.data.max(1)
        if fg_cnt == 0:
            tp = torch.zeros_like(fg_cnt)
        else:
            tp = torch.sum(predict[:fg_cnt].eq(labels.data[:fg_cnt]))
        tf = torch.sum(predict[fg_cnt:].eq(labels.data[fg_cnt:]))
        fg_cnt = fg_cnt
        bg_cnt = bg_cnt
        return cross_entropy, (tp, tf, fg_cnt, bg_cnt)

def build_loss_bbox(bbox_pred, roi_data, fg_cnt):
        bbox_targets, bbox_inside_weights = roi_data[2:4]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)
        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-5)
        return loss_box




def interpret_relationships(cls_prob, bbox_pred, rois, cls_prob_predicate,
                        	mat_phrase, im_info, nms=-1., clip=True, min_score=0.01,
                        	top_N=100, use_gt_boxes=False, triplet_nms=-1., topk=10, 
                            reranked_score=None):

        scores, inds = cls_prob[:, 1:].data.max(1)
        if reranked_score is not None:
            if isinstance(reranked_score, Variable):
                reranked_score = reranked_score.data
            scores *= reranked_score.squeeze()
        inds += 1
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        predicate_scores, predicate_inds = cls_prob_predicate[:, 1:].data.topk(dim=1, k=topk)
        predicate_inds += 1
        predicate_scores, predicate_inds = predicate_scores.cpu().numpy().reshape(-1), predicate_inds.cpu().numpy().reshape(-1)


        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        keep = range(scores.shape[0])
        if use_gt_boxes:
            triplet_nms = -1.
            pred_boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
        else:
            pred_boxes = bbox_transform_inv_hdn(rois.data.cpu().numpy()[:, 1:5], box_deltas) / im_info[0][2]
            pred_boxes = clip_boxes(pred_boxes, im_info[0][:2] / im_info[0][2])

            # nms
            if nms > 0. and pred_boxes.shape[0] > 0:
                assert nms < 1., 'Wrong nms parameters'
                pred_boxes, scores, inds, keep = nms_detections(pred_boxes, scores, nms, inds=inds)


        sub_list = np.array([], dtype=int)
        obj_list = np.array([], dtype=int)
        pred_list = np.array([], dtype=int)

        # mapping the object id
        mapping = np.ones(cls_prob.size(0), dtype=np.int64) * -1
        mapping[keep] = range(len(keep))


        sub_list = mapping[mat_phrase[:, 0]]
        obj_list = mapping[mat_phrase[:, 1]]
        pred_remain = np.logical_and(sub_list >= 0,  obj_list >= 0)
        pred_list = np.where(pred_remain)[0]
        sub_list = sub_list[pred_remain]
        obj_list = obj_list[pred_remain]

        # expand the sub/obj and pred list to k-column
        pred_list = np.vstack([pred_list * topk + i for i in range(topk)]).transpose().reshape(-1)
        sub_list = np.vstack([sub_list for i in range(topk)]).transpose().reshape(-1)
        obj_list = np.vstack([obj_list for i in range(topk)]).transpose().reshape(-1)

        if use_gt_boxes:
            total_scores = predicate_scores[pred_list]
        else:
            total_scores = predicate_scores[pred_list] * scores[sub_list] * scores[obj_list]

        top_N_list = total_scores.argsort()[::-1][:10000]
        total_scores = total_scores[top_N_list]
        pred_ids = predicate_inds[pred_list[top_N_list]] # category of predicates
        sub_assignment = sub_list[top_N_list] # subjects assignments
        obj_assignment = obj_list[top_N_list] # objects assignments
        sub_ids = inds[sub_assignment] # category of subjects
        obj_ids = inds[obj_assignment] # category of objects
        sub_boxes = pred_boxes[sub_assignment] # boxes of subjects
        obj_boxes = pred_boxes[obj_assignment] # boxes of objects


        if triplet_nms > 0.:
            sub_ids, obj_ids, pred_ids, sub_boxes, obj_boxes, keep = triplet_nms_py(sub_ids, obj_ids, pred_ids, sub_boxes, obj_boxes, triplet_nms)
            sub_assignment = sub_assignment[keep]
            obj_assignment = obj_assignment[keep]
            total_scores = total_scores[keep]
        if len(sub_list) == 0:
            print('No Relatinoship remains')
            #pdb.set_trace()

        return pred_boxes, scores, inds, sub_ids, obj_ids, sub_boxes, obj_boxes, pred_ids, sub_assignment, obj_assignment, total_scores

def interpret_objects(cls_prob, bbox_pred, rois, im_info, nms_thres=-1., min_score=0.00001, use_gt_boxes=False, max_per_image=2000):
        box_deltas = bbox_pred.data.cpu().numpy()
        cls_prob = cls_prob.data.cpu().numpy()
        all_boxes =[[ ] for _ in xrange(cls_prob.shape[1])]

        for j in xrange(1, cls_prob.shape[1]): # skip the background
            inds = np.where(cls_prob[:, j] > min_score)[0]
            if len(inds) == 0:
                continue
            cls_scores = cls_prob[inds, j]
            if use_gt_boxes:
                cls_boxes = rois.data.cpu().numpy()[inds, 1:5] / im_info[0][2]
            else:
                t_box_deltas = np.asarray([ box_deltas[i, (j * 4): (j * 4 + 4)] for i in inds], dtype=np.float)
                cls_boxes = bbox_transform_inv_hdn(rois.data.cpu().numpy()[inds, 1:5], t_box_deltas) / im_info[0][2]
                cls_boxes = clip_boxes(cls_boxes, im_info[0][:2] / im_info[0][2])

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            if nms_thres > 0.:
                keep = nms(cls_dets, nms_thres)
                cls_dets = cls_dets[keep, :]

            all_boxes[j] = cls_dets

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:, -1]  for j in xrange(1, cls_prob.shape[1]) if len(all_boxes[j]) > 0])
            #print('{} detections.'.format(len(image_scores)))
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, cls_prob.shape[1]):
                    if len(all_boxes[j]) == 0:
                        continue
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        return all_boxes


