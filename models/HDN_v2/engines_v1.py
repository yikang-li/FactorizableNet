import pdb
import os.path as osp
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

import lib.network as network
from lib.network import np_to_variable

def train(loader, model, optimizer, exp_logger, epoch, train_all, print_freq=100, clip_gradient=True, iter_size=1):

    model.train()
    meters = exp_logger.reset_meters('train')
    end = time.time()

    for i, sample in enumerate(loader): # (im_data, im_info, gt_objects, gt_relationships)
        # measure the data loading time
        batch_size = sample['visual'].size(0)

        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)

        input_visual = Variable(sample['visual'].cuda())
        target_objects = sample['objects']
        target_relations = sample['relations']
        image_info = sample['image_info']
        # RPN targets
        rpn_anchor_targets_obj = [
                np_to_variable(sample['rpn_targets']['object'][0],is_cuda=True, dtype=torch.LongTensor),
                np_to_variable(sample['rpn_targets']['object'][1],is_cuda=True),
                np_to_variable(sample['rpn_targets']['object'][2],is_cuda=True),
                np_to_variable(sample['rpn_targets']['object'][3],is_cuda=True)
                ]
        try:
            # compute output
            model(input_visual, image_info, target_objects, target_relations, rpn_anchor_targets_obj)
            # Determine the loss function
            if train_all:
                loss = model.loss + model.rpn.loss * 0.5
            else:
                loss = model.loss

            # to logging the loss and itermediate values
            meters['loss'].update(model.loss.data.cpu().numpy()[0], n=batch_size)
            meters['loss_cls_obj'].update(model.loss_cls_obj.data.cpu().numpy()[0], n=batch_size)
            meters['loss_reg_obj'].update(model.loss_reg_obj.data.cpu().numpy()[0], n=batch_size)
            meters['loss_cls_rel'].update(model.loss_cls_rel.data.cpu().numpy()[0], n=batch_size)
            meters['loss_rpn'].update(model.rpn.loss.data.cpu().numpy()[0], n=batch_size)
            meters['batch_time'].update(time.time() - end, n=batch_size)
            meters['epoch_time'].update(meters['batch_time'].val, n=batch_size)

            # add support for iter size
            # special case: last iterations
            if i % iter_size == 0 or i == len(loader) - 1:
                loss.backward()
                if clip_gradient:
                    network.clip_gradient(model, 10.)
                else:
                    network.avg_gradient(model, iter_size)
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss.backward()

            end = time.time()
            # Logging the training loss
            if  (i + 1) % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}] '
                      'Batch_Time: {batch_time.avg: .3f}\t'
                      'FRCNN Loss: {loss.avg: .4f}\t'
                      'RPN Loss: {rpn_loss.avg: .4f}\t'.format(
                       epoch, i + 1, len(loader), batch_time=meters['batch_time'],
                       loss=meters['loss'], rpn_loss=meters['loss_rpn']))


                print('\t[object] loss_cls_obj: {loss_cls_obj.avg:.4f} '
                	  'loss_reg_obj: {loss_reg_obj.avg:.4f} '
                	  'loss_cls_rel: {loss_cls_rel.avg:.4f} '.format(
                      loss_cls_obj = meters['loss_cls_obj'],
                      loss_reg_obj = meters['loss_reg_obj'],
                      loss_cls_rel = meters['loss_cls_rel'], ))
        except Exception:
            pdb.set_trace()

    exp_logger.log_meters('train', n=epoch)







def test(loader, model, top_Ns, nms=-1., triplet_nms=-1., use_gt_boxes=False):

    print '========== Testing ======='
    model.eval()

    rel_cnt = 0.
    rel_cnt_correct = np.zeros(2)
    phrase_cnt_correct = np.zeros(2)
    pred_cnt_correct = np.zeros(2)
    total_region_rois_num = 0
    max_region_rois_num = 0
    result = []

    batch_time = network.AverageMeter()
    end = time.time()


    for i, sample in enumerate(loader): # (im_data, im_info, gt_objects, gt_relationships)
        input_visual = Variable(sample['visual'].cuda(), volatile=True)
        gt_objects = sample['objects']
        gt_relationships = sample['relations']
        image_info = sample['image_info']
        # Forward pass
        total_cnt_t, cnt_correct_t, eval_result_t = model.evaluate(
            input_visual, image_info, gt_objects, gt_relationships,
            top_Ns = top_Ns, nms=nms, triplet_nms=triplet_nms,
            use_gt_boxes=use_gt_boxes)
        eval_result_t['path'] = sample['path'][0] # for visualization
        rel_cnt += total_cnt_t
        result.append(eval_result_t)
        rel_cnt_correct += cnt_correct_t[0]
        phrase_cnt_correct += cnt_correct_t[1]
        pred_cnt_correct += cnt_correct_t[2]
        total_region_rois_num += cnt_correct_t[3]
        max_region_rois_num = cnt_correct_t[3] if cnt_correct_t[3] > max_region_rois_num else max_region_rois_num
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 500 == 0 and i > 0:
            print('[Evaluation][%d/%d][%.2fs/img][avg: %d subgraphs, max: %d subgraphs]' %(i+1, len(loader), batch_time.avg, total_region_rois_num / float(i+1), max_region_rois_num))
            for idx, top_N in enumerate(top_Ns):
                print('\tTop-%d Recall:\t[Pred] %2.3f%%\t[Phr] %2.3f%%\t[Rel] %2.3f%%' % (
                    top_N, pred_cnt_correct[idx] / float(rel_cnt) * 100,
                    phrase_cnt_correct[idx] / float(rel_cnt) * 100,
                    rel_cnt_correct[idx] / float(rel_cnt) * 100))

    recall = [rel_cnt_correct / float(rel_cnt),
              phrase_cnt_correct / float(rel_cnt),
              pred_cnt_correct / float(rel_cnt)]
    print('\n====== Done Testing ====')

    return recall, result


def test_object_detection(loader, model, nms=-1., use_gt_boxes=False):
    print '========== Testing ======='
    model.eval()
    object_classes = loader.dataset.object_classes
    result = {obj: {} for obj in object_classes}

    for i, sample in enumerate(loader): # (im_data, im_info, gt_objects, gt_relationships)
        input_visual = Variable(sample['visual'].cuda(), volatile=True)
        gt_objects = sample['objects']
        image_info = sample['image_info']
        # Forward pass
        boxes = model.evaluate_object_detection( input_visual, image_info, gt_objects,nms=nms, use_gt_boxes=use_gt_boxes)
        filename = osp.splitext(sample['path'][0])[0] # for visualization
        assert len(boxes) == len(result), "The two should have same length (object categories)"
        for j, obj_class in enumerate(object_classes):
            if j == 0:
                continue
            result[obj_class][filename] = boxes[j]
        if (i + 1) % 500 == 0 and i > 0:
            print('[Evaluation][%d/%d] processed.' %(i+1, len(loader)))

    return result
