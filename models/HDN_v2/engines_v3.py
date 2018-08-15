import pdb
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

import lib.network as network
from lib.network import np_to_variable
from engines_v2 import test

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
        target_regions = sample['regions']
        image_info = sample['image_info']
        # RPN targets
        rpn_anchor_targets_obj = [
                np_to_variable(sample['rpn_targets']['object'][0],is_cuda=True, dtype=torch.LongTensor),
                np_to_variable(sample['rpn_targets']['object'][1],is_cuda=True),
                np_to_variable(sample['rpn_targets']['object'][2],is_cuda=True),
                np_to_variable(sample['rpn_targets']['object'][3],is_cuda=True)
                ]
        rpn_anchor_targets_reg = [
                np_to_variable(sample['rpn_targets']['region'][0],is_cuda=True, dtype=torch.LongTensor),
                np_to_variable(sample['rpn_targets']['region'][1],is_cuda=True),
                np_to_variable(sample['rpn_targets']['region'][2],is_cuda=True),
                np_to_variable(sample['rpn_targets']['region'][3],is_cuda=True)
                ]

        # compute output
        model(input_visual, image_info,
                gt_objects=target_objects,
                gt_relationships=target_relations,
                gt_regions=target_regions,
                rpn_anchor_targets_obj=rpn_anchor_targets_obj,
                rpn_anchor_targets_reg=rpn_anchor_targets_reg)

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
        meters['loss_cls_cap'].update(model.loss_caption_region.data.cpu().numpy()[0], n=batch_size)
        meters['loss_reg_cap'].update(model.loss_reg_region.data.cpu().numpy()[0], n=batch_size)
        meters['loss_cls_objectiveness'].update(model.loss_objectiveness_region.data.cpu().numpy()[0], n=batch_size)
        meters['batch_time'].update(time.time() - end, n=batch_size)
        meters['epoch_time'].update(meters['batch_time'].val, n=batch_size)

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
            print('\t[region] loss_cls_cap: {loss_cls_cap.avg:.4f} '
                  'loss_reg_cap: {loss_reg_cap.avg:.4f} '
                  'loss_cls_objectiveness: {loss_cls_objectiveness.avg:.4f} '.format(
                  loss_cls_cap = meters['loss_cls_cap'],
                  loss_reg_cap = meters['loss_reg_cap'],
                  loss_cls_objectiveness = meters['loss_cls_objectiveness'], ))

    exp_logger.log_meters('train', n=epoch)



# def test(loader, model, top_Ns, nms=-1., triplet_nms=-1., use_gt_boxes=False):

#     print '========== Testing ======='
#     model.eval()

#     rel_cnt = 0.
#     rel_cnt_correct = np.zeros(2)
#     pred_cnt_correct = np.zeros(2)
#     result = np.zeros((len(loader), 5))

#     batch_time = network.AverageMeter()
#     end = time.time()


#     for i, sample in enumerate(loader): # (im_data, im_info, gt_objects, gt_relationships)
#         input_visual = Variable(sample['visual'].cuda(), volatile=True)
#         gt_objects = sample['objects']
#         gt_relationships = sample['relations']
#         gt_regions = sample['regions']
#         image_info = sample['image_info']
#         # Forward pass
#         total_cnt_t, rel_cnt_correct_t, pred_cnt_correct_t = model.evaluate(
#             input_visual, image_info, gt_objects, gt_relationships, gt_regions,
#             top_Ns = top_Ns, nms=nms, triplet_nms=triplet_nms,
#             use_gt_boxes=use_gt_boxes)
#         rel_cnt += total_cnt_t
#         result[i, 0] = total_cnt_t
#         result[i, 1:3] = rel_cnt_correct_t
#         result[i, 3:] = pred_cnt_correct_t
#         rel_cnt_correct += rel_cnt_correct_t
#         pred_cnt_correct += pred_cnt_correct_t
#         batch_time.update(time.time() - end)
#         end = time.time()
#         if (i + 1) % 500 == 0 and i > 0:
#             print('[Evaluation][%d/%d][%.2fs/img]' %(i+1, len(loader), batch_time.avg))
#             for idx, top_N in enumerate(top_Ns):
#                 print('\tTop-%d Recall:\t[Pred] %2.3f%%\t[Rel] %2.3f%%' % (
#                     top_N, pred_cnt_correct[idx] / float(rel_cnt) * 100,
#                     rel_cnt_correct[idx] / float(rel_cnt) * 100))

#     recall = rel_cnt_correct / rel_cnt
#     recall_pred = pred_cnt_correct / rel_cnt
#     print('\n====== Done Testing ====')

#     return recall, recall_pred, result
