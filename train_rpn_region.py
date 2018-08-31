import os
import torch
import numpy as np
import time
import yaml
import cPickle as pickle

from lib import network
from models.RPN import RPN_region as RPN # Hierarchical_Descriptive_Model
from lib.utils.timer import Timer
from lib.utils.metrics import check_recall
from lib.network import np_to_variable

from lib.datasets.visual_genome_loader import visual_genome
import argparse
from models.RPN import utils as RPN_utils

from torch.autograd import Variable

import pdb

parser = argparse.ArgumentParser('Options for training RPN_region in pytorch')

## training settings
parser.add_argument('--path_data_opts', type=str, default='options/data.yaml', help='Use options for ' )
parser.add_argument('--lr', type=float, default=0.01, help='To disable the Lanuage Model ')
parser.add_argument('--max_epoch', type=int, default=15, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=500, help='Interval for Logging')
parser.add_argument('--disable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')
parser.add_argument('--step_size', type=int, default=2, help='step to decay the learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='#images per batch')
parser.add_argument('--workers', type=int, default=4)
## Environment Settings
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/RPN', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='RPN_region', help='model name for snapshot')
parser.add_argument('--resume', type=str, help='The model we resume')
parser.add_argument('--path_rpn_opts', type=str, default='options/RPN/RPN_FN_v3.yaml', help='Path to RPN opts')
parser.add_argument('--evaluate', action='store_true', help='To enable the evaluate mode')
parser.add_argument('--dump_name', type=str, default='RPN_region_rois')
args = parser.parse_args()

def main():
    global args
    print "Loading training set and testing set..."
    with open(args.path_data_opts, 'r') as f:
        data_opts = yaml.load(f)
    args.model_name += '_' + data_opts['dataset_version'] + '_' + args.dataset_option
    train_set = visual_genome(data_opts, 'train', dataset_option=args.dataset_option,
                                batch_size=args.batch_size, use_region=True)
    test_set = visual_genome(data_opts, 'test', dataset_option=args.dataset_option,
                                batch_size=args.batch_size, use_region=True)
    print "Done."

    with open(args.path_rpn_opts, 'r') as f:
        opts = yaml.load(f)
        opts['scale'] = train_set.opts['test']['SCALES'][0]
    net = RPN(opts)

    # pass enough message for anchor target generation
    train_set._feat_stride = net._feat_stride
    train_set._rpn_opts = net.opts

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                shuffle=False if args.evaluate else True, num_workers=args.workers,
                                                pin_memory=True, collate_fn=visual_genome.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers,
                                                pin_memory=True, collate_fn=visual_genome.collate)

    if args.resume is not None:
        print('Resume training from: {}'.format(args.resume))
        RPN_utils.load_checkpoint(args.resume, net)
        optimizer = torch.optim.SGD([
                {'params': list(net.parameters())[26:]},
                ], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    else:
        print('Training from scratch.')
        optimizer = torch.optim.SGD(list(net.parameters())[26:], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

    network.set_trainable(net.features, requires_grad=False)
    net.cuda()

    if args.evaluate:
        # evaluate training set
        data_dir =os.path.join(data_opts['dir'], 'vg_cleansing', 'output', data_opts['dataset_version'])
        filename=args.dump_name + '_' + args.dataset_option
        net.eval()
        evaluate(test_loader, net,
                        path=os.path.join(data_dir, filename),
                        dataset='test')
        return

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    best_recall = np.array([0.0, 0.0])

    for epoch in range(0, args.max_epoch):

        # Training
        train(train_loader, net, optimizer, epoch)

        # Testing
        net.eval()
        recall, _, _ = test(test_loader, net)
        print('Epoch[{epoch:d}]: '
              'Recall: '
              'object: {recall[0]: .3f}%% (Best: {best_recall[0]: .3f}%%)'
              'region: {recall[1]: .3f}%% (Best: {best_recall[1]: .3f}%%)'.format(
                epoch = epoch, recall=recall * 100, best_recall=best_recall * 100))
        # update learning rate
        if epoch % args.step_size == 0 and epoch > 0:
            args.disable_clip_gradient = True
            args.lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        save_name = os.path.join(args.output_dir, '{}'.format(args.model_name))
        RPN_utils.save_checkpoint(save_name, net, epoch, np.all(recall > best_recall))
        best_recall = recall if np.all(recall > best_recall) else best_recall




def train(train_loader, target_net, optimizer, epoch):
    batch_time = network.AverageMeter()
    data_time = network.AverageMeter()
    train_loss = network.AverageMeter()
    train_loss_obj_box = network.AverageMeter()
    train_loss_obj_entropy = network.AverageMeter()
    train_loss_reg_box = network.AverageMeter()
    train_loss_reg_entropy = network.AverageMeter()

    accuracy_obj = network.AccuracyMeter()
    accuracy_reg = network.AccuracyMeter()

    target_net.train()
    end = time.time()

    for i, sample in enumerate(train_loader):
        # measure the data loading time
        data_time.update(time.time() - end)
        im_data = Variable(sample['visual'].cuda())
        im_info = sample['image_info']
        gt_objects = sample['objects']
        gt_regions = sample['regions']
        anchor_targets_obj = [
                np_to_variable(sample['rpn_targets']['object'][0],is_cuda=True, dtype=torch.LongTensor),
                np_to_variable(sample['rpn_targets']['object'][1],is_cuda=True),
                np_to_variable(sample['rpn_targets']['object'][2],is_cuda=True),
                np_to_variable(sample['rpn_targets']['object'][3],is_cuda=True)
                ]
        anchor_targets_region = [
                np_to_variable(sample['rpn_targets']['region'][0],is_cuda=True, dtype=torch.LongTensor),
                np_to_variable(sample['rpn_targets']['region'][1],is_cuda=True),
                np_to_variable(sample['rpn_targets']['region'][2],is_cuda=True),
                np_to_variable(sample['rpn_targets']['region'][3],is_cuda=True)
                ]
        # Forward pass
        target_net(im_data, im_info,
                    rpn_data_obj=anchor_targets_obj, rpn_data_region=anchor_targets_region)
        # record loss
        loss = target_net.loss
        # total loss
        train_loss.update(loss.data[0], im_data.size(0))
        # object bbox reg
        train_loss_obj_box.update(target_net.loss_box_obj.data[0], im_data.size(0))
        # object score
        train_loss_obj_entropy.update(target_net.loss_cls_obj.data[0], im_data.size(0))
        # region bbox reg
        train_loss_reg_box.update(target_net.loss_box_region.data[0], im_data.size(0))
        # region score
        train_loss_reg_entropy.update(target_net.loss_cls_region.data[0], im_data.size(0))
        # accuracy
        accuracy_obj.update(target_net.tp, target_net.tf, target_net.fg_cnt, target_net.bg_cnt)
        accuracy_reg.update(target_net.tp_region, target_net.tf_region, target_net.fg_cnt_region, target_net.bg_cnt_region)

        # backward
        optimizer.zero_grad()
        torch.cuda.synchronize()
        loss.backward()
        if not args.disable_clip_gradient:
            network.clip_gradient(target_net, 10.)
        torch.cuda.synchronize()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if  (i + 1) % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch_Time: {batch_time.avg:.3f}s\t'
                  'lr: {lr: f}\t'
                  'Loss: {loss.avg:.4f}\n'
                  '\t[object]:\t'
                  'tp: {accuracy_obj.true_pos:.2f}, \t'
                  'tf: {accuracy_obj.true_neg:.2f}, \t'
                  'fg/bg=({accuracy_obj.foreground:.1f}/{accuracy_obj.background:.1f})\t'
                  'cls_loss: {cls_loss_object.avg:.3f}\t'
                  'reg_loss: {reg_loss_object.avg:.3f}\n'
                  '\t[region]:\t'
                  'tp: {accuracy_reg.true_pos:.3f}, \t'
                  'tf: {accuracy_reg.true_neg:.3f}, \t'
                  'fg/bg=({accuracy_reg.foreground:.1f}/{accuracy_reg.background:.1f})\t'
                  'cls_loss: {cls_loss_region.avg:.3f}\t'
                  'reg_loss: {reg_loss_region.avg:.3f}'.format(
                   epoch, i + 1, len(train_loader), batch_time=batch_time,lr=args.lr,
                   data_time=data_time, loss=train_loss,
                   cls_loss_object=train_loss_obj_entropy, reg_loss_object=train_loss_obj_box,
                   cls_loss_region=train_loss_reg_entropy, reg_loss_region=train_loss_reg_box,
                   accuracy_reg=accuracy_reg, accuracy_obj=accuracy_obj))



def test(test_loader, target_net):
    box_num = np.array([0, 0])
    correct_cnt, total_cnt = np.array([0, 0]), np.array([0, 0])
    print '========== Testing ======='
    results_obj = []
    results_region = []

    batch_time = network.AverageMeter()
    end = time.time()
    im_counter = 0
    for i, sample in enumerate(test_loader):
        correct_cnt_t, total_cnt_t = np.array([0, 0]), np.array([0, 0])
        # Forward pass

        # measure the data loading time
        im_data = Variable(sample['visual'].cuda(), volatile=True)
        im_counter += im_data.size(0)
        im_info = sample['image_info']
        gt_objects = sample['objects']
        gt_regions = sample['regions']
        object_rois, region_rois = target_net(im_data, im_info)[1:]
        results_obj.append(object_rois.cpu().data.numpy())
        results_region.append(region_rois.cpu().data.numpy())
        box_num[0] += object_rois.size(0)
        box_num[1] += region_rois.size(0)
        correct_cnt_t[0], total_cnt_t[0] = check_recall(object_rois, gt_objects, 50)
        correct_cnt_t[1], total_cnt_t[1] = check_recall(region_rois, gt_regions, 50)
        correct_cnt += correct_cnt_t
        total_cnt += total_cnt_t
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 100 == 0 and i > 0:
            print('[{0}/{10}]  Time: {1:2.3f}s/img).'
                  '\t[object] Avg: {2:2.2f} Boxes/im, Top-50 recall: {3:2.3f} ({4:d}/{5:d})'
                  '\t[region] Avg: {6:2.2f} Boxes/im, Top-50 recall: {7:2.3f} ({8:d}/{9:d})'.format(
                    i + 1, batch_time.avg,
                    box_num[0] / float(im_counter), correct_cnt[0] / float(total_cnt[0])* 100, correct_cnt[0], total_cnt[0],
                    box_num[1] / float(im_counter), correct_cnt[1] / float(total_cnt[1])* 100, correct_cnt[1], total_cnt[1],
                    len(test_loader)))

    recall = correct_cnt / total_cnt.astype(np.float)
    print '====== Done Testing ===='
    return recall, results_obj, results_region

def evaluate(loader, net, path, dataset='train'):

    recall, rois_obj, rois_region = test(loader, net)
    print('[{}]\tRecall: '
                'object: {recall_obj: .3f}%%'
                'region: {recall_reg: .3f}%%'.format(
                    dataset, recall_obj=recall[0] * 100, recall_reg=recall[1]*100))
    print('Saving ROIs...'),
    with open(path + '_object_' + dataset + '.pkl', 'wb') as f:
        pickle.dump(rois_obj, f)
    with open(path + '_region_' + dataset + '.pkl', 'wb') as f:
        pickle.dump(rois_region, f)
    print('Done.')

if __name__ == '__main__':
    main()
