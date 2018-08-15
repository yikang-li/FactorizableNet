import os
import os.path as osp
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from .cython_bbox import bbox_overlaps, bbox_intersections

from ..network import set_trainable_param


def group_features(net_, has_RPN=True, group_MPS=False):
    if has_RPN:
        vgg_features_fix = list(net_.rpn.features.parameters())[:8]
        vgg_features_var = list(net_.rpn.features.parameters())[8:]
        vgg_feature_len = len(list(net_.rpn.features.parameters()))
        rpn_feature_len = len(list(net_.rpn.parameters())) - vgg_feature_len
        rpn_features = list(net_.rpn.parameters())[vgg_feature_len:]
        hdn_features = list(net_.parameters())[(rpn_feature_len + vgg_feature_len):]
        mps_features = list(net_.mps_list.parameters())
        hdn_features = list(set(hdn_features) - set(mps_features))
        print 'vgg feature length:', vgg_feature_len
        print 'rpn feature length:', rpn_feature_len
        print 'HDN feature length:', len(hdn_features)
        print 'MPS feature length:', len(mps_features)
        return vgg_features_fix, vgg_features_var, rpn_features, hdn_features, mps_features
    else:
        raise NotImplementedError
        vgg_features_fix = list(net_.features.parameters())[:8]
        vgg_features_var = list(net_.features.parameters())[8:]
        vgg_feature_len = len(list(net_.features.parameters()))
        hdn_features = list(net_.parameters())[vgg_feature_len:]
        print 'vgg feature length:', vgg_feature_len
        print 'HDN feature length:', len(hdn_features)
        return vgg_features_fix, vgg_features_var, [], hdn_features





def get_optimizer(lr, mode, opts, vgg_features_var, rpn_features, hdn_features, mps_features=[]):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: training with RPN
    mode 2: resume training
    """
    if mode == 0:
        hdn_features += mps_features
        set_trainable_param(vgg_features_var, False)
        set_trainable_param(rpn_features, True)
        set_trainable_param(hdn_features, True)
        if opts['optim']['optimizer'] == 0:
            optimizer = torch.optim.SGD([
                {'params': rpn_features},
                {'params': hdn_features},
                ], lr=lr, momentum=opts['optim']['momentum'], weight_decay=0.0005, nesterov=opts['optim']['nesterov'])
        elif opts['optim']['optimizer'] == 1:
            optimizer = torch.optim.Adam([
                {'params': rpn_features},
                {'params': hdn_features},
                ], lr=lr, weight_decay=0.0005)
        elif opts['optim']['optimizer'] == 2:
            optimizer = torch.optim.Adagrad([
                {'params': rpn_features},
                {'params': hdn_features},
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')

    elif mode == 1:
        hdn_features += mps_features
        set_trainable_param(vgg_features_var, False)
        set_trainable_param(rpn_features, False)
        set_trainable_param(hdn_features, True)
        if opts['optim']['optimizer'] == 0:
            optimizer = torch.optim.SGD([
                {'params': hdn_features},
                ], lr=lr, momentum=opts['optim']['momentum'], weight_decay=0.0005, nesterov=opts['optim']['nesterov'])
        elif opts['optim']['optimizer'] == 1:
            optimizer = torch.optim.Adam([
                {'params': hdn_features},
                ], lr=lr, weight_decay=0.0005)
        elif opts['optim']['optimizer'] == 2:
            optimizer = torch.optim.Adagrad([
                {'params': hdn_features},
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')


    elif mode == 2:
        hdn_features += mps_features
        set_trainable_param(rpn_features, True)
        set_trainable_param(vgg_features_var, True)
        set_trainable_param(hdn_features, True)
        if opts['optim']['optimizer'] == 0:
            optimizer = torch.optim.SGD([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.1},
                {'params': hdn_features},
                ], lr=lr, momentum=opts['optim']['momentum'], weight_decay=0.0005, nesterov=opts['optim']['nesterov'])
        elif opts['optim']['optimizer'] == 1:
            optimizer = torch.optim.Adam([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.1},
                {'params': hdn_features},
                ], lr=lr, weight_decay=0.0005)
        elif opts['optim']['optimizer'] == 2:
            optimizer = torch.optim.Adagrad([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.1},
                {'params': hdn_features},
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')

    elif mode == 3:
        # separately optimize [MPS] and [other HDN] parameters
        assert len(mps_features), 'MPS features should be separated.'
        print('Only optimizing [MPS] part.')
        set_trainable_param(vgg_features_var, False)
        set_trainable_param(rpn_features, False)
        set_trainable_param(hdn_features, True) # [TODO] whether needed to guarantee the backpropagation of gradients
        set_trainable_param(mps_features, True)
        if opts['optim']['optimizer'] == 0:
            optimizer = torch.optim.SGD([
                {'params': mps_features},
                ], lr=lr, momentum=opts['optim']['momentum'], weight_decay=0.0005, nesterov=opts['optim']['nesterov'])
        elif opts['optim']['optimizer'] == 1:
            optimizer = torch.optim.Adam([
                {'params': mps_features},
                ], lr=lr, weight_decay=0.0005)
        elif opts['optim']['optimizer'] == 2:
            optimizer = torch.optim.Adagrad([
                {'params': mps_features},
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')

    return optimizer

# general tools
def get_model_name(opts):


    model_name = opts['logs']['model_name']
    if  opts['data'].get('dataset_version', None) is not None:
        model_name += '_' + opts['data']['dataset_version'] # + '_' + opts['model']['fusion']
    if opts['data'].get('dataset_option', None) is not None:
        model_name += '_' + opts['data']['dataset_option']

    if opts['model']['MPS_iter'] < 0:
        print 'Using random MPS iterations to training'
        model_name += '_rand_iters'
    else:
        model_name += '_{}_iters'.format(opts['model']['MPS_iter'])


    if opts['model'].get('use_kernel', False):
        model_name += '_with_kernel'

    model_name += '_SGD'
    # if opts['optim']['optimizer'] == 0:
    #     model_name += '_SGD'
    #     opts['optim']['solver'] = 'SGD'
    # elif opts['optim']['optimizer'] == 1:
    #     model_name += '_Adam'
    #     opts['optim']['solver'] = 'Adam'
    # elif opts['optim']['optimizer'] == 2:
    #     model_name += '_Adagrad'
    #     opts['optim']['solver'] = 'Adagrad'
    # else:
    #     raise Exception('Unrecognized optimization algorithm specified!')

    opts['logs']['dir_logs'] = os.path.join(opts['logs']['dir_logs'], model_name)
    return opts


