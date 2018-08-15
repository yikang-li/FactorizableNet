import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from .cython_bbox import bbox_overlaps, bbox_intersections


def get_model_name(arguments):


    if arguments.nesterov:
        arguments.model_name += '_nesterov'

    if arguments.MPS_iter < 0:
        print 'Using random MPS iterations to training'
        arguments.model_name += '_rand_iters'
    else:
        arguments.model_name += '_{}_iters'.format(arguments.MPS_iter)


    if arguments.use_kernel_function:
        arguments.model_name += '_with_kernel'
    if arguments.load_RPN or arguments.resume_training:
        arguments.model_name += '_alt'
    else:
        arguments.model_name += '_end2end'
    if arguments.dropout:
        arguments.model_name += '_dropout'
    arguments.model_name += '_{}'.format(arguments.dataset_option)
    if arguments.disable_language_model:
        arguments.model_name += '_no_caption'
    else:
        if arguments.rnn_type == 'LSTM_im':
            arguments.model_name += '_H_LSTM'
        elif arguments.rnn_type == 'LSTM_normal':
            arguments.model_name += '_I_LSTM'
        elif arguments.rnn_type == 'LSTM_baseline':
            arguments.model_name += '_B_LSTM'
        else:
            raise Exception('Error in RNN type')
        if arguments.caption_use_bias:
            arguments.model_name += '_with_bias'
        else:
            arguments.model_name += '_no_bias'
        if arguments.caption_use_dropout > 0:
            arguments.model_name += '_with_dropout_{}'.format(arguments.caption_use_dropout).replace('.', '_')
        else:
            arguments.model_name += '_no_dropout'
        arguments.model_name += '_nembed_{}'.format(arguments.nembedding)
        arguments.model_name += '_nhidden_{}'.format(arguments.nhidden_caption)

        if arguments.region_bbox_reg:
            arguments.model_name += '_with_region_regression'

    if arguments.resume_training:
        arguments.model_name += '_resume'

    if arguments.finetune_language_model:
        arguments.model_name += '_finetune'
    if arguments.optimizer == 0:
        arguments.model_name += '_SGD'
        arguments.solver = 'SGD'
    elif arguments.optimizer == 1:
        arguments.model_name += '_Adam'
        arguments.solver = 'Adam'
    elif arguments.optimizer == 2:    
        arguments.model_name += '_Adagrad'
        arguments.solver = 'Adagrad'
    else:
        raise Exception('Unrecognized optimization algorithm specified!')

    return arguments


def group_features(net_):
    vgg_features_fix = list(net_.rpn.features.parameters())[:8]
    vgg_features_var = list(net_.rpn.features.parameters())[8:]
    vgg_feature_len = len(list(net_.rpn.features.parameters()))
    rpn_feature_len = len(list(net_.rpn.parameters())) - vgg_feature_len
    rpn_features = list(net_.rpn.parameters())[vgg_feature_len:]
    language_features = list(net_.caption_prediction.parameters())
    language_feature_len = len(language_features)
    hdn_features = list(net_.parameters())[(rpn_feature_len + vgg_feature_len):(-1 * language_feature_len)]
    print 'vgg feature length:', vgg_feature_len
    print 'rpn feature length:', rpn_feature_len
    print 'HDN feature length:', len(hdn_features)
    print 'language_feature_len:', language_feature_len
    return vgg_features_fix, vgg_features_var, rpn_features, hdn_features, language_features

