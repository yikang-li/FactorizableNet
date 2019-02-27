import os
import os.path as osp
import shutil
import time
import random
import numpy as np
import numpy.random as npr
import argparse
import yaml
import click
from pprint import pprint
# To restore the testing results for further analysis
import cPickle


import torch

from lib import network
from lib.utils.timer import Timer
import lib.datasets as datasets
from lib.utils.FN_utils import get_model_name, group_features, get_optimizer
import lib.utils.general_utils as utils
import lib.utils.logger as logger
import models
from models.HDN_v2.utils import save_checkpoint, load_checkpoint, save_results, save_detections

from models.modules.dataParallel import DataParallel


import pdb

# # To log the training process
# from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser('Options for training Hierarchical Descriptive Model in pytorch')

parser.add_argument('--path_opt', default='options/FN_v4/map_v2.yaml', type=str,
                    help='path to a yaml options file')

parser.add_argument('--dir_logs', type=str, help='dir logs')
parser.add_argument('--model_name', type=str, help='model name prefix')
parser.add_argument('--dataset_option', type=str, help='data split selection [small | fat | normal]')
parser.add_argument('--workers', type=int, default=4, help='#idataloader workers')

# Training parameters
parser.add_argument('-lr', '--learning_rate', type=float, help='initial learning rate')
parser.add_argument('--epochs', type=int, metavar='N', help='max iterations for training')
parser.add_argument('--eval_epochs', type=int, default= 1, help='Number of epochs to evaluate the model')
parser.add_argument('--print_freq', type=int, default=1000, help='Interval for Logging')
parser.add_argument('--step_size', type=int, help='Step size for decay learning rate')
parser.add_argument('--optimizer', type=int, choices=range(0, 3), help='Step size for decay learning rate')
parser.add_argument('-i', '--infinite', action='store_true', help='To enable infinite training')
parser.add_argument('--iter_size', type=int, default=1, help='Iteration size to update parameters')
parser.add_argument('--loss_weight', default=True)
parser.add_argument('--disable_loss_weight', dest='loss_weight', action='store_false', help='Set the dropout rate.')
parser.add_argument('--clip_gradient', default=True)
parser.add_argument('--disable_clip_gradient', dest='clip_gradient', action='store_false', help='Enable clip gradient')


# Model parameters
parser.add_argument('--MPS_iter', type=int, help='Message passing iterations')
parser.add_argument('--dropout', type=float, help='Set the dropout rate.')

# model init
parser.add_argument('--resume', type=str, help='path to latest checkpoint')
parser.add_argument('--pretrained_model', type=str, help='path to pretrained_model')
parser.add_argument('--warm_iters', type=int, default=-1, help='Indicate the model do not need')
parser.add_argument('--optimize_MPS', action='store_true', help='Only optimize the MPS part')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--save_all_from', type=int,
                    help='''delete the preceding checkpoint until an epoch,'''
                         ''' then keep all (useful to save disk space)')''')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation and test set')
parser.add_argument('--evaluate_object', dest='evaluate_object', action='store_true',
                    help='Evaluate model with object detection')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')


# structure settings
# Environment Settings
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--rpn', type=str, help='The Model used for initialize')
parser.add_argument('--nms', type=float, default=-1., help='NMS threshold for post object NMS (negative means not NMS)')
parser.add_argument('--triplet_nms', type=float, default=0.4, help='Triplet NMS threshold for post object NMS (negative means not NMS)')
# parser.add_argument('--rerank', action='store_true', help='Whether to rerankt the object score')
# testing settings
parser.add_argument('--use_gt_boxes', action='store_true', help='Use ground truth bounding boxes for evaluation')

args = parser.parse_args()
# Overall loss logger
overall_train_loss = network.AverageMeter()
overall_train_rpn_loss = network.AverageMeter()
overall_gradients_norm_logger = network.LoggerMeter()

is_best = False
best_recall = [0., 0.]
best_recall_phrase = [0., 0.]
best_recall_pred = [0., 0.]




def main():
    global args, is_best, best_recall, best_recall_pred, best_recall_phrase
    # To set the model name automatically

    # Set options
    options = {
        'logs': {
            'model_name': args.model_name,
            'dir_logs': args.dir_logs,
        },
        'data':{
            'dataset_option': args.dataset_option,
            'batch_size': torch.cuda.device_count(),
        },
        'optim': {
            'lr': args.learning_rate,
            'epochs': args.epochs,
            'lr_decay_epoch': args.step_size,
            'optimizer': args.optimizer,
            'clip_gradient': args.clip_gradient,
        },
        'model':{
            'MPS_iter': args.MPS_iter,
            'dropout': args.dropout,
            'use_loss_weight': args.loss_weight,
        },
    }

    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        options = utils.update_values(options, options_yaml)
        with open(options['data']['opts'], 'r') as f:
            data_opts = yaml.load(f)
            options['data']['dataset_version'] = data_opts.get('dataset_version', None)
            options['opts'] = data_opts

    print '## args'
    pprint(vars(args))
    print '## options'
    pprint(options)

    lr = options['optim']['lr']
    options = get_model_name(options)
    print 'Checkpoints are saved to: {}'.format(options['logs']['dir_logs'])

    # To set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)

    print("Loading training set and testing set..."),
    train_set = getattr(datasets, options['data']['dataset'])(data_opts, 'train',
                                dataset_option=options['data'].get('dataset_option', None),
                                use_region=options['data'].get('use_region', False),)
    test_set = getattr(datasets, options['data']['dataset'])(data_opts, 'test',
                                dataset_option=options['data'].get('dataset_option', None),
                                use_region=options['data'].get('use_region', False))
    print("Done")

    # Model declaration
    model = getattr(models, options['model']['arch'])(train_set, opts = options['model'])

    # pass enough message for anchor target generation
    train_set._feat_stride = model.rpn._feat_stride
    train_set._rpn_opts = model.rpn.opts
    print("Done.")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=options['data']['batch_size'],
                                                shuffle=True, num_workers=args.workers,
                                                pin_memory=True,
                                                collate_fn=getattr(datasets, options['data']['dataset']).collate, 
                                                drop_last=True,)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                shuffle=False, num_workers=args.workers,
                                                pin_memory=True,
                                                collate_fn=getattr(datasets, options['data']['dataset']).collate)


    ## For debug
    # params = list(net.parameters())
    # for param in params:
    #     print param.size()
    # print net
    # To group up the features
    vgg_features_fix, vgg_features_var, rpn_features, hdn_features, mps_features = group_features(model, has_RPN=True)

    network.set_trainable(model, False)
    exp_logger = None
    # 1. only optimize MPS
    if args.optimize_MPS:
        print('Optimize the MPS part ONLY.')
        assert args.pretrained_model, 'Please specify the [pretrained_model]'
        print('Loading pretrained model: {}'.format(args.pretrained_model))
        network.load_net(args.pretrained_model, model)
        args.train_all = False
        optimizer = get_optimizer(lr, 3, options, vgg_features_var, rpn_features, hdn_features, mps_features)
    # 2. resume training
    elif args.resume is not None:
        print('Loading saved model: {}'.format(os.path.join(options['logs']['dir_logs'], args.resume)))
        args.train_all = True
        optimizer = get_optimizer(lr, 2, options, vgg_features_var, rpn_features, hdn_features, mps_features)
        args.start_epoch, best_recall[0], exp_logger = load_checkpoint(model, optimizer,
            os.path.join(options['logs']['dir_logs'], args.resume))
    else:
        if os.path.isdir(options['logs']['dir_logs']):
            if click.confirm('Logs directory already exists in {}. Erase?'
                .format(options['logs']['dir_logs'], default=False)):
                os.system('rm -r ' + options['logs']['dir_logs'])
            else:
                return
        os.system('mkdir -p ' + options['logs']['dir_logs'])
        path_new_opt = os.path.join(options['logs']['dir_logs'],
                       os.path.basename(args.path_opt))
        path_args = os.path.join(options['logs']['dir_logs'], 'args.yaml')
        with open(path_new_opt, 'w') as f:
            yaml.dump(options, f, default_flow_style=False)
        with open(path_args, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

        # 3. If we have some initialization points
        if args.pretrained_model is not None:
            print('Loading pretrained model: {}'.format(args.pretrained_model))
            args.train_all = True
            network.load_net(args.pretrained_model, model)
            optimizer = get_optimizer(lr, 2, options, vgg_features_var, rpn_features, hdn_features, mps_features)
        # 4. training with pretrained RPN
        elif args.rpn is not None:
            print('Loading pretrained RPN: {}'.format(args.rpn))
            args.train_all = False
            network.load_net(args.rpn, model.rpn)
            optimizer = get_optimizer(lr, 2, options, vgg_features_var, rpn_features, hdn_features, mps_features)
            # if args.warm_iters < 0:
            #     args.warm_iters = options['optim']['lr_decay_epoch'] // 2
        # 5. train in an end-to-end manner: no RPN given
        else:
            print('\n*** End-to-end Training ***\n'.format(args.rpn))
            args.train_all = True
            optimizer = get_optimizer(lr, 0, options, vgg_features_var, rpn_features, hdn_features, mps_features)
            if args.warm_iters < 0:
                args.warm_iters = options['optim']['lr_decay_epoch']

        assert args.start_epoch == 0, 'Set [start_epoch] to 0, or something unexpected will happen.'

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                    step_size=options['optim']['lr_decay_epoch'],
                    gamma=options['optim']['lr_decay'])
    # Setting the state of the training model
    model = DataParallel(model)
    model.cuda()
    model.train()


    # Set loggers
    if exp_logger is None:
        exp_name = os.path.basename(options['logs']['dir_logs']) # add timestamp
        exp_logger = logger.Experiment(exp_name, options)
        exp_logger.add_meters('train', make_meters())
        exp_logger.add_meters('test', make_meters())
        exp_logger.info['model_params'] = utils.params_count(model)
        print('Model has {} parameters'.format(exp_logger.info['model_params']))

        # logger_path = "log/logger/{}".format(args.model_name)
        # if os.path.exists(logger_path):
        #     shutil.rmtree(logger_path)
        # configure(logger_path, flush_secs=5) # setting up the logger


    #  network.weights_normal_init(net, dev=0.01)
    top_Ns = [50, 100]

    if args.evaluate:
        recall, result = model.module.engines.test(test_loader, model, top_Ns,
                                            nms=args.nms,
                                            triplet_nms=args.triplet_nms,
                                            use_gt_boxes=args.use_gt_boxes)
        print('======= Testing Result =======')
        for idx, top_N in enumerate(top_Ns):
            print('Top-%d Recall'
                  '\t[Pred]: %2.3f%%'
                  '\t[Phr]: %2.3f%%'
                  '\t[Rel]: %2.3f%%' % (
                    top_N, float(recall[2][idx]) * 100,
                    float(recall[1][idx]) * 100,
                    float(recall[0][idx]) * 100))
        print('============ Done ============')
        save_results(result, None, options['logs']['dir_logs'], is_testing=True)
        return

    if args.evaluate_object:
        result = model.module.engines.test_object_detection(test_loader, model, nms=args.nms, use_gt_boxes=args.use_gt_boxes)
        print('============ Done ============')
        path_dets = save_detections(result, None, options['logs']['dir_logs'], is_testing=True)
        print('Evaluating...')
        python_eval(path_dets, osp.join(data_opts['dir'], 'object_xml'))
        return

    print '========== [Start Training] ==========\n'

    FLAG_infinite = False
    loop_counter = 0
    _ = None # for useless assignment
    # infinite training scheme
    while True:
        if FLAG_infinite: # not the first loop
            if not args.infinite:
                print('Infinite Training is disabled. Done.')
                break
            loop_counter += 1
            args.train_all = True
            optimizer = get_optimizer(lr, 2, options, vgg_features_var, rpn_features, hdn_features, mps_features)
            args.start_epoch, _, exp_logger = load_checkpoint(model, optimizer,
                os.path.join(options['logs']['dir_logs'], 'ckpt'))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                    step_size=options['optim']['lr_decay_epoch'],
                    gamma=options['optim']['lr_decay'])
            options['optim']['epochs'] = args.start_epoch  + options['optim']['lr_decay_epoch'] * 5
            args.iter_size *= 2
            print('========= [{}] loop ========='.format(loop_counter) )
            print('[epoch {}] to [epoch {}]'.format(args.start_epoch, options['optim']['epochs'] ))
            print('[iter_size]: {}'.format(args.iter_size))

        FLAG_infinite = True
        for epoch in range(args.start_epoch, options['optim']['epochs']):
            # Training
            scheduler.step()
            print('[Learning Rate]\t{}'.format(optimizer.param_groups[0]['lr']))
            is_best=False
            model.module.engines.train(train_loader, model, optimizer, exp_logger, epoch, args.train_all, args.print_freq,
                clip_gradient=options['optim']['clip_gradient'], iter_size=args.iter_size)
            if (epoch + 1) % args.eval_epochs == 0:
                print('\n============ Epoch {} ============'.format(epoch))
                recall, result = model.module.engines.test(test_loader, model, top_Ns,
                                                                    nms=args.nms,
                                                                    triplet_nms=args.triplet_nms)
                # save_results(result, epoch, options['logs']['dir_logs'], is_testing=False)
                is_best = (recall[0] > best_recall).all()
                best_recall = recall[0] if is_best else best_recall
                best_recall_phrase = recall[1] if is_best else best_recall_phrase
                best_recall_pred = recall[2] if is_best else best_recall_pred
                print('\n[Result]')
                for idx, top_N in enumerate(top_Ns):
                    print('\tTop-%d Recall'
                          '\t[Pred]: %2.3f%% (best: %2.3f%%)'
                          '\t[Phr]: %2.3f%% (best: %2.3f%%)'
                          '\t[Rel]: %2.3f%% (best: %2.3f%%)' % (
                            top_N, float(recall[2][idx]) * 100, float(best_recall_pred[idx]) * 100,
                            float(recall[1][idx]) * 100, float(best_recall_phrase[idx]) * 100,
                            float(recall[0][idx]) * 100, float(best_recall[idx]) * 100 ))

                save_checkpoint({
                        'epoch': epoch,
                        'arch': options['model']['arch'],
                        'exp_logger': exp_logger,
                        'best_recall': best_recall[0],
                    },
                    model.module, #model.module.state_dict(),
                    optimizer.state_dict(),
                    options['logs']['dir_logs'],
                    args.save_all_from,
                    is_best)
                print('====================================')


            # updating learning policy
            if (epoch + 1) == args.warm_iters:
                print('Free the base CNN part.')
                # options['optim']['clip_gradient'] = False
                args.train_all = True
                # update optimizer and correponding requires_grad state
                optimizer = get_optimizer(lr, 2, options,
                        vgg_features_var, rpn_features, hdn_features, mps_features)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                step_size=options['optim']['lr_decay_epoch'],
                                gamma=options['optim']['lr_decay'])



def make_meters():
    meters_dict = {
        'loss': logger.AvgMeter(),
        'loss_rpn': logger.AvgMeter(),
        'loss_cls_obj': logger.AvgMeter(),
        'loss_reg_obj': logger.AvgMeter(),
        'loss_cls_rel': logger.AvgMeter(),
        'loss_cls_cap': logger.AvgMeter(),
        'loss_reg_cap': logger.AvgMeter(),
        'loss_cls_objectiveness': logger.AvgMeter(),
        'batch_time': logger.AvgMeter(),
        'data_time': logger.AvgMeter(),
        'epoch_time': logger.SumMeter(),
        'best_recall': logger.AvgMeter(),
    }
    return meters_dict

if __name__ == '__main__':
    main()
