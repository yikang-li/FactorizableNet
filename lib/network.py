import torch
import torch.nn as nn
import numpy as np
import math
import pdb

class GroupDropout(nn.Module):
    '''
    features will be grouped long the channel-wise dimension
    To avoid the inter-group co-adaptation across channels, but keep the intra-channel one.
    if p < 0, then no dropout will be applied.
    '''
    def __init__(self, p=0.5, inplace=False, group=16):
        super(GroupDropout, self).__init__()
        self.group = group
        # cannot use inplace operation: [.view] is used
        self.inplace = False
        self.p = p

    def forward(self, x):
        if self.training and self.p > 1e-5:
            assert x.size(1) % self.group == 0, "Channels should be divided by group number [{}]".format(self.group)
            original_size = x.size()
            if x.dim() == 2:
                x = x.view(x.size(0), self.group, x.size(1) / self.group, 1)
            else:
                x = x.view(x.size(0), self.group,  x.size(1) / self.group * x.size(2), x.size(3))
            x = nn.functional.dropout2d(x, p=self.p, inplace=self.inplace, training=True)
            x = x.view(original_size)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True, bn=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn = nn.BatchNorm1d(out_features, eps=0.001, momentum=0, affine=True) if bn else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())
        #print '[Saved]: {}'.format(k)


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        try:
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                v.copy_(param)
                #print '[Copied]: {}'.format(k)
            else:
                print '[Missed]: {}'.format(k)
                print('[Manually copy instructions]: \n'
                         'check the existence of new name:\n'
                         '\t \'{}\' in h5f\n'
                         'if True, then copy\n'
                         '\t param = torch.from_numpy(np.asarray(h5f[\'{}\']))\n'
                         '\t v.copy_(param)\n'.format(k, k))
                pdb.set_trace()
        except Exception as e:
            print '[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k)
            pdb.set_trace()



def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    return np_to_tensor(x, is_cuda, dtype)

def np_to_tensor(x, is_cuda=True, dtype=torch.FloatTensor):
    v = torch.from_numpy(x).type(dtype)
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    set_trainable_param(model.parameters(), requires_grad)

def set_trainable_param(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad


def weight_init_fun_kaiming(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight.data, mode='fan_in')
        m.bias.data.fill_(0.)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_MSRA_init(model):
    if isinstance(model, list):
        for m in model:
            weights_MSRA_init(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)
    return totalnorm

def avg_gradient(model, iter_size):
    """Computes a gradient clipping coefficient based on gradient norm."""
    if iter_size >1:
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.div_(iter_size)

def get_optimizer(lr, mode, args, vgg_features_var, rpn_features, hdn_features, language_features=None):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: training with RPN
    mode 2: resume training
    mode 3: finetune language model"""

    assert False, "Please use [get_optimizer] in [lib/utils/FN_utils.py]"

    if mode == 0:
        set_trainable_param(rpn_features, True)
        set_trainable_param(hdn_features, True)
        set_trainable_param(language_features, True)
        if args.optimizer == 0:
            optimizer = torch.optim.SGD([
                {'params': rpn_features},
                {'params': hdn_features},
                {'params': language_features, 'weight_decay':0.0}
                ], lr=lr, momentum=args.momentum, weight_decay=0.0005, nesterov=args.nesterov)
        elif args.optimizer == 1:
            optimizer = torch.optim.Adam([
                {'params': rpn_features},
                {'params': hdn_features},
                {'params': language_features, 'weight_decay':0.0}
                ], lr=lr, weight_decay=0.0005)
        elif args.optimizer == 2:
            optimizer = torch.optim.Adagrad([
                {'params': rpn_features},
                {'params': hdn_features},
                {'params': language_features, 'weight_decay':0.0}
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')

    elif mode == 1:
        set_trainable_param(hdn_features, True)
        set_trainable_param(language_features, True)
        if args.optimizer == 0:
            optimizer = torch.optim.SGD([
                {'params': hdn_features},
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, momentum=args.momentum, weight_decay=0.0005, nesterov=args.nesterov)
        elif args.optimizer == 1:
            optimizer = torch.optim.Adam([
                {'params': hdn_features},
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, weight_decay=0.0005)
        elif args.optimizer == 2:
            optimizer = torch.optim.Adagrad([
                {'params': hdn_features},
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')


    elif mode == 2:
        set_trainable_param(rpn_features, True)
        set_trainable_param(vgg_features_var, True)
        set_trainable_param(hdn_features, True)
        set_trainable_param(language_features, True)
        if args.optimizer == 0:
            optimizer = torch.optim.SGD([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.1},
                {'params': hdn_features},
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, momentum=args.momentum, weight_decay=0.0005, nesterov=args.nesterov)
        elif args.optimizer == 1:
            optimizer = torch.optim.Adam([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.1},
                {'params': hdn_features},
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, weight_decay=0.0005)
        elif args.optimizer == 2:
            optimizer = torch.optim.Adagrad([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.1},
                {'params': hdn_features},
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')



    elif mode == 3:
        set_trainable_param(rpn_features, True)
        set_trainable_param(vgg_features_var, True)
        set_trainable_param(hdn_features, True)
        set_trainable_param(language_features, True)
        if args.optimizer == 0:
            optimizer = torch.optim.SGD([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.01},
                {'params': hdn_features[:-4]},
                {'params': hdn_features[-4:], 'lr': lr},
                {'params': language_features, 'weight_decay': 0.0, 'lr': lr}
                ], lr=lr * 0.1, momentum=args.momentum, weight_decay=0.0005, nesterov=args.nesterov)
        elif args.optimizer == 1:
            optimizer = torch.optim.Adam([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.01},
                {'params': hdn_features[:-4]},
                {'params': hdn_features[-4:], 'lr': lr},
                {'params': language_features, 'weight_decay': 0.0, 'lr': lr}
                ], lr=lr * 0.1, weight_decay=0.0005)
        elif args.optimizer == 2:
            optimizer = torch.optim.Adagrad([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.01},
                {'params': hdn_features[:-4]},
                {'params': hdn_features[-4:], 'lr': lr},
                {'params': language_features, 'weight_decay': 0.0, 'lr': lr}
                ], lr=lr * 0.1, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')


    return optimizer



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LoggerMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = []

    def reset(self):
        self.val.append([])

    @property
    def epoch_num(self):
        return len(self.val)

    @property
    def avg(self):
        return np.mean(self.val, 1)

    @property
    def count(self):
        return np.array(self.val).shape[0]

    @property
    def sum(self):
        return np.sum(self.val, 1)

    def update(self, val):
        self.val[-1].append(val)


class AccuracyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.
        self.tf = 0.
        self.fg = 0.
        self.bg = 0.
        self.count = 0

    def update(self, tp, tf, fg, bg, count=1):
        self.tp += tp
        self.tf += tf
        self.fg += fg
        self.bg += bg
        self.count += 1

    @property
    def true_pos(self):
        return self.tp / self.fg

    @property
    def true_neg(self):
        return self.tf / self.bg

    @property
    def foreground(self):
        return self.fg / self.count

    @property
    def background(self):
        return self.bg / self.count

