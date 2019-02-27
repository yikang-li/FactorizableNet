from __future__ import print_function
from PIL import Image
import os
import os.path as osp
import errno
import numpy as np
import numpy.random as npr
import sys
import json
import cv2

import pdb

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from lib.rpn_msr.anchor_target_layer import anchor_target_layer

class VRD(data.Dataset):
    def __init__(self, opts, image_set='train', batch_size=1, dataset_option=None, use_region=False):
        super(VRD, self).__init__()
        self._name = image_set
        self.opts = opts
        # self._batch_size = batch_size
        self._image_set = image_set
        self._data_path = osp.join(self.opts['dir'], 'images', 'sg_{}_images'.format(image_set))
        # load category names and annotations
        annotation_dir = self.opts['dir']
        inverse_weight = json.load(open(osp.join(annotation_dir, 'inverse_weight.json')))

        self.inverse_weight_object = torch.FloatTensor(inverse_weight['object'])
        self.inverse_weight_predicate = torch.FloatTensor(inverse_weight['predicate'])
        # print self.inverse_weight_predicate
        ann_file_path = osp.join(annotation_dir, self.name + '.json')
        self.annotations = json.load(open(ann_file_path))

        # categories
        obj_cats = json.load(open(osp.join(annotation_dir, 'objects.json')))
        self._object_classes = tuple(['__background__'] + obj_cats)
        pred_cats = json.load(open(osp.join(annotation_dir, 'predicates.json')))
        self._predicate_classes = tuple(['__background__'] + pred_cats)
        self._object_class_to_ind = dict(zip(self.object_classes, xrange(self.num_object_classes)))
        self._predicate_class_to_ind = dict(zip(self.predicate_classes, xrange(self.num_predicate_classes)))

        # image transformation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        self.cfg_key = image_set.split('_')[0]
        self._feat_stride = None
        self._rpn_opts = None

    def __getitem__(self, index):
        # Sample random scales to use for each image in this batch
        item = {'rpn_targets': {}}

        target_scale = self.opts[self.cfg_key]['SCALES'][npr.randint(0, high=len(self.opts[self.cfg_key]['SCALES']))]
        img = cv2.imread(osp.join(self._data_path, self.annotations[index]['path']))
        img_original_shape = img.shape
        item['path']= self.annotations[index]['path']
        img, im_scale = self._image_resize(img, target_scale, self.opts[self.cfg_key]['MAX_SIZE'])
        # restore the [image_height, image_width, scale_factor, max_size]
        item['image_info'] = np.array([img.shape[0], img.shape[1], im_scale, 
                    img_original_shape[0], img_original_shape[1]], dtype=np.float)
        item['visual'] = Image.fromarray(img)

        if self.transform is not None:
            item['visual']  = self.transform(item['visual'])

        # if self._batch_size > 1:
        #     # padding the image to MAX_SIZE, so all images can be stacked
        #     pad_h = self.opts[self.cfg_key]['MAX_SIZE'] - item['visual'].size(1)
        #     pad_w = self.opts[self.cfg_key]['MAX_SIZE'] - item['visual'].size(2)
        #     item['visual'] = F.pad(item['visual'], (0, pad_w, 0, pad_h)).data

        _annotation = self.annotations[index]
        gt_boxes_object = np.zeros((len(_annotation['objects']), 5))
        gt_boxes_object[:, 0:4] = np.array([obj['bbox'] for obj in _annotation['objects']], dtype=np.float) * im_scale
        gt_boxes_object[:, 4]   = np.array([obj['class'] for obj in _annotation['objects']])
        item['objects'] = gt_boxes_object
        if self.cfg_key == 'train': # calculate the RPN target
            item['rpn_targets']['object'] = anchor_target_layer(item['visual'], gt_boxes_object, item['image_info'],
                                self._feat_stride, self._rpn_opts['object'],
                                mappings = self._rpn_opts['mappings'])


        gt_relationships = np.zeros([len(_annotation['objects']), (len(_annotation['objects']))], dtype=np.long)
        for rel in _annotation['relationships']:
            gt_relationships[rel['sub_id'], rel['obj_id']] = rel['predicate']
        item['relations'] = gt_relationships

        return item

    @staticmethod
    def collate(items):
        batch_item = {}
        for key in items[0]:
            if key == 'visual':
                batch_item[key] = [x[key].unsqueeze(0) for x in items]
            #     out = None
            #     # If we're in a background process, concatenate directly into a
            #     # shared memory tensor to avoid an extra copy
            #     numel = sum([x[key].numel() for x in items])
            #     storage = items[0][key].storage()._new_shared(numel)
            #     out = items[0][key].new(storage)
            #     batch_item[key] = torch.stack([x[key] for x in items], 0, out=out)
            elif key == 'rpn_targets':
                batch_item[key] = {}
                for subkey in items[0][key]:
                    batch_item[key][subkey] = [x[key][subkey] for x in items]
            elif items[0][key] is not None:
                batch_item[key] = [x[key] for x in items]

        return batch_item


    def __len__(self):
        return len(self.annotations)


    @property
    def voc_size(self):
        return len(self.idx2word)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(i)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = self.annotations[index]['path']
        image_path = osp.join(self._data_path, file_name)
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _image_resize(self, im, target_size, max_size):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return im, im_scale

    @property
    def name(self):
        return self._name

    @property
    def num_object_classes(self):
        return len(self._object_classes)

    @property
    def num_predicate_classes(self):
        return len(self._predicate_classes)

    @property
    def object_classes(self):
        return self._object_classes

    @property
    def predicate_classes(self):
        return self._predicate_classes
