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

class visual_genome(data.Dataset):
    def __init__(self, opts, image_set='train', dataset_option='normal', batch_size=1, use_region=False):
        self.opts = opts
        self.use_region = use_region
        self._name = 'vg_' + dataset_option + '_' + image_set
        self.unknown_token='<unknown>'
        self.start_token='<start>'
        self.end_token='<end>'
        self._set_option = dataset_option
        # self._batch_size = batch_size
        self._image_set = image_set
        self._data_path = osp.join(self.opts['dir'], 'images')
        # load category names and annotations
        annotation_dir = osp.join(self.opts['dir'])
        cats = json.load(open(osp.join(annotation_dir, 'categories.json')))
        dictionary = json.load(open(osp.join(annotation_dir, 'dict.json')))
        inverse_weight = json.load(open(osp.join(annotation_dir, 'inverse_weight.json')))
        self.idx2word = dictionary['idx2word']
        self.word2idx = dictionary['word2idx']
        dict_len = len(dictionary['idx2word'])
        self.idx2word.append(self.unknown_token)
        self.idx2word.append(self.start_token)
        self.idx2word.append(self.end_token)
        self.word2idx[self.unknown_token] = dict_len
        self.word2idx[self.start_token] = dict_len + 1
        self.word2idx[self.end_token] = dict_len + 2
        self.voc_sign = {'start': self.word2idx[self.start_token],
                         'null': self.word2idx[self.unknown_token],
                         'end': self.word2idx[self.end_token]}

        self._object_classes = tuple(['__background__'] + cats['object'])
        self._predicate_classes = tuple(['__background__'] + cats['predicate'])
        self._object_class_to_ind = dict(zip(self.object_classes, xrange(self.num_object_classes)))
        self._predicate_class_to_ind = dict(zip(self.predicate_classes, xrange(self.num_predicate_classes)))
        self.inverse_weight_object = torch.ones(self.num_object_classes)
        for idx in xrange(1, self.num_object_classes):
            self.inverse_weight_object[idx] = inverse_weight['object'][self._object_classes[idx]]
        self.inverse_weight_object = self.inverse_weight_object / self.inverse_weight_object.min()
        # print self.inverse_weight_object
        self.inverse_weight_predicate = torch.ones(self.num_predicate_classes)
        for idx in xrange(1, self.num_predicate_classes):
            self.inverse_weight_predicate[idx] = inverse_weight['predicate'][self._predicate_classes[idx]]
        self.inverse_weight_predicate = self.inverse_weight_predicate / self.inverse_weight_predicate.min()
        # print self.inverse_weight_predicate
        ann_file_name = {'vg_normal_train': 'train.json',
                           'vg_normal_test': 'test.json',
                           'vg_small_train': 'train_small.json',
                           'vg_small_test': 'test_small.json',
                           'vg_fat_train': 'train_fat.json',
                           'vg_fat_test': 'test_small.json'}

        ann_file_path = osp.join(annotation_dir, ann_file_name[self.name])
        self.annotations = json.load(open(ann_file_path))
        self.max_size = 11 # including the <end> token excluding <start> token
        self.tokenize_annotations(self.max_size)


        # image transformation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        self.cfg_key = image_set
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
        gt_boxes_object[:, 0:4] = np.array([obj['box'] for obj in _annotation['objects']], dtype=np.float) * im_scale
        gt_boxes_object[:, 4]   = np.array([obj['class'] for obj in _annotation['objects']])
        item['objects'] = gt_boxes_object
        if self._image_set == 'train': # calculate the RPN target
            item['rpn_targets']['object'] = anchor_target_layer(item['visual'], gt_boxes_object, item['image_info'],
                                self._feat_stride, self._rpn_opts['object'],
                                mappings = self._rpn_opts['mappings'])


        gt_relationships = np.zeros([len(_annotation['objects']), (len(_annotation['objects']))], dtype=np.long)
        for rel in _annotation['relationships']:
            gt_relationships[rel['sub_id'], rel['obj_id']] = rel['predicate']
        item['relations'] = gt_relationships

        if self.use_region:
            gt_boxes_region = np.zeros((len(_annotation['regions']), self.max_size + 4)) # 4 for box and 40 for sentences
            gt_boxes_region[:, 0:4] = np.array([reg['box'] for reg in _annotation['regions']], dtype=np.float) * im_scale
            gt_boxes_region[:, 4:]  = np.array([np.pad(reg['phrase'],
                                    (0,self.max_size-len(reg['phrase'])),'constant',constant_values=self.voc_sign['end'])
                                        for reg in _annotation['regions']])

            item['regions'] = gt_boxes_region
            if self._image_set == 'train' and 'region' in self._rpn_opts.keys(): # calculate the RPN target
                item['rpn_targets']['region'] = anchor_target_layer(item['visual'], gt_boxes_region, item['image_info'],
                                self._feat_stride, self._rpn_opts['region'],
                                mappings = self._rpn_opts['mappings'])
        else:
            item['regions'] = None

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


    def get_regions(self, idx, length_constraint=50):
        boxes = np.array([reg['box'] for reg in self.annotations[idx]['regions']]).astype(np.float32)
        text = []
        mask_ = np.ones(boxes.shape[0], dtype=np.bool)
        for i in range(len(self.annotations[idx]['regions'])):
            reg = self.annotations[idx]['regions'][i]
            if len(reg['phrase']) > length_constraint:
                mask_[i] = False
                continue
            text.append(self.untokenize_single_sentence(reg['phrase']))
        boxes = boxes[mask_]
        return boxes, text

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

    def tokenize_annotations(self, max_size):

        counter = 0
        # print 'Tokenizing annotations...'
        for im in self.annotations:
            for obj in im['objects']:
                obj['class'] = self._object_class_to_ind[obj['class']]
            for rel in im['relationships']:
                rel['predicate'] = self._predicate_class_to_ind[rel['predicate']]
            for region in list(im['regions']):
                region['phrase'] = [self.word2idx[word] if word in self.word2idx else self.word2idx[self.unknown_token] \
                                        for word in (['<start>'] + region['phrase'] + ['<end>'])]
                if len(region['phrase']) < 5 or len(region['phrase']) >= max_size:
                    im['regions'].remove(region)


    def tokenize_sentence(self, sentence):
        return [self.word2idx[word] for word in (sentence.split() + ['<end>'])]

    def untokenize_single_sentence(self, sentence):
        word_sentence = []
        for idx in sentence:
            if idx == self.voc_sign['end']:
                break
            if idx == self.voc_sign['null'] or idx == self.voc_sign['start']:
                continue
            else:
                word_sentence.append(self.idx2word[idx])
        return ' '.join(word_sentence)

    def untokenize_sentence(self, sentence):
        result = []
        keep_id = []
        for i in range(sentence.shape[0]):
            word_sentence = []
            for idx in sentence[i]:
                if idx == self.voc_sign['end']:
                    break
                else:
                    word_sentence.append(self.idx2word[idx])
            if len(word_sentence) > 0:
                result.append(' '.join(word_sentence))
                keep_id.append(i)
        return result, np.array(keep_id, dtype=np.int)


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
