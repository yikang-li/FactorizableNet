import os
import os.path as osp
import torch
import numpy as np
import random
import numpy.random as npr
import json
import cPickle as pickle
import yaml
import cv2

from pprint import pprint

# from faster_rcnn.datasets.factory import get_imdb
import lib.datasets as datasets
from lib.visualize_graph.vis_utils import ground_predictions
from lib.visualize_graph.visualize import viz_scene_graph, draw_scene_graph


import argparse
import pdb

from PIL import Image 

from eval.evaluator import DenseCaptioningEvaluator


parser = argparse.ArgumentParser('Options for Meteor evaluation')

parser.add_argument('--path_data_opts', default='options/data_VRD.yaml', type=str,
                    help='path to a data file')
parser.add_argument('--path_result', default='output/testing_result.pkl', type=str,
                    help='path to the evaluation result file')
parser.add_argument('--output_dir', default='output/graph_results/VRD', type=str,
                    help='path to the evaluation result file')
parser.add_argument('--dataset_option', default='small', type=str,
                    help='path to the evaluation result file')
parser.add_argument('--dataset', default='VRD', type=str,
                    help='path to the evaluation result file')

args = parser.parse_args()

if args.dataset is not 'visual_genome':
    args.dataset_option = None

# def prepare_rel_matrix(relationships, object_num):
#     rel_mat = np.zeros()
#     for rel in len(relationships):
#         rel_mat[rel[0], rel[1]] = rel_cls[i]
#     return rel_mat


def visualize():

    global args
    print('=========== Visualizing Scene Graph =========')


    print('Loading dataset...'),
    with open(args.path_data_opts, 'r') as handle:
        options = yaml.load(handle)
    test_set = getattr(datasets, args.dataset)(options, 'test',
                             dataset_option=args.dataset_option,
                             use_region=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                shuffle=False, num_workers=4,
                                                pin_memory=True, 
                                                collate_fn=getattr(datasets, args.dataset).collate)
    print('Done Loading')

    with open(args.path_result, 'rb') as f:
        print('Loading result....'),
        result = pickle.load(f)
        print('Done')
        print('Total: {} images'.format(len(result)))

    for i, sample in enumerate(test_loader): # (im_data, im_info, gt_objects, gt_relationships)
        objects = result[i]['objects']
        relationships = result[i]['relationships']
        gt_boxes = sample['objects'][0][:, :4] / sample['image_info'][0][2]
        gt_relations = sample['relations'][0]
        gt_relations = zip(*np.where(gt_relations > 0))
        gt_to_pred = ground_predictions(objects['bbox'], gt_boxes, 0.5)
        assert sample['path'][0] == result[i]['path'], 'Image mismatch.'
        im = cv2.imread(osp.join(test_set._data_path, sample['path'][0]))
        image_name = sample['path'][0].split('/')[-1].split('.')[0]
        image_name = osp.join(args.output_dir, image_name)
        draw_graph_pred(im, objects['bbox'], objects['class'], relationships,
                             gt_to_pred, gt_relations, test_set._object_classes, 
                             test_set._predicate_classes, filename=image_name)

    print 'Done generating scene graphs.'


def draw_graph_pred(im, boxes, obj_ids, pred_relationships, gt_to_pred, 
            gt_relations, ind_to_class, ind_to_predicate, filename):
    """
    Draw a predicted scene graph. To keep the graph interpretable, only draw
    the node and edge predictions that have correspounding ground truth
    labels.
    args:
        im: image
        boxes: prediceted boxes
        obj_ids: object id list
        rel_pred_mat: relation classification matrix
        gt_to_pred: a mapping from ground truth box indices to predicted box indices
        idx: for saving
        gt_relations: gt_relationships
    """
    rel_pred = []
    all_rels = []

    for pred_rel in pred_relationships:
        for rel in gt_relations:
            if rel[0] not in gt_to_pred or rel[1] not in gt_to_pred:
                continue

            # discard duplicate grounding
            if pred_rel[0] == gt_to_pred[rel[0]] and pred_rel[1] == gt_to_pred[rel[1]]:
                rel_pred.append(pred_rel)
                all_rels.append([pred_rel[0], pred_rel[1]])
                break
    # rel_pred = pred_relationships[:5]  # uncomment to visualize top-5 relationships
    rel_pred = np.array(rel_pred)
    if rel_pred.size < 4:
        print('Image Skipped.')
        return
    # indices of predicted boxes
    pred_inds = rel_pred[:, :2].ravel()

    # draw graph predictions
    graph_dict = draw_scene_graph(obj_ids, pred_inds, rel_pred, ind_to_class, ind_to_predicate, filename=filename)
    viz_scene_graph(im, boxes, obj_ids, ind_to_class, ind_to_predicate, pred_inds, rel_pred, filename=filename)
    """
    out_boxes = []
    for box, cls in zip(boxes[pred_inds], cls_pred[pred_inds]):
        out_boxes.append(box[cls*4:(cls+1)*4].tolist())

    graph_dict['boxes'] = out_boxes

    if do_save == 'y':
        scipy.misc.imsave('cherry/im_%i.png' % idx, im)
        fn = open('cherry/graph_%i.json' % idx, 'w+')
        json.dump(graph_dict, fn)
    print(idx)
    """


if __name__ == '__main__':
    visualize()


