import torch

from .utils import build_loss_bbox, build_loss_cls

import lib.network as network


def loss_FN_v1(pred_obj, pred_rel, roi_data_object, roi_data_predicate, 
				obj_loss_weight, rel_loss_weight):
	roi_data_object = [network.np_to_variable(roi_data_object[0], is_cuda=True, dtype=torch.LongTensor), 
					   network.np_to_variable(roi_data_object[1], is_cuda=True), 
					   network.np_to_variable(roi_data_object[2], is_cuda=True), 
					   network.np_to_variable(roi_data_object[3], is_cuda=True), ]
	roi_data_predicate = [network.np_to_variable(roi_data_predicate[0], is_cuda=True, dtype=torch.LongTensor)]
	# object cls loss
	loss_cls_obj, acc_obj = build_loss_cls(pred_obj[0], roi_data_object[0], obj_loss_weight)
	loss_reg_obj= build_loss_bbox(pred_obj[1], roi_data_object, acc_obj[2])
	loss_cls_rel,  acc_rel= build_loss_cls(pred_rel[0], roi_data_predicate[0], rel_loss_weight)

	return loss_cls_obj, loss_reg_obj, loss_cls_rel