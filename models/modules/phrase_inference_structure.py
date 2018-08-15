import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from lib.utils.timer import Timer
import pdb
from lib.network import GroupDropout
from copy import deepcopy

class Abstract_Phrase_Inference_Structure(nn.Module):
	def __init__(self, opts):
		super(Abstract_Phrase_Inference_Structure, self).__init__()
		self.opts = opts

	def forward(self, feature_obj, feature_region, mat_predicate):

		raise NotImplementedError


class Basic_Phrase_Inference_Structure(Abstract_Phrase_Inference_Structure):
	def __init__(self, opts):
		super(Basic_Phrase_Inference_Structure, self).__init__(opts)
		self.opts = opts
		#self.w_object = Parameter()

		# To transform the attentioned features
		self.transform_subject = nn.Sequential(
											nn.ReLU(),
											#nn.BatchNorm1d(opts['dim_ho'], eps=0.001, momentum=0, affine=True),
											nn.Linear(opts['dim_ho'], opts['dim_mm'], bias=opts['use_bias']))
		self.transform_object = nn.Sequential(
											nn.ReLU(),
											#nn.BatchNorm1d(opts['dim_ho'], eps=0.001, momentum=0, affine=True),
											nn.Linear(opts['dim_ho'], opts['dim_mm'], bias=opts['use_bias']))
		self.transform_region = None

	def _fusion(self, transformed_feat_sub, transformed_feat_obj, transformed_feat_region):
		raise NotImplementedError

	def _prepare(self, feature_obj, feature_region, indices_sub, indices_obj, indices_region):
		raise NotImplementedError

	def forward(self, feature_obj, feature_region, mat_predicate):
		indices_sub = Variable(torch.from_numpy(mat_predicate[:, 0]).type(torch.LongTensor)).cuda().detach()
		indices_obj = Variable(torch.from_numpy(mat_predicate[:, 1]).type(torch.LongTensor)).cuda().detach()
		indices_region = Variable(torch.from_numpy(mat_predicate[:, 2]).type(torch.LongTensor)).cuda().detach()
		transformed_feat_sub, transformed_feat_obj, transformed_feat_region = self._prepare(
			feature_obj, feature_region, indices_sub, indices_obj, indices_region)
		# y = x_[p] + W_[s,p] * x_[s] + W_[o,p] * x_[o]
		out_feature_phrase = self._fusion(transformed_feat_sub, transformed_feat_obj, transformed_feat_region)
		return out_feature_phrase


class PI_v5(Basic_Phrase_Inference_Structure):
	'''
	sub/obj feature vector --> feature map --> merge with region
	--> Full connection for inference
	'''
	def __init__(self, opts):
		super(PI_v5, self).__init__(opts)
		self.transform_region = nn.Sequential(
								nn.ReLU(),
								#nn.BatchNorm2d(opts['dim_hr'], eps=0.001, momentum=0, affine=True),
								nn.Conv2d(opts['dim_hr'], opts['dim_mm'], kernel_size=1, bias=opts['use_bias']),
								GroupDropout(p=opts['dropout'], inplace=True),)
		if opts.get('bottleneck', False):
			print('Bottleneck enabled.')
			self.predicate_feat_pre = nn.Sequential(
								nn.ReLU(),
								nn.Conv2d(opts['dim_mm'], opts['dim_mm'] // 2, kernel_size=1, bias=opts['use_bias']),
								GroupDropout(p=opts['dropout'], inplace=True),
								nn.ReLU(),)
								#nn.BatchNorm2d(opts['dim_mm'], eps=0.001, momentum=0, affine=True),)
			self.predicate_feat_fc = nn.Sequential(
								nn.Linear((opts['dim_mm'] // 2)* opts['pool_size'] * opts['pool_size'] ,
									opts['dim_hp'], bias=opts['use_bias']),
								GroupDropout(p=opts['dropout'], inplace=True),)
		else:
			print('Bottleneck disabled.')
			self.predicate_feat_pre = nn.Sequential(
								nn.ReLU(),)
			self.predicate_feat_fc = nn.Sequential(
								nn.Linear(opts['dim_mm'] * opts['pool_size'] * opts['pool_size'] ,
									opts['dim_hp'], bias=opts['use_bias']),
								GroupDropout(p=opts['dropout'], inplace=True),)





	def _prepare(self, feature_obj, feature_region, indices_sub, indices_obj, indices_region):
		transformed_feat_sub = self.transform_subject(feature_obj)
		transformed_feat_sub = torch.index_select(transformed_feat_sub, 0, indices_sub)
		transformed_feat_obj = self.transform_object(feature_obj)
		transformed_feat_obj = torch.index_select(transformed_feat_obj, 0, indices_obj)
		transformed_feat_region = self.transform_region(feature_region)
		transformed_feat_region = torch.index_select(transformed_feat_region, 0, indices_region)
		return transformed_feat_sub, transformed_feat_obj, transformed_feat_region

	# @staticmethod
	# def _attention_merge(reference, query):
	# 	B, C, H, W = reference.size()
	# 	similarity = torch.sum(query * reference, dim=1, keepdim=True)
	# 	prob = F.sigmoid(similarity) # use sigmoid to retain scale of feature
	# 	weighted_feature = query * prob
	# 	return weighted_feature


	def _fusion(self, transformed_feat_sub, transformed_feat_obj, transformed_feat_region):
		batch_size = transformed_feat_sub.size(0)
		transformed_feat_sub = transformed_feat_sub.view(batch_size, -1, 1, 1)
		transformed_feat_obj = transformed_feat_obj.view(batch_size, -1, 1, 1)
		op = self.opts.get('op', 'Sum')
		if op == 'Sum':
			output_feature = transformed_feat_region + transformed_feat_sub + transformed_feat_obj
		elif op == 'Prod':
			output_feature = transformed_feat_region * transformed_feat_sub * transformed_feat_obj
		elif op == 'Sum_Prod':
			output_feature = transformed_feat_region * (transformed_feat_sub + transformed_feat_obj)
		output_feature = self.predicate_feat_pre(output_feature).view(batch_size, -1)
		output_feature = self.predicate_feat_fc(output_feature)
		return output_feature
