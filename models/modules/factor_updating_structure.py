import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from lib.utils.timer import Timer
import pdb


VISUALIZE_RESULTS = False

class Kernel_Attention_Module(nn.Module):
	def __init__(self, dim_source, dim_target, dim_mm):
		super(Kernel_Attention_Module, self).__init__()
		self.ws = nn.Linear(dim_source, dim_mm, bias=False)
		self.wt = nn.Linear(dim_target, dim_mm, bias=False)

	def forward(self, source_feat, target_feat, return_gate_value=False):
		# print '[unary_term, pair_term]', [unary_term, pair_term]
		gate = torch.sigmoid(torch.mean((self.ws(source_feat) * self.wt(target_feat)), 1, keepdim=True))
		# print 'gate', gate
		output = source_feat * gate.expand(gate.size(0), source_feat.size(1))
		if return_gate_value:
			return output, gate
		else:
			return output

class Attention_Module(nn.Module):
	def __init__(self, dim_source, dim_target, filter_size = 128):
		super(Attention_Module, self).__init__()
		self.filter_size = filter_size
		if filter_size > 0:
			self.w = nn.Linear(dim_source+dim_target, filter_size, bias=True)

	def forward(self, source_feat, target_feat, return_gate_value=False):

		if self.filter_size > 0:
			gate = torch.cat([source_feat, target_feat], 1)
			gate = F.relu(gate)
			gate = torch.mean(torch.sigmoid(self.w(gate)), 1, keepdim=True)
			# print 'gate', gate
			output = source_feat * gate.expand_as(source_feat)
			if return_gate_value:
				return output, gate
			else:
				return output
		else:
			return source_feat


class factor_updating_structure(nn.Module):
	def __init__(self, opts):
		super(factor_updating_structure, self).__init__()

		# Attention modules
		if opts['use_attention']:
			if opts['use_kernel']:
				self.gate_object2region = Kernel_Attention_Module(opts['dim_ho'], opts['dim_hr'], opts['dim_mm'])
				self.gate_region2object = Kernel_Attention_Module(opts['dim_hr'], opts['dim_ho'], opts['dim_mm'])
			else:
				self.gate_object2region = Attention_Module(opts['dim_ho'], opts['dim_hr'], opts['gate_width'])
				self.gate_region2object = Attention_Module(opts['dim_hr'], opts['dim_ho'], opts['gate_width'])
		else:
			self.gate_object2region = None
			self.gate_region2object = None
		# To transform the attentioned features
		self.transform_object2region = nn.Sequential(
											nn.ReLU(),
											nn.Linear(opts['dim_ho'], opts['dim_hr'], bias=opts['use_bias']))
		self.transform_region2object = nn.Sequential(
											nn.ReLU(),
											nn.Linear(opts['dim_hr'], opts['dim_ho'], bias=opts['use_bias']))

		self.use_average = opts['mps_use_average']



	def forward(self, feature_obj, feature_region, mat_object, mat_region):

		feature_region2object = self.prepare_message(feature_obj, feature_region, mat_object, self.gate_region2object)
		# Transform the features
		out_feature_object = feature_obj + self.transform_region2object(feature_region2object)
		# gather the attentioned features
		feature_object2region = self.prepare_message(feature_region, feature_obj, mat_region, self.gate_object2region)
		# Transform the features
		out_feature_region = feature_region + self.transform_object2region(feature_object2region)

		return out_feature_object, out_feature_region

	def prepare_message(self, target_features, source_features, select_mat, attend_module=None):
		feature_data = []
		transfer_list = np.where(select_mat > 0)

		for f_id in range(target_features.size(0)):
			if len(np.where(select_mat[f_id, :] > 0)[0]) > 0:
				source_indices = transfer_list[1][transfer_list[0] == f_id]
				source_indices = Variable(torch.from_numpy(source_indices).type(torch.LongTensor)).cuda()
				features = torch.index_select(source_features, 0, source_indices)
				if attend_module is not None:
					target_f = target_features[f_id].view(1, -1).expand(features.size(0), -1)
					features = attend_module(features, target_f)
				if self.use_average:
					features = features.mean(0)
				else:
					features = features.sum(0)
				feature_data.append(features)
			else:
				temp = Variable(torch.zeros(target_features.size()[1:]), requires_grad=False).type(torch.FloatTensor).cuda()
				feature_data.append(temp)
		return torch.stack(feature_data, 0)



