import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from lib.utils.timer import Timer
import pdb
from lib.network import GroupDropout


VISUALIZE_RESULTS = False
TIME_IT = False


class factor_updating_structure(nn.Module):
	def __init__(self, opts):
		super(factor_updating_structure, self).__init__()

		# To transform the attentioned features
		self.transform_object2region = nn.Sequential(
											nn.ReLU(),
											# nn.BatchNorm2d(opts['dim_ho'], eps=0.001, momentum=0, affine=True),
											nn.Conv2d(opts['dim_ho'], opts['dim_hr'], kernel_size=1,
												padding=0,bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		self.transform_region2object = nn.Sequential(
											nn.ReLU(),
											# nn.BatchNorm1d(opts['dim_hr'], eps=0.001, momentum=0, affine=True),
											nn.Linear(opts['dim_hr'], opts['dim_ho'], bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)

		self.att_region2object_obj = nn.Sequential(
											nn.ReLU(),
											# nn.BatchNorm1d(opts['dim_ho'], eps=0.001, momentum=0, affine=True),
											nn.Linear(opts['dim_ho'], opts['dim_mm'], bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		self.att_region2object_reg = nn.Sequential(
											nn.ReLU(),
											# nn.BatchNorm1d(opts['dim_hr'], eps=0.001, momentum=0, affine=True),
											nn.Conv2d(opts['dim_hr'], opts['dim_mm'], kernel_size=1,
												padding=0,bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		self.att_object2region_reg = nn.Sequential(
											nn.ReLU(),
											# nn.BatchNorm2d(opts['dim_hr'], eps=0.001, momentum=0, affine=True),
											nn.Conv2d(opts['dim_hr'], opts['dim_mm'], kernel_size=1,
												padding=0, bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		self.att_object2region_obj = nn.Sequential(
											nn.ReLU(),
											# nn.BatchNorm1d(opts['dim_ho'], eps=0.001, momentum=0, affine=True),
											nn.Linear(opts['dim_ho'], opts['dim_mm'], bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		self.timer_r2o = Timer()
		self.timer_o2r = Timer()
		self.opts = opts


	def forward(self, feature_obj, feature_region, mat_object, mat_region):

		self.timer_r2o.tic()
		feature_region2object = self.region_to_object(feature_obj, feature_region, mat_object)
		# Transform the features
		out_feature_object = feature_obj + self.transform_region2object(feature_region2object)
		self.timer_r2o.toc()


		self.timer_o2r.tic()
		# gather the attentioned features
		feature_object2region = self.object_to_region(feature_region, feature_obj, mat_region)
		# Transform the features
		out_feature_region = feature_region + self.transform_object2region(feature_object2region)
		self.timer_o2r.toc()

		if TIME_IT:
			print('[MPS Timing:]')
			print('\t[R2O]: {0:.3f} s'.format(self.timer_r2o.average_time))
			print('\t[O2R]: {0:.3f} s'.format(self.timer_o2r.average_time))

		return out_feature_object, out_feature_region

	@staticmethod
	def _attention_merge(reference, query, features):
		'''
		input:
			reference: vector [C] | [C x H x W]
			query: batched vectors [B x C] | [B x C x 1 x 1]
		output:
			merged message vector: [C] or [C x H x W]
		'''
		C = query.size(1)
		assert query.size(1) == reference.size(0)
		similarity = torch.sum(query * reference.unsqueeze(0), dim=1, keepdim=True) / np.sqrt(C + 1e-10) #  follow operations in [Attention is all you need]
		prob = F.softmax(similarity, dim=0)
		weighted_feature = torch.sum(features * prob, dim=0, keepdim=False)
		return weighted_feature

	# @staticmethod
	# def _attention_squeeze_feature(reference, query, features):
	# 	'''
	# 	input:
	# 		reference: vector [C]
	# 		query: batched vectors [B x C x W x H]
	# 	output:
	# 		merged message vector: [B X C]
	# 	'''
	# 	B, C, W, H = query.size()
	# 	assert query.size(1) == reference.size(0)
	# 	similarity = torch.sum(query.view(B, C, -1) * reference.view(1, C, 1), dim=1, keepdim=True) / np.sqrt(C + 1e-10)
	# 	prob = F.softmax(similarity, dim=2)
	# 	att_features = torch.sum(features.view(B, -1, W*H) * prob, dim=2, keepdim=False)
	# 	return att_features

	# def region_to_object(self, feat_obj, feat_region, select_mat):
	# 	feat_region_vec = torch.mean(feat_region.view(feat_region.size(0), feat_region.size(1), -1), dim=2, keepdim=False)
	# 	feat_obj_att = self.att_region2object_obj(feat_obj)
	# 	feat_reg_att_vec = self.att_region2object_reg_vec(feat_region_vec)
	# 	feat_reg_att_conv = self.att_region2object_reg_conv(feat_region)

	# 	feature_data = []
	# 	transfer_list = np.where(select_mat > 0)
	# 	for f_id in range(feat_obj.size(0)):
	# 		assert len(np.where(select_mat[f_id, :] > 0)[0]) > 0, "Something must be wrong. Please check the code."
	# 		source_indices = transfer_list[1][transfer_list[0] == f_id]
	# 		source_indices = Variable(torch.from_numpy(source_indices).type(torch.LongTensor), requires_grad=False).cuda()
	# 		feat_obj_att_target = feat_obj_att[f_id]
	# 		feat_reg_att_source = torch.index_select(feat_reg_att_conv, 0, source_indices)
	# 		feat_region_source = torch.index_select(feat_region, 0, source_indices)
	# 		att_features_reg = self._attention_squeeze_feature(feat_obj_att_target, feat_reg_att_source, feat_region_source)
	# 		feature_data.append(self._attention_merge(feat_obj_att_target,
	# 							torch.index_select(feat_reg_att_vec, 0, source_indices),
	# 							att_features_reg,))
	# 	return torch.stack(feature_data, 0)

	def region_to_object(self, feat_obj, feat_region, select_mat):
		feat_obj_att = self.att_region2object_obj(feat_obj)
		feat_reg_att = self.att_region2object_reg(feat_region).transpose(1, 3) # transpose the [channel] to the last
		feat_region_transposed = feat_region.transpose(1, 3)
		C_att = feat_reg_att.size(3)
		C_reg = feat_region_transposed.size(3)

		feature_data = []
		transfer_list = np.where(select_mat > 0)
		for f_id in range(feat_obj.size(0)):
			assert len(np.where(select_mat[f_id, :] > 0)[0]) > 0, "Something must be wrong. Please check the code."
			source_indices = transfer_list[1][transfer_list[0] == f_id]
			source_indices = Variable(torch.from_numpy(source_indices).type(torch.cuda.LongTensor), requires_grad=False)
			feat_region_source = torch.index_select(feat_region_transposed, 0, source_indices)
			feature_data.append(self._attention_merge(feat_obj_att[f_id],
								torch.index_select(feat_reg_att, 0, source_indices).view(-1, C_att),
								feat_region_source.view(-1, C_reg),))
		return torch.stack(feature_data, 0)

	def object_to_region(self, feat_region, feat_obj, select_mat):
		'''
		INPUT:
			feat_region: B x C x H x W
			feat_obj: B x C
		'''
		feat_reg_att = self.att_object2region_reg(feat_region)
		feat_obj_att = self.att_object2region_obj(feat_obj).view(feat_obj.size(0), -1, 1, 1)
		feat_obj = feat_obj.view(feat_obj.size(0), -1, 1, 1)
		feature_data = []
		transfer_list = np.where(select_mat > 0)
		for f_id in range(feat_region.size(0)):
			assert len(np.where(select_mat[f_id, :] > 0)[0]) > 0, "Something must be wrong!"
			source_indices = transfer_list[1][transfer_list[0] == f_id]
			source_indices = Variable(torch.from_numpy(source_indices).type(torch.cuda.LongTensor), requires_grad=False)
			feature_data.append(self._attention_merge(feat_reg_att[f_id],
								torch.index_select(feat_obj_att, 0, source_indices),
								torch.index_select(feat_obj, 0, source_indices)))
		return torch.stack(feature_data, 0)

	

	# ## clean version but consume a lot of memories
	# @staticmethod
	# def _attention(reference, query, features):
	# 	'''
	# 	input:
	# 		reference: vector [C] | [C x H x W]
	# 		query: batched vectors [B x C] | [B x C x 1 x 1]
	# 	output:
	# 		merged message vector: [C] or [C x H x W]
	# 	'''
	# 	C = query.size(2) # second dimension corresponds to the feature
	# 	similarity = torch.sum(query * reference, dim=2, keepdim=False) / np.sqrt(C + 1e-10) #  follow operations in [Attention is all you need]
	# 	prob = F.softmax(similarity, dim=1).unsqueeze(2)
	# 	weighted_feature = torch.sum(features * prob, dim=1, keepdim=False)

	# 	return weighted_feature



	# def region_to_object(self, feat_obj, feat_region, select_mat):
		
	# 	feat_region = torch.mean(feat_region.view(feat_region.size(0), feat_region.size(1), -1), dim=2, keepdim=False)
	# 	feat_obj_att = self.att_region2object_obj(feat_obj)
	# 	feat_reg_att = self.att_region2object_reg(feat_region)
	# 	select_mat = Variable(torch.Tensor(select_mat.astype(float)).type_as(feat_obj.data)).unsqueeze(2)
	# 	feature_data = self._attention(feat_obj_att.unsqueeze(1) * select_mat, 
	# 										 feat_reg_att.unsqueeze(0), 
	# 										 feat_region.unsqueeze(0))
	# 	return feature_data

	# def object_to_region(self, feat_region, feat_obj, select_mat):
	# 	'''
	# 	INPUT:
	# 		feat_region: B x C x H x W
	# 		feat_obj: B x C
	# 	'''
	# 	feat_reg_att = self.att_object2region_reg(feat_region)
	# 	feat_obj_att = self.att_object2region_obj(feat_obj).view(feat_obj.size(0), -1, 1, 1)
	# 	feat_obj = feat_obj.view(feat_obj.size(0), -1, 1, 1)
	# 	select_mat = Variable(torch.Tensor(select_mat.astype(float)).type_as(feat_obj.data)).unsqueeze(2).unsqueeze(2).unsqueeze(2)
	# 	feature_data = self._attention(feat_reg_att.unsqueeze(1) * select_mat, 
	# 										 feat_obj_att.unsqueeze(0), 
	# 										 feat_obj.unsqueeze(0))


	# 	return feature_data



