import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


def geometry_transform(rois_keys, rois_queries=None):
      if rois_queries is None:
            rois_queries = rois_keys
      if isinstance(rois_keys, Variable): # transform to Tensor
            rois_keys = rois_keys.data
            rois_queries = rois_queries.data
      if rois_keys.size(1) == 5: # Remove the ID
            rois_keys = rois_keys[:, 1:]
            rois_queries = rois_queries[:, 1:]

      assert rois_keys.size(1) == 4
      # keys
      w_keys = (rois_keys[:, 2] - rois_keys[:, 0] + 1e-10).unsqueeze(1)
      h_keys = (rois_keys[:, 3] - rois_keys[:, 1] + 1e-10).unsqueeze(1)
      x_keys = ((rois_keys[:, 2] + rois_keys[:, 0]) / 2).unsqueeze(1)
      y_keys = ((rois_keys[:, 3] + rois_keys[:, 1]) / 2).unsqueeze(1)
      # queries
      w_queries = (rois_queries[:, 2] - rois_queries[:, 0] + 1e-10).unsqueeze(0)
      h_queries = (rois_queries[:, 3] - rois_queries[:, 1] + 1e-10).unsqueeze(0)
      x_queries = ((rois_queries[:, 2] + rois_queries[:, 0]) / 2).unsqueeze(0)
      y_queries = ((rois_queries[:, 3] + rois_queries[:, 1]) / 2).unsqueeze(0)

     # slightly different from [Relation Networks for Object Detection]
      geometry_feature = torch.stack(
          [ (x_keys - x_queries).abs() / w_keys,
           (y_keys - y_queries).abs() / h_keys,
           w_keys / w_queries,
           h_keys / h_queries,], dim=2)

      geometry_log = geometry_feature.log()
      geometry_log[geometry_feature == 0] = 0

      return geometry_log

def positional_encoding(position_mat, dim_output, wave_length=1000):
      '''Sinusoidal Positional_Encoding.
      Returns:
         Sinusoidal Positional embedding of different objects
      '''
      # position_mat: [num_keys, num_queries, 4]
      assert dim_output % 8 == 0, "[dim_output] is expected to be an integral multiple of 8"
      position_enc = torch.Tensor([np.power(wave_length, 8.*i/dim_output) for i in range(dim_output / 8)]).view(1, 1, 1, -1).type_as(position_mat)
      # position_enc: [num_keys, num_queries, 4, dim_output / 8]
      position_enc = position_mat.unsqueeze(-1) * 100 / position_enc
      # Second part, apply the cosine to even columns and sin to odds.
      # position_enc: [num_keys, num_queries, 4, dim_output / 4]
      position_enc = torch.cat([torch.sin(position_enc), torch.cos(position_enc)], dim=3)
      position_enc = position_enc.view(position_enc.size(0), position_enc.size(1), -1)

      return position_enc 

class Geometry_Transform_v1(nn.Module):
      def __init__(self, dim_mm):
            super(Geometry_Transform_v1, self).__init__()
            self.transform_geometry = nn.Sequential(
                nn.Linear(4, dim_mm),
                nn.ReLU(),
                nn.Linear(dim_mm, 1),
                nn.ReLU(),)

      def forward(self, rois_keys, rois_queries=None):
            position_mat = Variable(geometry_transform(rois_keys, rois_queries), requires_grad=True)
            geometry_weight = self.transform_geometry(position_mat).squeeze(2)
            return geometry_weight


class Geometry_Transform_v2(nn.Module):
      '''
      expand the geometry features
      '''
      def __init__(self, dim_mm):
            super(Geometry_Transform_v2, self).__init__()
            self.transform_geometry = nn.Sequential(
                nn.Linear(dim_mm, 1),
                nn.ReLU(),)
            self.dim_mm = dim_mm

      def forward(self, rois_keys, rois_queries=None):
            position_mat = geometry_transform(rois_keys, rois_queries)
            geometry_weight = positional_encoding(position_mat, self.dim_mm)
            geometry_weight = Variable(geometry_weight, requires_grad=True)  
            geometry_weight = self.transform_geometry(geometry_weight).squeeze(2)
            return geometry_weight