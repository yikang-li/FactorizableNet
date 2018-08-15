import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

import geometry_transform 


class Relation_Module(nn.Module):
      def __init__(self, dim_v, dim_o, dim_mm, geometry_trans='Geometry_Transform_v2'):
            super(Relation_Module, self).__init__()
            self.dim_key = dim_mm
            self.transform_key = nn.Linear(dim_v, dim_mm)
            self.transform_query = nn.Linear(dim_v, dim_mm)
            self.transform_visual = nn.Linear(dim_v, dim_o)
            self.transform_geometry = getattr(geometry_transform, geometry_trans)(dim_mm)


      def forward(self, feature_visual, rois):
            '''
            Relation Module adopts pre-non-linear-activated features
            '''
            feature_visual = nn.functional.relu(feature_visual)
            feature_key = self.transform_key(feature_visual)
            feature_query = self.transform_query(feature_visual)
            feature_visual = self.transform_visual(feature_visual)

            visual_weight = (feature_query.unsqueeze(0) * feature_key.unsqueeze(1)).sum(dim=2, keepdim=False) / np.sqrt(self.dim_key)
            geometry_weight = self.transform_geometry(rois)

            attention = visual_weight.exp() * geometry_weight
            for i in range(attention.size(0)):
                  attention[i, i] = 0
            attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-10)
            feature_out = torch.sum(attention.unsqueeze(2) * feature_visual.unsqueeze(0), dim=1, keepdim=False)

            return feature_out

if __name__ == '__main__':
      relation_module = Relation_Module_v2(4, 5, 3, 4)
      visual_features = Variable(torch.normal(torch.zeros(10, 4)))
      rois = Variable(torch.cat((torch.zeros(10, 1), (torch.rand(10, 4) + torch.FloatTensor([[0, 1, 2, 3], ])) * 100 ), dim=1))
      feature_out = relation_module(visual_features, rois)

      print(feature_out)








