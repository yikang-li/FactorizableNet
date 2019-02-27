import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import pdb
import relation_module
#from options.config_FN import cfg

class Dumplicate_Removal(nn.Module):
  def __init__(self, opts):
    super(Dumplicate_Removal, self).__init__()
    self.opts = opts
    self.relation_transform = relation_module.Relation_Module(
        self.opts['dim_mm'],
        self.opts['dim_mm'],
        self.opts['dim_mm'] // 2,
        geometry_trans=self.opts.get('geometry', 'Geometry_Transform_v2')
      )
    self.transform_visual = nn.Linear(self.opts['dim_ho'], self.opts['dim_mm'])
    self.rank_embeddings = nn.Embedding(256, self.opts['dim_mm']) # cfg.TRAIN.BATCH_SIZE, self.opts['dim_mm'])
    self.transform_rescore = nn.Linear(self.opts['dim_mm'], 1)


  def forward(self, feature_obj, highest_prob, rois_obj):
    '''
    Training stage: object probability is that of the assigned ground truth label
    Testing stage: object probability is the one with highest probability
    '''
    assert highest_prob.size(0) <= self.rank_embeddings.num_embeddings
    if isinstance(highest_prob, Variable):
        highest_prob = highest_prob.data
    _, rank = torch.sort(highest_prob, descending=True, dim=0)
    rank = Variable(rank)
    feature_rank = self.rank_embeddings(rank)
    feature_obj = self.transform_visual(feature_obj)
    feature_visual = feature_rank + feature_obj
    feature_visual = self.relation_transform(feature_visual, rois_obj)
    reranked_score = self.transform_rescore(F.relu(feature_visual, inplace=True)) 
    reranked_score = torch.sigmoid(reranked_score)

    return reranked_score





if __name__ == '__main__':
    opts = {
      'dim_mm': 6,
      'dim_ho': 4,
    }
    nms_module = Dumplicate_Removal(opts)
    visual_features = Variable(torch.normal(torch.zeros(10, 4)))
    rois = Variable(torch.cat((torch.zeros(10, 1), (torch.rand(10, 4) + torch.FloatTensor([[0, 1, 2, 3], ])) * 100 ), dim=1))
    duplicate_labels = Variable(torch.ones(5, 1)).type(torch.LongTensor)
    cls_prob_object = Variable(torch.rand(10, 20))

    mask = torch.zeros_like(cls_prob_object[:duplicate_labels.size(0)]).type(torch.ByteTensor)
    for i in range(duplicate_labels.size(0)):
        mask[i, duplicate_labels.data[i][0]] = 1
    selected_prob = torch.masked_select(cls_prob_object[:duplicate_labels.size(0)], mask)
    reranked_score = nms_module(visual_features[:duplicate_labels.size(0)], selected_prob, rois[:duplicate_labels.size(0)])
    selected_prob = selected_prob.unsqueeze(1) * reranked_score

    loss = F.binary_cross_entropy(selected_prob, duplicate_labels.float())
    loss.backward()
    print(nms_module.transform_rescore.weight.grad)
