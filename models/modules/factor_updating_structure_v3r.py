import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from lib.utils.timer import Timer
import pdb
from lib.network import GroupDropout

from .factor_updating_structure_v3 import factor_updating_structure as FS_v3
from .relation_module import Relation_Module


VISUALIZE_RESULTS = False
TIME_IT = False


class factor_updating_structure(FS_v3):
  def __init__(self, opts):
    super(factor_updating_structure, self).__init__(opts)

    kernel_size = opts.get('kernel_size', 1)
    assert kernel_size % 2, 'Odd kernel size required.'
    padding = (kernel_size - 1) // 2
    # To transform the attentioned features
    self.transform_object2object = Relation_Module(opts['dim_ho'], opts['dim_ho'], opts['dim_ho'] // 2, 
                                      geometry_trans=self.opts.get('geometry', 'Geometry_Transform_v2'))



  def forward(self, feature_obj, feature_region, mat_object, mat_region, object_rois, region_rois):

    self.timer_r2o.tic()
    feature_region2object = self.region_to_object(feature_obj, feature_region, mat_object)
    # Transform the features
    out_feature_object = feature_obj + self.transform_region2object(feature_region2object) \
                          + self.transform_object2object(feature_obj, object_rois)
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




