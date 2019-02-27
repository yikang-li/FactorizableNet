# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from lib.layer_utils.roi_layers import nms as nms_gpu
from lib.nms.nms_retain_all import nms_retain_all
import torch
# from ..nms import cpu_nms
# from ..nms import gpu_nms

def nms(dets, thresh, retain_all=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    
    if dets.shape[0] == 0:
        return []
    # ---numpy version---
    # original: return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    # ---pytorch version---
    if retain_all:
    	return nms_retain_all(dets, thresh)
    else:
        dets = torch.Tensor(dets).cuda()
    	return nms_gpu(dets[:, :4], dets[:, 4], thresh).cpu().numpy()