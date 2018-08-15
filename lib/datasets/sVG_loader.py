from .VRD_loader import VRD
import os.path as osp

class sVG(VRD):
    def __init__(self, opts, image_set='train', batch_size=1, dataset_option=None, use_region=False):
        image_set = image_set + '_' + dataset_option
        super(sVG, self).__init__(opts, image_set, batch_size, dataset_option, use_region)
        self._data_path = osp.join(self.opts['dir'], 'images')
