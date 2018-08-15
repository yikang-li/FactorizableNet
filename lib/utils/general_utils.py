import itertools
import collections
import torch
import numpy as np


def update_values(dict_from, dict_to):
	for key, value in dict_from.items():
		if isinstance(value, dict):
			update_values(dict_from[key], dict_to[key])
		elif value is not None:
			dict_to[key] = dict_from[key]

	return dict_to


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count