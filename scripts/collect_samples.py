import shutil
import os
import os.path as osp

import argparse
import pdb


parser = argparse.ArgumentParser('Options')

parser.add_argument('--path_files', default='output/graph_top_100/high_recall_cases.txt', type=str,
                    help='path to a data file')
parser.add_argument('--output_dir', default='output/graph_top_100/high_recall_cases', type=str)
parser.add_argument('--base_dir', default='output/graph_top_100')


args = parser.parse_args()

def main():
	global args

	if osp.isdir(args.output_dir):
		shutil.rmtree(args.output_dir)

	os.makedirs(args.output_dir)


	with open(args.path_files, 'r') as f:
		data = f.readlines()
	data = [v.strip('\n') for v in data]
	for f in data:
		try:
			shutil.copyfile(osp.join(args.base_dir, f+'.png'),
							osp.join(args.output_dir, f+'.png'))
			shutil.copyfile(osp.join(args.base_dir, f+'.pdf'),
							osp.join(args.output_dir, f+'.pdf'))
		except:
			continue
	print('Done.')



if __name__ == '__main__':
	main()