#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/11/13 17:17 PM 
#
#
from __future__ import print_function

import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))

from evaluation import Evaluation
from svm import SVM

def get_parser():
    parser = argparse.ArgumentParser(description='Label-efficient Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of batch')
    parser.add_argument('--dataset_name', type=str, default='modelnet40', metavar='dataset_name',
                        help='classifer dataset')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=16)
    parser.add_argument('--gpu_mode', action='store_true', help='Enables CUDA training')
    parser.add_argument('--model_path', type=str, default='./experiment/foldingnet_model_249.pth', metavar='N',
                        help='Path to load model')
    parser.add_argument('--percentage', type=int, default=100, metavar='percentage', 
                        help='percentage of data used for svm training')
    parser.add_argument('--feature_dir', type=str, default='', metavar='N',
                        help='Path to load svm data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    if args.feature_dir == '':
        inference = Evaluation(args)
        feature_dir = inference.evaluate()
        print(feature_dir)
    else:
        feature_dir = args.feature_dir
    svm = SVM(feature_dir,args.percentage)
    svm.classify()