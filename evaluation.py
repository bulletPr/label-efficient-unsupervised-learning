#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------------
#
#      Implements: Used code/feature in pretained model to infer features of input and the output features are used to SVM
#
# ----------------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/11/13 14:32 PM 
#
#


# ----------------------------------------
# Import packages and constant
# ----------------------------------------
from __future__ import print_function
import time
import os
import sys
import numpy as np
import shutil
import h5py
import torch
from torch.autograd import Variable

#from tensorboardX import SummaryWriter
from datasets import PartDataset
import modelnet40_loader
from pointnet import FoldingNet
from pointnet import FoldingNet_1024


class Evaluation(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode
        self.workers = args.workers
        self.dataset = args.dataset_name

        #create outpu directory and files
        #file = [f for f in args.model_path.split('/')]
        self.experiment_id = self.dataset
        cache_root = 'cache/%s' % self.experiment_id
        os.makedirs(cache_root, exist_ok=True)
        self.feature_dir = os.path.join(cache_root, 'features/')
        #sys.stdout = Logger(os.path.join(cache_root, 'inference_log.txt'))
        
        #check directory
        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir)
        else:
            shutil.rmtree(self.feature_dir)
            os.makedirs(self.feature_dir)
        
        #print args
        print(str(args))
        print('-Preparing evaluation dataset...')  
        
        if self.dataset == 'shapenet_part':
            dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints = 2048)
            self.infer_loader_train = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=self.workers)

            test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, train = False, npoints = 2500)
            self.infer_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=self.workers)
        elif self.dataset == 'modelnet40':
            dataset = modelnet40_loader.ModelNetH5Dataset(root = 'modelnet40_ply_hdf5_2048', train=True, npoints = 2048)
            self.infer_loader_train = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=self.workers)

            test_dataset = modelnet40_loader.ModelNetH5Dataset(root = 'modelnet40_ply_hdf5_2048', train = False, npoints = 2048)
            self.infer_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, 
                                          shuffle=True, num_workers=self.workers)


        print(len(dataset), len(test_dataset))
        #num_classes = len(dataset.classes)
        #print('classes', num_classes)

        self.model = FoldingNet_1024()

        self.model.cuda()

        if args.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if self.gpu_mode:
            self.model = self.model.cuda()


    def evaluate(self):
        self.model.eval()
        # generate train set for SVM
        #loss_buf = []
        feature_train = []
        lbs_train = []
        n = 0
        log_string("start preparing svm train dataset")
        for iter, (pts, lbs) in enumerate(self.infer_loader_train):
            pts, lbs = Variable(pts), Variable(lbs[:,0])
            pts = pts.transpose(2,1)
            if self.gpu_mode:
                pts = pts.cuda()
                lbs = lbs.cuda()
            output, _, feature  = self.model(pts) #output of reconstruction network
            #Sprint("shape of feature_train: " + str(lbs.shape))
            feature_train.append(feature.detach().cpu().numpy())  #output feature used to train a svm classifer
            if self.dataset == 'modelnet40':
                lbs_train.append(lbs.cpu().numpy().squeeze(1))
            else:
                lbs_train.append(lbs.cpu().numpy().squeeze)
            if ((iter+1)*self.batch_size % 2048) == 0 or (iter+1)==len(self.infer_loader_train):
                feature_train = np.concatenate(feature_train, axis=0)
                lbs_train = np.concatenate(lbs_train, axis=0)
                f = h5py.File(os.path.join(self.feature_dir, 'train' + str(n) + '.h5'), 'w')
                f['data']=feature_train
                f['label']=lbs_train
                f.close()
                log_string("size of generate traing set: " + str(feature_train.shape) + " ," + str(lbs_train.shape))
                log_string(f"Original train set {n} for SVM saved.")
                feature_train = []
                lbs_train = []
                n += 1
            #loss = self.model.get_loss(pts, output)
            #loss_buf.append(loss.detach().cpu().numpy())
        #print(f"Avg loss {np.mean(loss_buf)}.")
        log_string("finish generating train set for SVM.")

        # genrate test set for SVM
        #loss_buf = []
        feature_test = []
        lbs_test = []
        n = 0
        log_string("start preparing svm test dataset")
        for iter, (pts, lbs) in enumerate(self.infer_loader_test):
            #log_string("batch idx: " + str(iter) + " for generating test set for SVM...")
            pts, lbs = Variable(pts), Variable(lbs[:,0])
            pts = pts.transpose(2,1)
            if self.gpu_mode:
                pts = pts.cuda()
                lbs = lbs.cuda()
            output, _, feature = self.model(pts)
            feature_test.append(feature.detach().cpu().numpy())
            if self.dataset == 'modelnet40':
                lbs_test.append(lbs.cpu().numpy().squeeze(1))
            else:
                lbs_test.append(lbs.cpu().numpy().squeeze)
            if ((iter+1)*self.batch_size % 2048) == 0 or (iter+1)==len(self.infer_loader_test):
                feature_test = np.concatenate(feature_test, axis=0)
                lbs_test = np.concatenate(lbs_test, axis=0)
                f = h5py.File(os.path.join(self.feature_dir, 'test' + str(n) + '.h5'), 'w')
                f['data'] = feature_test
                f['label'] = lbs_test
                f.close()
                log_string("size of generate test set: " + str(feature_test.shape) + " ," + str(lbs_test.shape))
                log_string(f"Test set {n} for SVM saved.")
                feature_test = []
                lbs_test = []
                n += 1
            #loss = self.model.get_loss(pts, output)
            #loss_buf.append(loss.detach().cpu().numpy())
        #print(f"Avg loss {np.mean(loss_buf)}.")
        log_string("finish generating test set for SVM.")

        return self.feature_dir


    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Load model from {pretrain}.")    


LOG_FOUT = open('evaluation_log.txt', 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)