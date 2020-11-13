
#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: Support Vector Machine
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/11/13 13:05 PM
#
#


# ----------------------------------------
# Import packages and constant
# ----------------------------------------
import os
import h5py
import numpy as np
from glob import glob
from sklearn.svm import LinearSVC


class SVM(object):
    def __init__(self, feature_dir, args):
        self.feature_dir = feature_dir

        if(args.percentage<100):
            ResizeDataset(path = self.feature_dir, percentage=args.percentage, n_classes=16, shuffle=True)
            self.train_path = glob(os.path.join(self.feature_dir, 'train*_resized.h5'))
            self.test_path = glob(os.path.join(self.feature_dir, 'test*_resized.h5'))
        else:
            self.train_path = glob(os.path.join(self.feature_dir, 'train*.h5'))
            self.test_path = glob(os.path.join(self.feature_dir, 'test*.h5'))

        print("Loading feature dataset...")
        train_data = []
        train_label = []
        for path in self.train_path:
            f = h5py.File(path, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            train_data.append(data)
            train_label.append(label)
        self.train_data = np.concatenate(train_data, axis=0)
        self.train_label = np.concatenate(train_label, axis=0)
        print("Training set size:", np.size(self.train_data, 0))

        test_data = []
        test_label = []
        for path in self.test_path:
            f = h5py.File(path, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            test_data.append(data)
            test_label.append(label)
        self.test_data = np.concatenate(test_data, axis=0)
        self.test_label = np.concatenate(test_label, axis=0)
        print("Testing set size:", np.size(self.test_data, 0))


    def ResizeDataset(path, percentage, n_classes, shuffle):
        print('-Preparing resized evaluation dataset...')  
        path_h5py = os.path.join(path, '*.h5')
        path_h5py_all += glob(path_h5py)
        for h5_name in path_h5py_all:      
            out_file_name=h5_name + "_resized.h5"

            if os.path.exists(out_file_name):
                os.remove(out_file_name)
            fw = h5py.File(out_file_name, 'w', libver='latest')
            dset = fw.create_dataset("data", (1,1024,),maxshape=(None,1024), dtype='<f4')
            dset_l = fw.create_dataset("label",(1,),maxshape=(None,),dtype='uint8')
            fw.swmr_mode = True   
            f = h5py.File(h5_name)
            data = f['data'][:]
            cls_label = f['label'][:]
    
            #data shuffle
            if shuffle:        
                idx = np.arange(len(cls_label))
                np.random.shuffle(idx)
                data,cls_label = data[idx, ...], cls_label[idx]
    
            class_dist= np.zeros(self.n_classes)
            for c in range(len(data)):
            class_dist[cls_label[c]]+=1
            print('Ori data to size of :', np.sum(class_dist))
            print ('class distribution of this dataset :',class_dist)
        
            class_dist_new= (self.percentage*class_dist/100).astype(int)
            for i in range(self.n_classes):
                if class_dist_new[i]<1:
                    class_dist_new[i]=1
            class_dist_count=np.zeros(self.n_classes)

            data_count=0
            for c in range(len(data)):
                label_c=cls_label[c]
                if(class_dist_count[label_c] < class_dist_new[label_c]):
                    class_dist_count[label_c]+=1
                    new_shape = (data_count+1,1024,)
                    dset.resize(new_shape)
                    dset_l.resize((data_count+1,))
                    dset[data_count,:,:] = data[c]
                    dset_l[data_count] = cls_label[c]
                    dset.flush()
                    dset_l.flush()
                    data_count+=1
            print('Finished resizing data to size of :', np.sum(class_dist_new))
            print ('class distribution of resized dataset :',class_dist_new)
            fw.close


    def classify(self):
        clf = LinearSVC(random_state=0)
        clf.fit(self.train_data, self.train_label)
        result = clf.predict(self.test_data)
        accuracy = np.sum(result==self.test_label).astype(float) / np.size(self.test_label)
        print("Transfer linear SVM accuracy: {:.2f}%".format(accuracy*100))
