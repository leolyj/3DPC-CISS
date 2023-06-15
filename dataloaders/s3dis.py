"""Class Split (Base and Incre) for S3DIS Dataset"""
import os
import glob
import numpy as np
import pickle

class S3DISDataset(object):
    def __init__(self, cvfold, tasks, data_path):
        self.data_path = data_path
        self.classes = 13
        # self.class2type = {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window', 6:'door', 7:'table',
        #                    8:'chair', 9:'sofa', 10:'bookcase', 11:'board', 12:'clutter'}
        class_names = open(os.path.join(os.path.dirname(data_path[:-14]), 'meta', 's3dis_classnames.txt')).readlines()
        self.class2type = {i: name.strip() for i, name in enumerate(class_names)}
        print(self.class2type)
        self.type2class = {self.class2type[t]: t for t in self.class2type}
        self.types = self.type2class.keys()

        if cvfold == 0:
            if tasks == '12-1':
                self.base = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board']
                self.incre = ['clutter']
            elif tasks == '10-3':
                self.base = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa']
                self.incre = ['bookcase', 'board', 'clutter']
            elif tasks == '8-5':
                self.base = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table']
                self.incre = ['chair', 'sofa', 'bookcase', 'board', 'clutter']
            elif tasks == '8-1':
                self.base = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table']
                self.incre = ['chair', 'sofa', 'bookcase', 'board', 'clutter']
            elif tasks == "Joint": # For Joint training, combine the base+novel classes, random
                self.base = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
                self.incre = []
            else:
                NotImplementedError('Unknown tasks mode!')
        elif cvfold == 1:
            if tasks == '12-1':
                self.base = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall']
                self.incre = ['window']
            elif tasks == '10-3':
                self.base = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa']
                self.incre = ['table', 'wall', 'window']
            elif tasks == '8-5':
                self.base = ['beam', 'board', 'bookcase', 'ceiling',  'chair', 'clutter', 'column', 'door']
                self.incre = ['floor', 'sofa', 'table', 'wall', 'window']
            elif tasks == '8-1':
                self.base = ['beam', 'board', 'bookcase', 'ceiling',  'chair', 'clutter', 'column', 'door']
                self.incre = ['floor', 'sofa', 'table', 'wall', 'window']
            elif tasks == "Joint":
                self.base = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor',
                             'sofa', 'table', 'wall', 'window']
                self.incre = []
            else:
                NotImplementedError('Unknown tasks mode!')
        else:
            raise NotImplementedError('Unknown cvfold (%s). [Options: 0,1]' %cvfold)

        self.base_classes = [self.type2class[i] for i in self.base]
        if len(self.incre) != 0:
            self.incre_classes = [self.type2class[i] for i in self.incre]
            print('base_class:{0}'.format(self.base_classes))
            print('incre_class:{0}'.format(self.incre_classes))
        else: # For Joint Training
            self.incre_classes = []
            print('all_class:{0}'.format(self.base_classes))

        self.class2scans = self.get_class2scans()

    def get_class2scans(self):
        class2scans_file = os.path.join(self.data_path, 'class2scans.pkl')
        if os.path.exists(class2scans_file):
            # load class2scans (dictionary)
            with open(class2scans_file, 'rb') as f:
                class2scans = pickle.load(f)
        else:
            min_ratio = .05  # to filter out scans with only rare labelled points
            min_pts = 100  # to filter out scans with only rare labelled points
            class2scans = {k: [] for k in range(self.classes)}

            for file in glob.glob(os.path.join(self.data_path, 'data', '*.npy')):
                scan_name = os.path.basename(file)[:-4]
                data = np.load(file)
                labels = data[:, 6].astype(np.int)
                classes = np.unique(labels)
                print('{0} | shape: {1} | classes: {2}'.format(scan_name, data.shape, list(classes)))
                for class_id in classes:
                    # if the number of points for the target class is too few, do not add this sample into the dictionary
                    num_points = np.count_nonzero(labels == class_id)
                    threshold = max(int(data.shape[0] * min_ratio), min_pts)
                    if num_points > threshold:
                        class2scans[class_id].append(scan_name)

            print('==== class to scans mapping is done ====')
            for class_id in range(self.classes):
                print(
                    '\t class_id: {0} | min_ratio: {1} | min_pts: {2} | class_name: {3} | num of scans: {4}'.format(
                        class_id, min_ratio, min_pts, self.class2type[class_id], len(class2scans[class_id])))

            with open(class2scans_file, 'wb') as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        return class2scans

if __name__ == '__main__':
    dataset = S3DISDataset(0, '12-1', '../datasets/S3DIS/blocks_bs1_s1')
    dataset1 = S3DISDataset(0, '10-3', '../datasets/S3DIS/blocks_bs1_s1')
    dataset2 = S3DISDataset(0, '8-5', '../datasets/S3DIS/blocks_bs1_s1')
    dataset3 = S3DISDataset(0, '8-1', '../datasets/S3DIS/blocks_bs1_s1')
    dataset4 = S3DISDataset(0, 'Joint', '../datasets/S3DIS/blocks_bs1_s1')
    # --------------------------------------------------------------------
    dataset5 = S3DISDataset(1, '12-1', '../datasets/S3DIS/blocks_bs1_s1')
    dataset6 = S3DISDataset(1, '10-3', '../datasets/S3DIS/blocks_bs1_s1')
    dataset7 = S3DISDataset(1, '8-5', '../datasets/S3DIS/blocks_bs1_s1')
    dataset8 = S3DISDataset(1, '8-1', '../datasets/S3DIS/blocks_bs1_s1')
    dataset9 = S3DISDataset(1, 'Joint', '../datasets/S3DIS/blocks_bs1_s1')
