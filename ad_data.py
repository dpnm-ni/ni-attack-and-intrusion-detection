import pandas as pd
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset
import sys
from ad_utils import get_const

class AD_RNN_Dataset(Dataset):
    def __init__(self, mode, csv_path, ids_path, stat_path, data_name, rnn_len, test_dnn):
         # load csv, ids
        df_data = pd.read_csv(csv_path)
        np_data = np.array(df_data)

        #print(df_data)
        #print(len(df_data))
        #print(np_data)

        if "anomalydetection" in csv_path:
            self.data = np_data[:,:-3].astype(np.float32) #for numerical data (in sla detection)
            
            # assume sla label
            label_i = -3
            self.label = np_data[:,label_i].astype(np.int64)
        
            if mode == 'plot':
                self.ids = list(range((rnn_len-1), len(self.data)))
            else:
                with open(ids_path, 'rb') as fp:
                    ids = pkl.load(fp)
                if mode == 'train':
                    self.ids = ids['train']
                elif mode == 'valid':
                    self.ids = ids['valid']
                elif mode == 'test':
                    self.ids = ids['test']
                else:
                    print('mode must be either train, valid, or test')

            # normalize
            if mode == 'train':
                self.x_avg = np.mean(self.data, axis=0)
                self.x_std = np.std(self.data, axis=0)
    
                for i in range(len(self.x_std)):
                    if self.x_std[i] == 0:
                        self.x_std[i] = 0.001
    
                fp = open(stat_path, 'w')
                for i in range(self.x_avg.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_avg[i]))
                fp.write('\n')
                for i in range(self.x_std.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_std[i]))
                fp.write('\n')
                fp.close()
            else:
                fp = open(stat_path, 'r')
                lines = fp.readlines()
                self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
                self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
                fp.close()
        
            self.data -= np.expand_dims(self.x_avg, axis=0)
            self.data /= np.expand_dims(self.x_std, axis=0)

            n_nodes, n_features = get_const(data_name)
            self.n_nodes = n_nodes
            self.n_features = n_features
        
            # prepare lists
            label_col = "SLA_Label"
            datas = []
            headers = []

            for i in range(self.n_nodes):
                start, end = (i * self.n_features), ((i+1) * self.n_features)
                vnf_data = self.data[:, start:end]
                datas.append(np.copy(vnf_data))

            ## replace with add tvt to the dataset paths
            self.input = np.stack(datas).astype(np.float32)
            self.input = self.input.transpose(1,0,2) # (Bn x V x D)

            self.ids_i = 0
            self.n_samples = self.input.shape[0]
            self.rnn_len = rnn_len
            self.n_ids = len(self.ids)
            self.test_dnn = test_dnn
       
        if "int" in csv_path:
            self.data = np_data[:,:-1].astype(np.float32)

            # assume sla label
            label_i = -1
            self.label = np_data[:,label_i].astype(np.int64)

            if mode == 'plot':
                self.ids = list(range((rnn_len-1), len(self.data)))
            else:
                with open(ids_path, 'rb') as fp:
                    ids = pkl.load(fp)
                if mode == 'train':
                    self.ids = ids['train']
                elif mode == 'valid':
                    self.ids = ids['valid']
                elif mode == 'test':
                    self.ids = ids['test']
                else:
                    print('mode must be either train, valid, or test')

            # normalize
            if mode == 'train':
                self.x_avg = np.mean(self.data, axis=0)
                self.x_std = np.std(self.data, axis=0)

                for i in range(len(self.x_std)):
                    if self.x_std[i] == 0:
                        self.x_std[i] = 0.001

                fp = open(stat_path, 'w')
                for i in range(self.x_avg.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_avg[i]))
                fp.write('\n')
                for i in range(self.x_std.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_std[i]))
                fp.write('\n')
                fp.close()
            else:
                fp = open(stat_path, 'r')
                lines = fp.readlines()
                self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
                self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
                fp.close()

            self.data -= np.expand_dims(self.x_avg, axis=0)
            self.data /= np.expand_dims(self.x_std, axis=0)

            n_nodes, n_features = get_const(data_name)
            self.n_nodes = n_nodes
            self.n_features = n_features

            # prepare lists
            label_col = "classification"
            datas = []
            headers = []

            for i in range(self.n_nodes):
                start, end = (i * self.n_features), ((i+1) * self.n_features)
                vnf_data = self.data[:, start:end]
                datas.append(np.copy(vnf_data))

            ## replace with add tvt to the dataset paths
            self.input = np.stack(datas).astype(np.float32)
            self.input = self.input.transpose(1,0,2) # (Bn x V x D)

            self.ids_i = 0
            self.n_samples = self.input.shape[0]
            self.rnn_len = rnn_len
            self.n_ids = len(self.ids)
            self.test_dnn = test_dnn


        if "nsl-kdd" in csv_path:
            self.data = np_data[:,:-1].astype(np.float32)

            # assume sla label
            label_i = -1
            self.label = np_data[:,label_i].astype(np.int64)

            if mode == 'plot':
                self.ids = list(range((rnn_len-1), len(self.data)))
            else:
                with open(ids_path, 'rb') as fp:
                    ids = pkl.load(fp)
                if mode == 'train':
                    self.ids = ids['train']
                elif mode == 'valid':
                    self.ids = ids['valid']
                elif mode == 'test':
                    self.ids = ids['test']
                else:
                    print('mode must be either train, valid, or test')

            # normalize
            if mode == 'train':
                self.x_avg = np.mean(self.data, axis=0)
                self.x_std = np.std(self.data, axis=0)

                for i in range(len(self.x_std)):
                    if self.x_std[i] == 0:
                        self.x_std[i] = 0.001

                fp = open(stat_path, 'w')
                for i in range(self.x_avg.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_avg[i]))
                fp.write('\n')
                for i in range(self.x_std.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_std[i]))
                fp.write('\n')
                fp.close()
            else:
                fp = open(stat_path, 'r')
                lines = fp.readlines()
                self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
                self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
                fp.close()

            self.data -= np.expand_dims(self.x_avg, axis=0)
            self.data /= np.expand_dims(self.x_std, axis=0)

            n_nodes, n_features = get_const(data_name)
            self.n_nodes = n_nodes
            self.n_features = n_features

            # prepare lists
            label_col = "classification"
            datas = []
            headers = []

            for i in range(self.n_nodes):
                start, end = (i * self.n_features), ((i+1) * self.n_features)
                vnf_data = self.data[:, start:end]
                datas.append(np.copy(vnf_data))

            ## replace with add tvt to the dataset paths
            self.input = np.stack(datas).astype(np.float32)
            self.input = self.input.transpose(1,0,2) # (Bn x V x D)

            self.ids_i = 0
            self.n_samples = self.input.shape[0]
            self.rnn_len = rnn_len
            self.n_ids = len(self.ids)
            self.test_dnn = test_dnn

        if "unsw-nb15" in csv_path:
            self.data = np_data[:,:-1].astype(np.float32)

            # assume sla label
            label_i = -1
            self.label = np_data[:,label_i].astype(np.int64)

            if mode == 'plot':
                self.ids = list(range((rnn_len-1), len(self.data)))
            else:
                with open(ids_path, 'rb') as fp:
                    ids = pkl.load(fp)
                if mode == 'train':
                    self.ids = ids['train']
                elif mode == 'valid':
                    self.ids = ids['valid']
                elif mode == 'test':
                    self.ids = ids['test']
                else:
                    print('mode must be either train, valid, or test')

            # normalize
            if mode == 'train':
                self.x_avg = np.mean(self.data, axis=0)
                self.x_std = np.std(self.data, axis=0)

                for i in range(len(self.x_std)):
                    if self.x_std[i] == 0:
                        self.x_std[i] = 0.001

                fp = open(stat_path, 'w')
                for i in range(self.x_avg.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_avg[i]))
                fp.write('\n')
                for i in range(self.x_std.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_std[i]))
                fp.write('\n')
                fp.close()
            else:
                fp = open(stat_path, 'r')
                lines = fp.readlines()
                self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
                self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
                fp.close()

            self.data -= np.expand_dims(self.x_avg, axis=0)
            self.data /= np.expand_dims(self.x_std, axis=0)

            n_nodes, n_features = get_const(data_name)
            self.n_nodes = n_nodes
            self.n_features = n_features

            # prepare lists
            label_col = "classification"
            datas = []
            headers = []

            for i in range(self.n_nodes):
                start, end = (i * self.n_features), ((i+1) * self.n_features)
                vnf_data = self.data[:, start:end]
                datas.append(np.copy(vnf_data))

            ## replace with add tvt to the dataset paths
            self.input = np.stack(datas).astype(np.float32)
            self.input = self.input.transpose(1,0,2) # (Bn x V x D)

            self.ids_i = 0
            self.n_samples = self.input.shape[0]
            self.rnn_len = rnn_len
            self.n_ids = len(self.ids)
            self.test_dnn = test_dnn

        if "cicids2017" in csv_path:
            self.data = np_data[:,:-1].astype(np.float32) #for numerical data (in sla detection)

            # assume sla label
            label_i = -1
            self.label = np_data[:,label_i].astype(np.int64)

            if mode == 'plot':
                self.ids = list(range((rnn_len-1), len(self.data)))
            else:
                with open(ids_path, 'rb') as fp:
                    ids = pkl.load(fp)
                if mode == 'train':
                    self.ids = ids['train']
                elif mode == 'valid':
                    self.ids = ids['valid']
                elif mode == 'test':
                    self.ids = ids['test']
                else:
                    print('mode must be either train, valid, or test')

            # normalize
            if mode == 'train':
                self.x_avg = np.mean(self.data, axis=0)
                self.x_std = np.std(self.data, axis=0)

                for i in range(len(self.x_std)):
                    if self.x_std[i] == 0:
                        self.x_std[i] = 0.001

                fp = open(stat_path, 'w')
                for i in range(self.x_avg.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_avg[i]))
                fp.write('\n')
                for i in range(self.x_std.shape[0]):
                    if i > 0:
                        fp.write(', ')
                    fp.write('%.9f' % (self.x_std[i]))
                fp.write('\n')
                fp.close()
            else:
                fp = open(stat_path, 'r')
                lines = fp.readlines()
                self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
                self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
                fp.close()

            self.data -= np.expand_dims(self.x_avg, axis=0)
            self.data /= np.expand_dims(self.x_std, axis=0)

            n_nodes, n_features = get_const(data_name)
            self.n_nodes = n_nodes
            self.n_features = n_features

            # prepare lists
            label_col = "classification"
            datas = []
            headers = []

            for i in range(self.n_nodes):
                start, end = (i * self.n_features), ((i+1) * self.n_features)
                vnf_data = self.data[:, start:end]
                datas.append(np.copy(vnf_data))

            ## replace with add tvt to the dataset paths
            self.input = np.stack(datas).astype(np.float32)
            self.input = self.input.transpose(1,0,2) # (Bn x V x D)

            self.ids_i = 0
            self.n_samples = self.input.shape[0]
            self.rnn_len = rnn_len
            self.n_ids = len(self.ids)
            self.test_dnn = test_dnn


    def __len__(self):
        return self.n_ids

    def __getitem__(self, idx):
        idx = self.ids[idx]

        # get segment and label
        data_range = range((idx-self.rnn_len+1),(idx+1))
        x_data = self.input[data_range,:,:]
        y_data = self.label[data_range]

        if self.test_dnn == True:
            x_data = x_data[-1:,:,:] # (Bn x V x D)

        return x_data, y_data

class TPI_RNN_Dataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as fp:
            self.data = pkl.load(fp)

        # split input and label
        self.x_data = self.data[:, :, :-1].astype(np.float32)
        self.y_data = self.data[:, :, -1].astype(np.int64)
        
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tvt', type=str, default='sup_train')
    parser.add_argument('--csv_path', type=str, default='/home/dpnm/cnsm2022/data/nsl-kdd/KDDTotal_multi2.csv')
    parser.add_argument('--ids_path', type=str, default='/home/dpnm/cnsm2022/data/nsl-kdd/KDDTotal_multi2.indices.rnn_len16.pkl')
    parser.add_argument('--stat_path', type=str, default='/home/dpnm/cnsm2022/data/nsl-kdd/KDDTotal_multi2.csv.stat')
    parser.add_argument('--data_name', type=str, default='nsl-kdd/KDDTotal_multi2')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rnn_len', type=int, default=16)
    args = parser.parse_args()

    iter = AD_SUP2_RNN_ITERATOR2(tvt=args.tvt, csv_path=args.csv_path, ids_path=args.ids_path, stat_path=args.stat_path, data_name=args.data_name, batch_size=args.batch_size, rnn_len=args.rnn_len)

    for iloop, (anno, label, end_of_data) in enumerate(iter):
        anno, label = anno.to(device), label.to(device)
        # print('from iterator: ', anno.shape, label.shape)
        # take hidden, obtain output and loss, fix the model
