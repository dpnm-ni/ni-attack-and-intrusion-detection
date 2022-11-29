import numpy as np
import argparse
import math
import pickle as pkl
import pandas as pd
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--rnn_len', type=int, default=16)
    parser.add_argument('--train_ratio', type=float)
    parser.add_argument('--valid_ratio', type=float)
    parser.add_argument('--test_ratio', type=float)
    args = parser.parse_args()
   
    print(args.data_dir)
    print(args.dataset)
    print(args.train_ratio)
    print(args.valid_ratio)
    print(args.test_ratio)
    print(args.train_ratio + args.valid_ratio + args.test_ratio)

    prob = args.train_ratio + args.valid_ratio + args.test_ratio
    if prob != 1.0:
        print("train, valid, and test ratios must equal 1.0")
        sys.exit(-1)

    #if 'int' in args.dataset:
        #data1 = pd.read_pickle(args.data_dir + 'int/int_train.pkl')
        #data2 = pd.read_pickle(args.data_dir + 'int/int_test.pkl')
      
        #print(type(data1))

        #tmp_df1 = pd.DataFrame(columns=['classification', 'path', 'proto', 'duration', 'hop_latency', 'egress_time', 'flow_latency', 'sink_time', 'port_tx_util', 'queue_occupancy'])
        #tmp_df2 = pd.DataFrame(columns=['classification', 'path', 'proto', 'duration', 'hop_latency', 'egress_time', 'flow_latency', 'sink_time', 'port_tx_util', 'queue_occupancy'])

        #for i in range(len(data1)):
        #    tmp_df1.loc[i] = data1[i]

        #for i in range(len(data2)):
        #    tmp_df2.loc[i] = data2[i]
        
        #tmp_df1 = tmp_df1[['path', 'proto', 'duration', 'hop_latency', 'egress_time', 'flow_latency', 'sink_time', 'port_tx_util', 'queue_occupancy', 'classification']]
        #tmp_df2 = tmp_df2[['path', 'proto', 'duration', 'hop_latency', 'egress_time', 'flow_latency', 'sink_time', 'port_tx_util', 'queue_occupancy', 'classification']]

        #print(tmp_df1)
        #print(tmp_df1.info())
        #print(len(tmp_df1))

        #print(tmp_df2)
        #print(tmp_df2.info())
        #print(len(tmp_df2))

        #print(len(data1))
        #print(len(data2))

        #tmp_df = pd.concat([tmp_df1, tmp_df2])
        #print(tmp_df.info())

        #tmp_df1.to_csv('/home/dpnm/cnsm2022/data/int/int_train.csv', index = False)
        #tmp_df2.to_csv('/home/dpnm/cnsm2022/data/int/int_test.csv', index = False)
        #tmp_df.to_csv('/home/dpnm/cnsm2022/data/int/int_total.csv', index = False)

        

    # load data for number of data
    if 'cicids2017' in args.dataset:
        #data1 = pd.read_csv(args.data_dir + 'cicids2017/binary/sub1_binary3.csv')
        #data2 = pd.read_csv(args.data_dir + 'cicids2017/binary/sub2_binary2.csv')
        #data3 = pd.read_csv(args.data_dir + 'cicids2017/binary/sub3_binary2.csv')
        #data4 = pd.read_csv(args.data_dir + 'cicids2017/binary/test3.csv')
        #data5 = pd.read_csv(args.data_dir + 'cicids2017/binary/test4.csv')
        #data6 = pd.read_csv(args.data_dir + 'cicids2017/binary/test5.csv')
        #data7 = pd.read_csv(args.data_dir + 'cicids2017/binary/test6.csv')
        #data8 = pd.read_csv(args.data_dir + 'cicids2017/binary/test7.csv')
        #data9 = pd.read_csv(args.data_dir + 'cicids2017/binary/test8.csv')
        #data10 = pd.read_csv(args.data_dir + 'cicids2017/binary/test9.csv')

        #print(len(data1))
        #print(len(data2))
        #print(len(data3))

        #tmp_data = pd.concat([data1, data2, data3])
        #print(tmp_data.info())
        #tmp_data.to_csv('/home/dpnm/cnsm2022/data/cicids2017/binary/sub_total2.csv', index = False)
        #data = pd.read_csv(args.data_dir + args.dataset + '.csv')
        data = pd.read_csv('/home/dpnm/cnsm2022/data/cicids2017/binary/sub_total2.csv')
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(axis=0)
        data = data.reset_index(drop=True)

        print(data['classification'].value_counts())
        print(data.isnull().sum().sum())
        print(data.info())
        #data.to_csv(args.data_dir + args.dataset + '.csv', index = False)
    else:
        #print(args.data_dir)
        #print(args.dataset)
        data = pd.read_csv(args.data_dir + args.dataset + '.csv')
        print(data['classification'].value_counts())
        print(data.isnull().sum().sum())
        print(data.info())
        #print(len(data))
    # obtain indices and ratios
    ids = list(range(len(data)))[args.rnn_len:]
    n_samples = len(ids)

    # split indices and make dicts
    tr_idx = math.ceil(args.train_ratio * n_samples)
    val_idx = math.ceil((args.train_ratio + args.valid_ratio) * n_samples) #125973

    # shuffle
    np.random.shuffle(ids)
    
    tr_ids = ids[:tr_idx]
    val_ids = ids[tr_idx:val_idx]
    test_ids = ids[val_idx:]

    out_dict = {'train':tr_ids, 'valid':val_ids, 'test':test_ids}
    target_path = args.data_dir + args.dataset + ".indices.rnn_len" + str(args.rnn_len) + ".pkl"

    with open(target_path, 'wb') as fp:
        pkl.dump(out_dict, fp)
