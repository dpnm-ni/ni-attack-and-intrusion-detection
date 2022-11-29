'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pickle as pkl

import math
import sys
import time
import random
import os
import argparse

from ad_data import AD_RNN_Dataset
from ad_utils import call_model
from ad_test import eval_forward, eval_binary, log_neptune, get_joint_valid_loss

from sklearn.metrics import classification_report

def initial_forward(model, x_data, y_data, device, criterion):
    x_data = x_data.to(dtype=torch.float32, device=device)
    y_data = y_data[:,-1].to(dtype=torch.int64, device=device)

    out, _ = model(x_data)
    
    loss = criterion(out, y_data)

    return loss

def modified_forward(model, x_data, y_data, device, criterion, ratio):
    loss = 0
    Bn, Tx, V, D = x_data.size()
    x_data = x_data.to(dtype=torch.float32, device=device) # Bn Tx V D
    y_data = y_data.to(dtype=torch.int64, device=device) # Bn Tx 1
    y_prev = torch.zeros(Bn,1).to(dtype=torch.int64, device=device) # Bn 1
    clf_hidden = model.init_clf_hidden(Bn, device)

    use_teacher_forcing = True if random.random() < ratio else False

    for di in range(Tx):
        # teacher forcing
        if use_teacher_forcing: # Bn, V, D
            x_t = x_data[:,di,:,:]
            y_prev = y_prev.unsqueeze(1).expand(-1,V,-1).contiguous()

            input_data = torch.cat((x_t, y_prev), dim=-1).unsqueeze(1)
            
            output, clf_hidden = model(input_data, clf_hidden)

            loss += criterion(output, y_data[:, di])

            y_prev = y_data[:, di].unsqueeze(-1)
        # free running
        else:
            x_t = x_data[:,di,:,:]
            y_prev = y_prev.unsqueeze(1).expand(-1,V,-1).contiguous()

            input_data = torch.cat((x_t, y_prev), dim=-1).unsqueeze(1)
            
            output, clf_hidden = model(input_data, clf_hidden)

            loss += criterion(output, y_data[:, di])

            topv, topi = output.topk(1)
            y_prev = topi.detach()  # detach from history as input

    return loss

def get_joint_valid_loss(model, dataiter1, dataiter2, criterion, use_prev_pred, device):
    model.eval()
    valid_loss = 0.0
    n_batches = 0
    ratio = 0.0
    
    for li, ((x_data1, y_data1), (x_data2, y_data2)) in enumerate(zip(dataiter1, dataiter2)):
        Tx, Bn, V, D = x_data1.size()

        if use_prev_pred == 1:
            loss1 = modified_forward(model, x_data1, y_data1, device, criterion, ratio)
            loss2 = modified_forward(model, x_data2, y_data2, device, criterion, ratio)

            valid_loss += float(loss1.detach()) + float(loss2.detach())
            valid_loss /= Tx

        else:
            loss1 = initial_forward(model, x_data1, y_data1, device, criterion)
            loss2 = initial_forward(model, x_data2, y_data2, device, criterion)

            valid_loss += loss1.item() + loss2.item()

    valid_loss /= (li + 1)
    model.train()

    return valid_loss

def train_main(args, neptune):
    device = torch.device('cuda')

    model = call_model(args, device)

    if args.classifier == 'rnn':
        test_dnn = False
    else:
        test_dnn = True

    train1 = AD_RNN_Dataset(mode="train",
                            csv_path=args.csv_path1,
                            ids_path=args.ids_path1,
                            stat_path=args.stat_path1,
                            data_name=args.dataset1,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)
    
    valid1 = AD_RNN_Dataset(mode="valid",
                            csv_path=args.csv_path1,
                            ids_path=args.ids_path1,
                            stat_path=args.stat_path1,
                            data_name=args.dataset1,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)
    
    test1 = AD_RNN_Dataset(mode="test",
                            csv_path=args.csv_path1,
                            ids_path=args.ids_path1,
                            stat_path=args.stat_path1,
                            data_name=args.dataset1,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)

    trainiter1 = torch.utils.data.DataLoader(train1, batch_size=args.batch_size, shuffle=True)
    validiter1 = torch.utils.data.DataLoader(valid1, batch_size=args.batch_size, shuffle=True)
    testiter1 = torch.utils.data.DataLoader(test1, batch_size=args.batch_size, shuffle=True)

    train2 = AD_RNN_Dataset(mode="train",
                            csv_path=args.csv_path2,
                            ids_path=args.ids_path2,
                            stat_path=args.stat_path2,
                            data_name=args.dataset2,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)
    
    valid2 = AD_RNN_Dataset(mode="valid",
                            csv_path=args.csv_path2,
                            ids_path=args.ids_path2,
                            stat_path=args.stat_path2,
                            data_name=args.dataset2,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)
    
    test2 = AD_RNN_Dataset(mode="test",
                            csv_path=args.csv_path2,
                            ids_path=args.ids_path2,
                            stat_path=args.stat_path2,
                            data_name=args.dataset2,
                            rnn_len=args.rnn_len,
                            test_dnn=test_dnn)
    
    trainiter2 = torch.utils.data.DataLoader(train2, batch_size=args.batch_size, shuffle=True)
    validiter2 = torch.utils.data.DataLoader(valid2, batch_size=args.batch_size, shuffle=True)
    testiter2 = torch.utils.data.DataLoader(test2, batch_size=args.batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()

    print('trainiter1: {} samples'.format(len(train1)))
    print('trainiter2: {} samples'.format(len(train2)))
    print('validiter1: {} samples'.format(len(valid1)))
    print('validiter2: {} samples'.format(len(valid2)))
    print('testiter1: {} samples'.format(len(test1)))
    print('testiter2: {} samples'.format(len(test2)))

    # declare optimizer
    estring = "optim." + args.optimizer
    optimizer = eval(estring)(model.parameters(), lr=args.lr)
    if args.use_scheduler == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # modify the dataset to produce labels
    # create a training loop
    train_loss = 0.0
    bc = 0 # bad counter
    sc = 0 # step counter
    best_valid_loss = None

    for ei in range(args.max_epoch):
        for li, ((x_data1, y_data1), (x_data2, y_data2)) in enumerate(zip(trainiter1, trainiter2)):
            if args.use_prev_pred == 1:
                loss1 = modified_forward(model, x_data1, y_data1, device, criterion, args.teacher_forcing_ratio)
                loss2 = modified_forward(model, x_data2, y_data2, device, criterion, args.teacher_forcing_ratio)
            else:
                loss1 = initial_forward(model, x_data1, y_data1, device, criterion)
                loss2 = initial_forward(model, x_data2, y_data2, device, criterion)

            optimizer.zero_grad()

            # go through loss function
            combined_loss = loss1 + loss2
            combined_loss.backward()

            # optimizer
            optimizer.step()
            train_loss += combined_loss.item()

        train_loss = train_loss / (li + 1)
        print('epoch: {:d} | train_loss: {:.4f}'.format(ei+1, train_loss))
        if neptune is not None: neptune.log_metric('train_loss', ei+1, train_loss)
        train_loss = 0.0

        valid_loss = get_joint_valid_loss(model, validiter1, validiter2, criterion, args.use_prev_pred, device)
        print('epoch: {:d} | valid_loss: {:.4f}'.format(ei+1, valid_loss))
        if neptune is not None: neptune.log_metric('valid_loss', ei+1, valid_loss)

        if ei == 0 or valid_loss < best_valid_loss:
            save_path = args.save_dir + args.out_file
            torch.save(model, save_path)
            bc = 0
            best_valid_loss = valid_loss
            print('found new best model')
        else:
            bc += 1
            if bc > args.patience:
                if args.use_scheduler == 1:
                    print("learning rate decay..")
                    scheduler.step()
                    bc = 0
                    sc += 1

                    if(sc >= args.n_decay):
                        break
                else:
                    print("early stopping..")
                    break

            print('bad counter == %d' % (bc))

    model = torch.load(save_path)
    
    # evaluation
    eval_modes = ['test']
    iter_nums = ['1', '2']
    
    for iter_num in iter_nums:
        for eval_mode in eval_modes:
            dataiter_str = eval_mode + 'iter' + iter_num
            dataiter_name = eval('args.dataset' + iter_num)
            dataiter = eval(dataiter_str)

            targets, preds = eval_forward(model, dataiter, args.use_prev_pred, device)

            # metric
            acc, prec, rec, f1 = eval_binary(targets, preds)

            # std and neptune
            print('{} | {} | acc: {:.4f} | prec: {:.4f} | rec: {:.4f} | f1: {:.4f} |'.format(dataiter_name, eval_mode, acc, prec, rec, f1))
            if neptune is not None:
                neptune.set_property(dataiter_name + ' acc', acc)
                neptune.set_property(dataiter_name + ' prec', prec)
                neptune.set_property(dataiter_name + ' rec', rec)
                neptune.set_property(dataiter_name + ' f1', f1)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument('--use_neptune', type=int)
    # exp_name
    parser.add_argument('--exp_name', type=str)
    # dataset
    parser.add_argument('--dataset1', type=str)
    parser.add_argument('--dataset2', type=str)
    parser.add_argument('--dim_input', type=int)
    parser.add_argument('--rnn_len', type=int)
    parser.add_argument('--csv_path1', type=str)
    parser.add_argument('--csv_path2', type=str)
    parser.add_argument('--ids_path1', type=str)
    parser.add_argument('--ids_path2', type=str)
    parser.add_argument('--stat_path1', type=str)
    parser.add_argument('--stat_path2', type=str)

    # feature mapping
    parser.add_argument('--use_feature_mapping', type=int)
    parser.add_argument('--dim_feature_mapping', type=int)

    # enc
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--nlayer', type=int)
    # dnn-enc
    parser.add_argument('--dim_enc', type=int)
    # rnn-enc
    parser.add_argument('--bidirectional', type=int)
    parser.add_argument('--dim_lstm_hidden', type=int)
    # transformer-enc
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--dim_feedforward', type=int)
    # readout
    parser.add_argument('--reduce', type=str)

    # clf
    parser.add_argument('--classifier', type=str)
    parser.add_argument('--clf_n_lstm_layers', type=int)
    parser.add_argument('--clf_n_fc_layers', type=int)
    parser.add_argument('--clf_dim_lstm_hidden', type=int)
    parser.add_argument('--clf_dim_fc_hidden', type=int)
    parser.add_argument('--clf_dim_output', type=int)

    # training parameter
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--patience', type=float)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--drop_p', type=float)

    # learning rate decay
    parser.add_argument('--use_scheduler', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n_decay', type=int, default=3)

    parser.add_argument('--use_prev_pred', type=int)
    parser.add_argument('--teacher_forcing_ratio', type=float)

    args = parser.parse_args()
    params = vars(args)

    args.save_dir = "./result/"
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.use_neptune == 1:
        import neptune
        neptune.init("hosewq/attackdetection")
        experiment = neptune.create_experiment(name=args.exp_name, params=params)
        args.out_file = experiment.id + '.pth'
    else:
        neptune=None
        args.out_file = 'dummy.pth'

    print('parameters:')
    print('='*90)
    print(params)
    print('='*90)

    train_main(args, neptune)
