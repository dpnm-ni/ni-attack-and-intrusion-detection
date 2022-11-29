#!/bin/bash

# logging
EXP_NAME="code_cleaning"
USE_NEPTUNE=0

# dataset
DATASET1='wsd'
DATASET2='lad2'
RNN_LEN=16
BASE_DIR=$HOME'/cnsm/data/'
CSV_PATH1=$BASE_DIR''$DATASET1'.csv'
CSV_PATH2=$BASE_DIR''$DATASET2'.csv'
IDS_PATH1=$BASE_DIR''$DATASET1'.indices.rnn_len16.pkl'
IDS_PATH2=$BASE_DIR''$DATASET2'.indices.rnn_len16.pkl'
STAT_PATH1=$CSV_PATH1'.stat'
STAT_PATH2=$CSV_PATH2'.stat'

# encoder
DIM_FEATURE_MAPPING=24
ENCODER="rnn"
NLAYER=2
DIM_ENC=-1              # DNN-enc
BIDIRECTIONAL=0         # RNN-enc
DIM_LSTM_HIDDEN=40      # RNN-enc
NHEAD=4                 # transformer
DIM_FEEDFORWARD=48      # transformer
REDUCE="self-attention" # mean, max, or self-attention

# classifier
CLASSIFIER='rnn' # dnn or rnn
CLF_N_LSTM_LAYERS=1
CLF_N_FC_LAYERS=3
CLF_DIM_LSTM_HIDDEN=200
CLF_DIM_FC_HIDDEN=600
CLF_DIM_OUTPUT=2

# modified model
USE_PREV_PRED=0
TEACHER_FORCING_RATIO=0.5
if [ $USE_PREV_PRED == 1 ]
then
    DIM_INPUT=24
else
    DIM_INPUT=23
fi

# optimization
OPTIMIZER='Adam'
LR=0.001
DROP_P=0.0
BATCH_SIZE=64
PATIENCE=10
MAX_EPOCH=1
USE_SCHEDULER=0
STEP_SIZE=1
GAMMA=0.5
N_DECAY=3

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDBmMTBmOS0zZDJjLTRkM2MtOTA0MC03YmQ5OThlZTc5N2YifQ=="
export CUDA_VISIBLE_DEVICES=$1

/usr/bin/python3.8 ad_joint_main.py \
    --reduce=$REDUCE \
    --optimizer=$OPTIMIZER \
    --lr=$LR \
    --patience=$PATIENCE \
    --exp_name=$EXP_NAME \
    --dataset1=$DATASET1 \
    --dataset2=$DATASET2 \
    --max_epoch=$MAX_EPOCH \
    --batch_size=$BATCH_SIZE \
    --dim_lstm_hidden=$DIM_LSTM_HIDDEN \
    --dim_feature_mapping=$DIM_FEATURE_MAPPING \
    --nlayer=$NLAYER \
    --bidirectional=$BIDIRECTIONAL \
    --nhead=$NHEAD \
    --dim_feedforward=$DIM_FEEDFORWARD \
    --dim_input=$DIM_INPUT \
    --encoder=$ENCODER \
    --classifier=$CLASSIFIER \
    --dim_enc=$DIM_ENC \
    --clf_n_lstm_layers=$CLF_N_LSTM_LAYERS \
    --clf_n_fc_layers=$CLF_N_FC_LAYERS \
    --clf_dim_lstm_hidden=$CLF_DIM_LSTM_HIDDEN \
    --clf_dim_fc_hidden=$CLF_DIM_FC_HIDDEN \
    --clf_dim_output=$CLF_DIM_OUTPUT \
    --csv_path1=$CSV_PATH1 \
    --csv_path2=$CSV_PATH2 \
    --ids_path1=$IDS_PATH1 \
    --ids_path2=$IDS_PATH2 \
    --stat_path1=$STAT_PATH1 \
    --stat_path2=$STAT_PATH2 \
    --rnn_len=$RNN_LEN \
    --use_neptune=$USE_NEPTUNE \
    --use_scheduler=$USE_SCHEDULER \
    --step_size=$STEP_SIZE \
    --gamma=$GAMMA \
    --n_decay=$N_DECAY \
    --drop_p=$DROP_P \
    --use_prev_pred=$USE_PREV_PRED \
    --teacher_forcing_ratio=$TEACHER_FORCING_RATIO
