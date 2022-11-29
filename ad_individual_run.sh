#!/bin/bash

# logging
EXP_NAME="code_cleaning"
USE_NEPTUNE=0

# dataset
DATASET="int/int_total_processed_mm"           # wsd, lad1, lad2, nsl-kdd/KDDTotal_binary, nsl-kdd/KDDTotal_multi, nsl-kdd/KDDTotal_multi2, unsw-nb15/total_binary_processed, unsw-nb15/total_multi_processed, unsw-nb15/total_multi2_processed, cicids2017/binary/sub_total2, cicids2017/multi/sub_total2, int/int_total_processed_mm
RNN_LEN=16  				    #16
BASE_DIR=$HOME'/cnsm2022/data/'
CSV_PATH=$BASE_DIR''$DATASET'.csv'
IDS_PATH=$BASE_DIR''$DATASET'.indices.rnn_len'$RNN_LEN'.pkl'
STAT_PATH=$CSV_PATH'.stat'

# encoder
DIM_FEATURE_MAPPING=8 #24 for sla detection, 122 for nsl-kdd, 197 for unsw-nb15, 79 for cicids2017, 8 for int
ENCODER=$2
NLAYER=2
DIM_ENC=-1              # DNN-enc
BIDIRECTIONAL=$5         # RNN-enc
DIM_LSTM_HIDDEN=40      # RNN-enc 40
NHEAD=4                 # transformer 4
DIM_FEEDFORWARD=48      # transformer 48
REDUCE=$3 # mean, max, or self-attention

# classifier
CLASSIFIER="rnn"        # dnn or rnn
CLF_N_LSTM_LAYERS=1
CLF_N_FC_LAYERS=3
CLF_DIM_LSTM_HIDDEN=200
CLF_DIM_FC_HIDDEN=600
CLF_DIM_OUTPUT=2

# modified model
USE_PREV_PRED=$4
TEACHER_FORCING_RATIO=0.5 #0.5 0 best
if [ $USE_PREV_PRED == 1 ]
then
    DIM_INPUT=8 # 24 for sla detection, 122 for nsl-kdd, 197 for unsw-nb15, 79 for cicids2017, 8 for int
else
    DIM_INPUT=7 # 23 for sla detection, 121 for nsl-kdd, 196 for unsw-nb15, 78 for cicids2017, 7 for int
fi

# optimization
OPTIMIZER='Adam'
LR=0.001
DROP_P=0.0
BATCH_SIZE=256
PATIENCE=10
MAX_EPOCH=1
USE_SCHEDULER=0
STEP_SIZE=1
GAMMA=0.5
N_DECAY=3

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMjBhY2VhOC1hM2FiLTQ2MDgtOWUxOS05OGEyNzhiNGVkZDgifQ=="
export CUDA_VISIBLE_DEVICES=$1

python3.8 ad_individual_main.py \
    --dataset=$DATASET \
    --reduce=$REDUCE \
    --optimizer=$OPTIMIZER \
    --lr=$LR \
    --patience=$PATIENCE \
    --exp_name=$EXP_NAME \
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
    --csv_path=$CSV_PATH \
    --ids_path=$IDS_PATH \
    --stat_path=$STAT_PATH \
    --data_name=$DATA_NAME \
    --rnn_len=$RNN_LEN \
    --dict_path=$DICT_PATH \
    --use_neptune=$USE_NEPTUNE \
    --use_scheduler=$USE_SCHEDULER \
    --step_size=$STEP_SIZE \
    --gamma=$GAMMA \
    --n_decay=$N_DECAY \
    --drop_p=$DROP_P \
    --use_prev_pred=$USE_PREV_PRED \
    --teacher_forcing_ratio=$TEACHER_FORCING_RATIO
