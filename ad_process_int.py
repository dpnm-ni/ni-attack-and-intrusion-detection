import numpy as np
import argparse
import math
import pickle as pkl
import pandas as pd
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

int_data = pd.read_csv('/home/dpnm/cnsm2022/data/int/int_total.csv')
print(int_data.head())
print(int_data.info())

int_df = int_data.fillna(0)
print(int_df.head())
print(int_df.info())

int_df = int_df.replace([np.inf, -np.inf], np.nan)
int_df = int_df.dropna(axis=0)
int_df = int_df.reset_index(drop=True)

print(int_df.head())
print(int_df.info())

int_df = int_df.drop(['path', 'proto'], axis=1)
print(int_df.head())
print(int_df.info())

int_df.to_csv('/home/dpnm/cnsm2022/data/int/int_total_processed.csv', index = False)
print('done')

std_scaler = StandardScaler()
int_df_x = int_df.drop('classification', axis=1, inplace=False)
int_df_y = int_df.loc[:, ['classification']]
y_series = int_df_y.squeeze()

print(int_df_x.head())
print(int_df_y.head())

# Std. Scaler
st_fitted = std_scaler.fit(int_df_x)
print(st_fitted.mean_)

st_output = std_scaler.transform(int_df_x)
st_output = pd.DataFrame(st_output, columns=int_df_x.columns, index=list(int_df_x.index.values))
print(st_output.head())
print(st_output.info())

st_df = pd.concat([st_output, y_series], axis=1)
print(st_df)
print(st_df.info())

st_df.to_csv('/home/dpnm/cnsm2022/data/int/int_total_processed_st.csv', index = False)



# MinMax Scaler
min_max_scaler = MinMaxScaler()
mm_fitted = min_max_scaler.fit(int_df_x)
print(mm_fitted.data_max_)

mm_output = min_max_scaler.transform(int_df_x)
mm_output = pd.DataFrame(mm_output, columns=int_df_x.columns, index=list(int_df_x.index.values))
print(mm_output.head())
print(mm_output.info())

mm_df = pd.concat([mm_output, y_series], axis=1)
print(mm_df)
print(mm_df.info())

mm_df.to_csv('/home/dpnm/cnsm2022/data/int/int_total_processed_mm.csv', index = False)
