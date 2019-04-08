#coding:utf-8

from config import *
import pandas as pd
import numpy as np

# 提取时间的 年-月-日
# 2018-03-22 16:31:44  =>  2018-03-22
train_log = pd.read_csv(train_log_path, sep='\t')
train_log_copy = train_log.copy()
# 在后面添加一列
train_log_copy['OCC_TIM_1'] = None
# 给OCC_TIM_1列赋值
for idx in range(len(train_log)):
    train_log_copy.loc[idx, 'OCC_TIM_1'] = train_log_copy.loc[idx, 'OCC_TIM'].split(' ')[0]
train_log_copy.to_csv(train_log_feature_path,index=False,sep='\t')
