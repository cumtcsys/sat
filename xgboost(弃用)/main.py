#coding:utf-8

import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score
import operator
from sklearn.model_selection import StratifiedKFold

from data_process import *
# from lgb_model import *
from model import *
from predict import *



if __name__ == '__main__':

    start_time =time.time()

    print('begin',start_time)

    print 'data_process...'
    data_dict = build_dataset()  # baseline的数据处理

    print 'train modeland predict...'
    xx_cv = []
    xx_pre = []

    X = data_dict['train_X']
    y = data_dict['train_y']
    test = data_dict['test']  # 所有测试阉割版

    # 采取分层采样，保证每个子数据集的类别比例一致
    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=42)
    KFold_datasets = skf.split(X, y)

    for k, (train_in, test_in) in enumerate(KFold_datasets):
        X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]

        # 返回每个模型的子数据集预测结果 y_pred 和所有测试样本的结果 test_pred
        # y_pred, test_pred = lgb_model(X_train, X_test, y_train, y_test, test)
        y_pred, test_pred = xgb_model(X_train, X_test, y_train, y_test, test)

        xx_cv.append(roc_auc_score(y_test, y_pred))
        xx_pre.append(test_pred)

    # 对n个模型的预测得分求平均
    print 'predict all testset...'
    pred_result = 0
    for i in xx_pre:
        pred_result = pred_result + i
    pred_result = pred_result / N


    # 将结果写入文件
    res = pd.DataFrame()
    res['USRID'] = list(data_dict['test_userid'].values)
    res['RST'] = list(pred_result)

    time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    res.to_csv('../submit/%s_%s.csv'%(str(time_date),str(np.mean(xx_cv)).split('.')[1]),index=False,sep='\t')

    print('end time',time.time() - start_time)
    print('offline score',np.mean(xx_cv))
    print('done')
