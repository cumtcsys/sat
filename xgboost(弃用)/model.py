#coding:utf-8

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def lgb_model(X_train, X_test, y_train, y_test, test):
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # valid_sets = [lgb_train, lgb_eval]
    watchlist = [(lgb_train, 'train'), (lgb_eval, 'eval')]

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0  # 表示日志信息
    }

    # train
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=40000,
        valid_sets=lgb_eval,
        early_stopping_rounds=50,
        verbose_eval=100)  # 每多少次显示一次 日志信息

    # 预测分层的测试样本的得分
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # 预测分层的所有样本的得分
    test_pred = gbm.predict(test, num_iteration=gbm.best_iteration)

    return y_pred, test_pred

def xgb_model(X_train, X_test, y_train, y_test, test):
    # 模型参数
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eta': 0.02,
        'max_depth': 5,  # 4 3
        'colsample_bytree': 0.7,  #0.8
        'subsample': 0.7,
        'min_child_weight': 9,  # 2 3
        'silent': 1,
        'eval_metric': 'auc',
        # 'seed': 2018,
    }

    # create dataset for xgboost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_test, label=y_test)
    dtest = xgb.DMatrix(test)
    watchlist = [(dtrain, 'train'), (deval, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        evals=watchlist,
        num_boost_round=500,
        early_stopping_rounds=50,
        verbose_eval=100)  # 每多少次显示一次 日志信息
    # model = xgb.train(
    #     plst,
    #     xgb_train,
    #     watchlist)

    print('Start predicting...')

    # 预测分层的测试样本的得分
    y_pred = model.predict(deval, ntree_limit=model.best_ntree_limit)
    test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)

    return y_pred, test_pred
