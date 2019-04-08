#coding:utf-8

from config import *
import pandas as pd
import numpy as np
import time

start_time =time.time()
print('begin',start_time)

# 读取客户的个人属性与信用卡消费数据
train_agg = pd.read_csv('../data_original/train/train_agg.csv', sep='\t')
test_agg = pd.read_csv('../data_original/test/test_agg.csv', sep='\t')
agg = pd.concat([train_agg, test_agg])

# 读取日志信息
train_log = pd.read_csv('../data_original/train/train_log.csv', sep='\t')
test_log = pd.read_csv('../data_original/test/test_log.csv', sep='\t')
log = pd.concat([train_log, test_log])

#用户唯一标识
train_flg = pd.read_csv('../data_original/train/train_flg.csv', sep='\t')
test_flg = pd.read_csv('../data_original/submit_sample.csv', sep='\t')
test_flg['FLAG'] = -1
flg = pd.concat([train_flg, test_flg])
del test_flg['RST']
flg = pd.concat([train_flg, test_flg])

# 只是用信用卡的匿名特征
data = pd.merge(flg,agg,on=['USRID'])

# 采取分层采样
from sklearn.model_selection import StratifiedKFold

# 提取测试集和训练集合
train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]
print('train',train.shape)
print('test',test.shape)

# 构造数据
# 提取userid和单独把标签赋值一个变量
train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test = test[col].values

N = 2
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

xx_cv = []
xx_pre = []
xx_beat = {}

import operator

for k,(train_in,test_in) in enumerate(skf.split(X,y)):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

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
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=250)

    print('Start predicting...')
    # 预测分层的测试样本的得分
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # 遍历选择最好的阈值
    # 并且把最高线下分数对应的与之存入字典
    tmp_auc = {}
    for t in [float(x)/50 for x in range(1,30)]:
        tmp_auc[t] = roc_auc_score(y_test,np.where(y_pred>t,1,0))

    tmp_auc = sorted(tmp_auc.items(), key=operator.itemgetter(1),reverse=True)
    # 选取最好roc得分对应的阈值
    xx_beat[k] = tmp_auc[0][0]

    # 与最好的阈值再进行比较，再分类和打分
    xx_cv.append(roc_auc_score(y_test,np.where(y_pred>xx_beat[k],1,0)))
    # 预测所有测试样本的得分
    xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))
    print 'tmp_auc', tmp_auc


print('threshold set', xx_beat)
print('rescore', xx_cv)
print('best xx score', np.mean(xx_cv))
print('all samples score', xx_pre)


# 线下投票根据每次的阈值选择
# 根据每一次的阈值，在相对应的那次的测试集中再进行分类，并转化为内部array为20000*1
# 得到每个阈值对应的测试集的分类结果的汇总array
xx_pre_yu = []
s = 0
for k,i in enumerate(xx_pre):
    print(k,xx_beat[k])
    if k == 0: # 第一个阈值
        xx_pre_yu = np.where(np.array(i)>xx_beat[k],1,0).reshape(-1,1)
    else:
        xx_pre_yu = np.hstack((xx_pre_yu,np.where(np.array(i)>xx_beat[k],1,0).reshape(-1,1)))

result = []
for i in xx_pre_yu:
    # np.argmax(np.bincount(i)): 相当于“与”运算，如两次预测均为1，则最后结果为1，其余为0
    result.append(np.argmax(np.bincount(i)))

res = pd.DataFrame()
res['USRID'] = list(test_userid.values)
res['RST'] = list(result)
print(res[res['RST']==1].shape)

time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
res.to_csv('../result/%s_%s.csv'%(str(time_date),str(np.mean(xx_cv)).split('.')[1]),index=False,sep='\t')

print('end time',time.time() - start_time)

print('info')
print('offline score',np.mean(xx_cv))
print('testset: predict positive samples ratio',float(res[res['RST']==1].shape[0])/ res.shape[0])
print('trainset: positive samples ratio', float(train_flg[train_flg['FLAG'] == 1].shape[0]) / train_flg.shape[0])
