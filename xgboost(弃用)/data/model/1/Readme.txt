ѵ��������
params = {
#        'booster': 'dart',
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eta': 0.2,
        'max_depth': 4,  # 4 3
        'colsample_bytree': 0.7,  #0.8
        'subsample': 0.7,
        'min_child_weight': 1,  # 2 3
        'silent': 0,
        'eval_metric': ['error'],
        # 'seed': 2018,
    }

model = xgb.train(
        params,
        dtrain,
        evals=watchlist,#������DMatrix
        num_boost_round=2000,
        early_stopping_rounds=200,
        verbose_eval=100)  # ÿ���ٴ���ʾһ�� ��־��Ϣ

far:���ž�����Ϊ60 ����������Ϊ��434
mid:���ž�����Ϊ47 ����������Ϊ��418
near:���ž�����Ϊ61 ����������Ϊ��586