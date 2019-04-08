训练参数：
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
        evals=watchlist,#评估用DMatrix
        num_boost_round=2000,
        early_stopping_rounds=200,
        verbose_eval=100)  # 每多少次显示一次 日志信息

far:最优决策树为60 无用特征数为：434
mid:最优决策树为47 无用特征数为：418
near:最优决策树为61 无用特征数为：586