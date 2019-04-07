# -*- coding: utf-8 -*-
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import xgbtrain
from xgboost import plot_importance
from matplotlib import pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")
 
#alg传入XGBOOST,X_train传入训练数据特征信息，Y_train传入训练数据标签信息  X_testdata最后要预测的值
def XGBmodelfit(alg, X_train, Y_train,X_test=None,Y_test=None,X_predictions=None,useTrainCV=True, cv_folds=5, early_stopping_rounds=200):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=Y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='error', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])
 
    #训练模型
    alg.fit(X_train, Y_train,eval_metric='error')
 
    #预测结果:
    dtrain_predictions = alg.predict(X_test)  #输出 0 或 1
    # dtrain_predprob = alg.predict_proba(X_test)[:,1]   #输出概率
 
    #打印报告信息:
    print("\nModel Report")
    print("Accuracy  (Train) : %.4g" % metrics.accuracy_score(Y_test, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(Y_test, dtrain_predictions))
    print(alg)
    print("the best:")
    print(cvresult.shape[0])
    plot_importance(alg)
    plt.show()
 
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
 

 
x_train,y_train,x_valid,y_valid,x_test,y_test = xgbtrain.built_dataset()
 
xgb1 = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)
 
# XGBmodelfit(xgb1,X_train,y_train,X_test,y_test)
 
param_grid = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(1,9,1),
 'subsample': np.arange(0.1,1.0,0.1),
 'colsample_bytree':np.arange(0.1,1.0,0.1)
}
# param_grid = {
#  'max_depth':[7,8],
#  'min_child_weight':[4,5]
# }


#gsearch1 = GridSearchCV(estimator = XGBClassifier(
#       learning_rate =0.1, n_estimators=140, max_depth=9,
#       min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,
#       objective= 'binary:logistic', nthread=4,scale_pos_weight=1, seed=27),
#       param_grid=param_grid,cv=10)


gsearch1 = GridSearchCV(estimator = XGBRegressor(
       learning_rate =0.2, 
       objective= 'binary:logistic', 
       booster= 'gbtree',
        eta=0.2,
        max_depth=4,  # 4 3
        colsample_bytree=0.7,  #0.8
        subsample= 0.7,
        min_child_weight=1,  # 2 3
        silent= 0,
        eval_metric='error',),
       param_grid=param_grid,cv=10)
gsearch1.fit(np.array(x_train),np.array(y_train))
print(gsearch1.best_params_,gsearch1.best_score_)

