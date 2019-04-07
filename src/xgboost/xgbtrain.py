#coding:utf-8
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt

'''
    ============================================
    训练xgboost、网格搜索调优xgboost
    ============================================
'''
'''
    @Description: 解析特征文件
    @param path: 文件路径
    @return datas(list),labels(list): datas与labels标签对应
'''
def read_data(path,dim):
    # 读取数据
    fobj = open(path)
    datas = []#样本特征值
    labels = []#标签
    i = 0#解析到第i行
    for line in fobj.readlines()[:-2]:
        if line[2:].strip()=='':continue#去掉空行
        label = int(line[0:2])#取行首标签(+1,-1)
        if label == -1:label = 0#负样本标签设置为0
        #中距离与远距离特征维数为756，近距离特征维数为972
        features = np.zeros(dim,dtype = float)
        features_strs = line[2:].strip().split(' ')#分割各特征值
        for fstr in features_strs:
            index = int(fstr[:fstr.find(':')])#取特征值索引
            feature = float(fstr[fstr.find(':')+1:])#特征值
            features[index] = feature
        datas.append(features)
        labels.append(label)
        i = i + 1
        if i%1000 == 0:
            print('解析第%d个样本'%(i))
    print('样本数：%d'%(i))
    return datas,labels
'''
    @Description:将数据集分成训练集、验证集、测试集
    @return x_train(list),y_train(list),x_valid(list),y_valid(list),x_test(list),y_test(list):
'''
def built_dataset(typestr = 'near',dim=972):
    ratio = 0.2#取20%作为验证集，余下作为测试集
    trainpath = 'H:\\飒特项目\\输出文件\\特征\\features1\\0.1_0.5-0.5_1//'+typestr+'_train.txt'
    testvalidpath = 'H:\\飒特项目\\输出文件\\特征\\features1\\0.1_0.5-0.5_1//'+typestr+'_valid.txt'
    x_train,y_train = read_data(trainpath,dim)#导入训练集
    x_valid_test,y_valid_test = read_data(testvalidpath,dim)#导入验证集与测试集
    splitpos = int(len(x_valid_test)*ratio)
    x_valid,y_valid = x_valid_test[:splitpos],y_valid_test[:splitpos]#验证集
    x_test,y_test = x_valid_test[splitpos:],y_valid_test[splitpos:]#测试集
    return x_train,y_train,x_valid,y_valid,x_test,y_test
'''
    @Description:网格搜索选择最优超参数
'''
def xgb_grid_search():
    x_train,y_train,x_valid,y_valid,x_test,y_test = built_dataset('far',756)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    dtest = xgb.DMatrix(x_test, label=y_test)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]#训练过程中输出精度
    out_info = open('./data/gridsearch.log','a')
    eta_list = [0.1,0.2,0.3,0.4,0.5,0.7,0.9]
    max_depth_list = [2,3,4,5,6,7,9]
    colsample_bytree_list = [0.1,0.3,0.5,0.7,0.8,0.9]
    subsample_list = [0.1,0.3,0.5,0.7,0.8,0.9]
    min_child_weight_list = [1,3,4,5,7,9]
    finished_i = 0
    for eta in eta_list:
        for max_depth in max_depth_list:
            for colsample_bytree in colsample_bytree_list:
                for subsample in subsample_list:
                    for min_child_weight in min_child_weight_list:
                        params = {
                #        'booster': 'dart',
                        'booster': 'gbtree',
                        'objective': 'binary:logistic',
                        'eta': eta,
                        'max_depth': max_depth,  # 4 3
                        'colsample_bytree': colsample_bytree,  #0.8
                        'subsample': subsample,
                        'min_child_weight': min_child_weight,  # 2 3
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
                        out_info.writelines('eta:%f max_depth:%d colsample_bytree:%f subsample:%f min_child_weight:%f'%(eta,max_depth,colsample_bytree,subsample,min_child_weight))
                        out_info.writelines("best_ntree_limit=%d\n"%(model.best_ntree_limit))#打印验证集最优的树个数
                        train_pred = model.predict(dtrain, ntree_limit=model.best_ntree_limit)#训练集预测结果
                        valid_pred = model.predict(dvalid, ntree_limit=model.best_ntree_limit)#验证集预测结果
                        test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)#测试集预测结果
                        
                        train_result = eval_xgboost(train_pred,y_train)
                        out_info.writelines('训练集：\n')
                        out_info.writelines(train_result)
                        
                        valid_result = eval_xgboost(valid_pred,y_valid)
                        out_info.writelines('验证集：\n')
                        out_info.writelines(valid_result)
                        
                        test_result = eval_xgboost(test_pred,y_test)
                        out_info.writelines('测试集：\n')
                        out_info.writelines(test_result)
                        out_info.writelines('++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
                        out_info.flush()
                    finished_i += 1
                    rounds = len(eta_list)*len(max_depth_list)*len(colsample_bytree_list)*len(subsample)
                    print('完成度：%lf'%(finished_i*1.0/rounds))
    out_info.close()                      
                        
'''
    @Description:xgb模型训练
    @Param typestr: far,mid,near,远，中，近训练，dim特征维度数
'''
def xgb_train(typestr,dim):
    x_train,y_train,x_valid,y_valid,x_test,y_test = built_dataset(typestr,dim)
    # 模型参数
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
    # create dataset for xgboost
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    dtest = xgb.DMatrix(x_test, label=y_test)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]#训练过程中输出精度
    model = xgb.train(
        params,
        dtrain,
        evals=watchlist,#评估用DMatrix
        num_boost_round=2000,
        early_stopping_rounds=200,
        verbose_eval=100)  # 每多少次显示一次 日志信息
    print("best_ntree_limit=%d"%(model.best_ntree_limit))#打印验证集最优的树个数
    train_pred = model.predict(dtrain, ntree_limit=model.best_ntree_limit)#训练集预测结果
    valid_pred = model.predict(dvalid, ntree_limit=model.best_ntree_limit)#验证集预测结果
    test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)#测试集预测结果
    print('训练集：')
    eval_xgboost(train_pred,y_train)
    print('验证集：')
    eval_xgboost(valid_pred,y_valid)
    print('测试集：')
    eval_xgboost(test_pred,y_test)
    print('dir model')
    model.dump_model('./data/model/'+typestr+'_surf.model.dump',fmap = '',with_stats = True)
    model.save_model('./data/model/'+typestr+'_surf.model')
#    xgb.plot_importance(model)
    plt.show()
    print('save model success')
    
'''
    @Description:导入训练好的模型进行预测
'''
def predict():
    testpath = r'H:\飒特项目\输出文件\特征\features1\0.1_0.5-0.5_1/far_train.txt'
    x_test,y_test = read_data(testpath)
    model = xgb.Booster(model_file = './data/model/far_surf.model')
    
    dtest = xgb.DMatrix(x_test[:50], label=y_test[:50])

#    y_pre = model.predict(dtest,pred_leaf = True,pred_contribs = True)
    y_pre_value = model.predict(dtest)
    y_pre_leaf = model.predict(dtest,pred_leaf = True)
    y_pre_contrib = model.predict(dtest,pred_contribs = True)
#    import math
#    a1 = math.exp(sum(y_pre_contrib[0]))
#    a2 = math.exp(sum(y_pre_contrib[1]))
#    score1 = 1.0 / (1.0 + math.exp(-sum(y_pre_contrib[0])))
#    score2 = 1.0 / (1.0 + math.exp(-sum(y_pre_contrib[1])))
#    aa1 = math.exp(sum(y_pre_contrib[0][:-2]))
#    sscore1 = 1.0 / (1.0 + math.exp(-aa1))
#    shap.summary_plot(y_pre_contrib, dtest)
    return y_pre_value
'''
    @Description:计算混淆矩阵TP,TN,FP,FN,计算精度与召回率
'''
def eval_xgboost(pre,lbl,threshold = 0.5):
    TP,TN,FP,FN = 0,0,0,0
    for i in range(len(pre)):
        if pre[i]>threshold:#预测为正
            if abs(lbl[i] - 1) <0.0001:#标签为正
                TP = TP + 1
            else: FP = FP + 1
        else:#预测为负
            if abs(lbl[i] - 1) <0.0001:#标签为正
                FN = FN + 1
            else: TN = TN + 1
    ACCU = (TP + TN) * 1.0 / len(pre)
    RECALL = (TP)*1.0 / (TP + FN)
    res = 'ACCU = %lf;RECALL = %lf;threshold = %.4lf;TP = %d;TN = %d;FP = %d;FN = %d;\n' % (ACCU,RECALL,threshold,TP,TN,FP,FN)
    print(res)
    return res
if __name__ == "__main__":
    xgb_train('mid',756)
#    xgb_grid_search()
#    
#    x_train,y_train = read_data('./data/wk/16/train/far.txt')
#    x_train,y_train = read_data('./data/TrainSamples/middle.txt')
##    x_test,y_test = read_data('./data/wk/svm_fp/far.txt')
#    x_test,y_test = read_data('./data/wk/test/annoed_pos/middle.txt')
#    for i in range(20000):
#        x_train.append(x_test[i])
#        y_train.append(y_test[i])
#    xgb_train(x_train,y_train,x_test[20000:],y_test[20000:], test = 0)
#    
#    pre,lbl = predict()
#    eval_xgboost(pre,lbl,0.5)
#    for i in range(10000):
#        eval_xgboost(pre,lbl,i*1.0/10000)
    

    
#    for r in result:
#        print(r)
    
#    x_train,y_train,x_test,y_test = built_seg_dataset()
#    xgb_train(x_train,y_train,x_test,y_test,0)
#    
