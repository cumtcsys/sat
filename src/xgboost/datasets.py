# -*- coding: utf-8 -*-
from src.utils import datatools
from src.utils import evaltools
import os
'''
    ==============================================
    将PD程序提取的特征切分成训练集、验证集、测试集
    ==============================================
'''
'''
    @Description: 解析15集程序提取出的特征文件
    @param inttype: far表示远距离，mid表示中距离，near表示近距离
'''
def parse_15sat_features(typestr):
    inpath = 'H:\\飒特项目\\输出文件\\特征\\features1\\完整20个set\\'
    features_dict = {}
    if typestr == 'far':#far
        inpath = inpath + 'far.txt'
    elif typestr == 'mid':#mid
        inpath = inpath + 'mid.txt'
    elif typestr == 'near':#near
        inpath = inpath + 'near.txt'
    infile = open(inpath,'r')
    for line in infile.readlines():
        line = line.strip()
        if line[-1] == ':' or line[-1] == ';':continue#没有ROI,或者特征项全为0
        framekey = line[:10]
        roistr = line[11:line.find(';')]
        roi = [int(item) for item in roistr.split(' ')]
        features = line[line.find(';')+1:]
        if framekey in features_dict.keys():
            fealist = features_dict[framekey]
            fealist.append([roi,features])
        else:
            features_dict[framekey] = [[roi,features]]
    return features_dict
'''
    @Description: 将特征数据集分成训练集与验证集，set00-set10为训练集，set11-set20为验证集
'''
def split_15sat_features_train_valid(prepath,typestr,pos_iou,neg_iou):
    outdir = prepath +'%.1f_%.1f-%.1f_%.1f\\'%(neg_iou[0],neg_iou[1],pos_iou[0],pos_iou[1])
    if not os.path.exists(outdir):os.mkdir(outdir)
    outtrainpath,outvalidpath = '',''
    if typestr == 'far':#far
        outtrainpath = outdir + 'far_train.txt'
        outvalidpath = outdir + 'far_valid.txt'
    elif typestr == 'mid':#mid
        outtrainpath = outdir + 'mid_train.txt'
        outvalidpath = outdir + 'mid_valid.txt'
    elif typestr == 'near':#near
        outtrainpath = outdir + 'near_train.txt'
        outvalidpath = outdir + 'near_valid.txt'
    trainfile,validfile = open(outtrainpath,'w'),open(outvalidpath,'w')
    anns_dict = datatools.parse_all_ann_filtered()
    far = parse_15sat_features(typestr)
    for key in far.keys():#仅处理有提取到特征的帧
        for roiandfea in far[key]:#处理每一个RoI
            roi = roiandfea[0]
            features = roiandfea[1]
            setnum = int(key[:2])
#            if setnum > 15:continue#set16-set20暂时不要
            for GTs in anns_dict[key]:#一帧中的GTs
                if evaltools.iou(GTs[:4],roi) >= pos_iou[0] and evaltools.iou(GTs[:4],roi)<=pos_iou[1]:#大于正样本阈值，当成正样本
                    if(setnum<=10):
                        trainfile.write('+1 ' + features + '\n')
                    elif(setnum > 10 and setnum <= 20):
                        validfile.write('+1 ' + features + '\n')
                elif evaltools.iou(GTs[:4],roi) < neg_iou[1] and evaltools.iou(GTs[:4],roi) >= neg_iou[0]:
                    if(setnum<=10):
                        trainfile.write('-1 ' + features + '\n')
                    elif(setnum > 10 and setnum <= 20):
                        validfile.write('-1 ' + features + '\n')
    trainfile.close()
    validfile.close()
if __name__ == '__main__':
#    pass
    prepath = 'H:\\飒特项目\\输出文件\\特征\\features1\\'
    split_15sat_features_train_valid(prepath,'far',pos_iou=[0.3,1],neg_iou=[0,0.3])
    split_15sat_features_train_valid(prepath,'mid',pos_iou=[0.3,1],neg_iou=[0,0.3])
    split_15sat_features_train_valid(prepath,'near',pos_iou=[0.3,1],neg_iou=[0,0.3])