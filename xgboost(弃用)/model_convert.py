# -*- coding: utf-8 -*-
import re
import math
TREE_NUMS = 50#使用多少棵树来构造模型
TREE_NODENUMS = 32#单棵树节点数为32
def sigmoid(x):
    return (float) ( 1.0 / (1.0 + math.exp(-x)) );
'''解析XGBoost.dump模型'''
def load_model():
    model_fobj = open(r'E:\workspace\spyder_workspace\xgboost_sat\data\model\1\near_surf.model.dump','r')
    pattern_booster = re.compile('booster\[\d+\]')
    node_booster = re.compile('\d+:(\[f\d+<\d+.\d+\]|leaf=(-)?(\d.)?\d+)')
    boosters = []
    tree = []
    for line in model_fobj.readlines():
        line = line.strip()
        booster = re.search(pattern_booster,line)
        if booster != None:#booster[d]
#            print(line)
            if len(tree) != 0:
                boosters.append(tree)
                tree = []
        else:
            node = re.search(node_booster,line)
            if node != None:
                tree.append(node.group(0))
    return boosters  
'''将模型转化成数组形式表示'''  
def covert_to_dsp_model():

    boosters = load_model()
    boosters = boosters[:TREE_NUMS]
    nodeattrs = [0 for i in range(TREE_NUMS*TREE_NODENUMS)]#节点值为-1表示叶子节点
    nodevalues = [0.0 for i in range(TREE_NUMS*TREE_NODENUMS)]#节点值为-1表示叶子节点
    for ii in range(TREE_NUMS):
        booster = boosters[ii]
        for strnode in booster:
            if strnode.find('<')!=-1:#非叶子节点
                pair = strnode.split('<')
                node_num = int(pair[0][:pair[0].find(':')])
                node_num = ii*TREE_NODENUMS + node_num
                attr_num = pair[0][pair[0].find('f')+1:]
                node_value = pair[1][:-1]
                nodeattrs[node_num] = int(attr_num)
                if node_value.find('-') != -1:
                    nodevalues[node_num] = float(node_value[:9])#9表示负数精确到小数点后6位
                else:nodevalues[node_num] = float(node_value[:8])#8表示正数精确到小数点后6位
#                print(node_num)
            elif strnode.find('leaf')!=-1:#叶子节点
                node_num = int(strnode[:strnode.find(':')])
                node_num = ii*TREE_NODENUMS + node_num
                node_value = strnode[strnode.find('=')+1:]
                nodeattrs[node_num] = -1#表示叶子节点
                if node_value.find('-') != -1:
                    nodevalues[node_num] = float(node_value[:9])#9表示负数精确到小数点后6位
                else:nodevalues[node_num] = float(node_value[:8])#8表示正数精确到小数点后6位
    return nodeattrs,nodevalues
'''将模型数据写成数组形式'''
def write_xgboost_dsp_model_data():
    nodeattrs,nodevalues = covert_to_dsp_model()  
    attrs = '{'
    for attr in nodeattrs:
        attrs = attrs + str(attr) + ','
    attrs = attrs[:-1]#去掉最后的逗号
    attrs +='}'
      
    values = '{'
    for value in nodevalues:
        values = values + str(value) + ','
    values = values[:-1]#去掉最后的逗号
    values +='}'
    outobj = open('./data/model/xgboost_dsp_model.txt','w')
    outobj.writelines(attrs + ';\n')
    outobj.writelines(values + ';')
    outobj.close()
        
def predict(feature):
    nodeattrs,nodevalues = covert_to_dsp_model()  
    score = 0
    for ii in range(TREE_NUMS):#处理第ii棵树
        index = 0#第index个节点
        offset = ii*TREE_NODENUMS#偏移到当前树
        while True:
            attr_num = nodeattrs[index+offset]
            if attr_num == -1:#到叶子节点了
                score += nodevalues[index+offset]
                print(nodevalues[index+offset])
                break
            if feature[attr_num] < nodevalues[index+offset] :#找节点对应的属性编号                   
                index = index*2 + 1
            else:index = index*2 + 2
    print('score before sigmoid',score)
    score = sigmoid(score)
    return score
def test():
    objtype = 'near'
    dim = 972
    featspath = "H:\\飒特项目\\输出文件\\特征\\features1\\0.1_0.5-0.5_1\\"+objtype+"_valid.txt"
    featobj = open(featspath,'r')
    for line in featobj.readlines():
        feature = [0.0 for i in range(dim)]
        line = line.strip()
        label = line[:line.find(' ')]
        line = line[line.find(' ') + 1:]
        attrs = line.split(' ')
        for attr in attrs:
            index = int(attr[:attr.find(':')])
            value = float(attr[attr.find(':')+1:])
            feature[index] = value
        score = predict(feature)
        print(score)
        break
#write_xgboost_dsp_model_data()
test()
#nodeattrs,nodevalues = covert_to_dsp_model()   
             
            

#print(boosters)