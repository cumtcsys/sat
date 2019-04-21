# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
def parseDataFile(typestr):
    fobj = open('data_'+typestr+'.txt')
    pos = 0
    items = []
    data = []
    for line in fobj.readlines():
        line = line.strip()
        if '0' == line[0]:
            items[-1] = items[-1] + ' ' + line
            continue
        items.append(line)
    for item in items:
        thresh = int(item[item.find('=')+1:item.find(',')].strip())
        recall = float(item[item.rfind(' ')+1:][:6])
        data.append((thresh,recall))
        
    return data
        
def normalize_data(data_typestr):
    data_max,data_min = data_typestr[0],data_typestr[-1]
#    data_typestr = [(data[1] - data_min[1])/(data_max[1] - data_min[1]) for data in data_typestr]
    result = []
    for data in data_typestr:
        temp = (data[1] - data_min[1])/(data_max[1] - data_min[1])
        result.append(temp)
    return result
            
'''显示召回率随surf响应值变化关系'''
def visual_thresh_recall():
    data_addROI = parseDataFile('addROI')
    normal_data_addROI = normalize_data(data_addROI)
    
    data_recycleProcess = parseDataFile('recycleProcess')
    normal_data_recycleProcess = normalize_data(data_recycleProcess)
    data_final = parseDataFile('final')
    normal_data_final = normalize_data(data_final)
    
    x_addROI = [d[0] for d in data_addROI]
    y_addROI = normal_data_addROI
    x_recycleProcess = [d[0] for d in data_recycleProcess]
    y_recycleProcess = normal_data_recycleProcess
    x_final = [d[0] for d in data_final]
    y_final = normal_data_final
    plt.figure()
    plt.title('召回率随阈值变化趋势')
    plt.plot(x_addROI,y_addROI,color = 'red',label = 'addROI')
    plt.plot(x_recycleProcess,y_recycleProcess,color = 'green',label = 'recycleProcess')
    plt.plot(x_final,y_final,color = 'blue',label = 'final')
    plt.plot([120,120],[0,0.1],color = 'yellow')
    plt.legend()
    plt.ylabel('recall')
    plt.xlabel('thresh')   
    return

def visual_surf_RoI_nums():
    data_source = [4215858,2841518,2103153,1669133,1380377,1177264,1023101,903294,806577,728450,663896,609148,562381,521679 ,
     486472, 427890 ,403415, 380983, 360757, 342293, 
     325721, 310408, 296490, 283491, 271616,260389 ,249953 ,240192 ,231212 ,222633 ,214420 ,206597 ,199182 ,192274 ,185580 ,179286 ,173297, 167546,162031 ,156704 ,151530 ,146641,141815,137232]
    datas = []
    for i in range(len(data_source)):
        if i%4==0:datas.append(data_source[i])
            
#    label_list =  range(len(datas))*4
    num_list = datas
    x = range(len(num_list))
    x = [(item*4+1)*6 for item in x]
    rects = plt.bar(left=x,height = num_list,width = 15,color = 'yellow',label = 'surf')
    plt.xticks(x)
    plt.xlabel('surf阈值')
    plt.ylabel('RoI数量')
    plt.title('头部RoI数量')
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2 , height+2, str(height), ha="center", va="bottom")

def visual_frames_RoI_nums():
    #recall24 
    datas_fast = [41594, 16690, 15697, 13917, 12628, 11537, 10278, 9365, 8561, 7502, 6857, 6153, 5453, 5069, 4544, 4132, 3608, 3170, 2800, 2649, 2300, 2073, 1859, 1649, 1478, 1347, 1060, 926, 767, 663, 569, 498, 491, 432, 380, 319, 274, 230, 207, 179, 139, 111, 100, 76, 73, 61, 47, 64, 47, 35, 40, 32, 24, 26, 31, 25, 27, 16, 14, 11, 7, 20, 9, 8, 11, 9, 11, 3, 12, 1, 6, 3, 2, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #recall25 
    datas_road = [34619, 18640, 17086, 16154, 13970, 12252, 11619, 10222, 8844, 7907, 7469, 6976, 6402, 5858, 5121, 4296, 3841, 3301, 2733, 2306, 1930, 1655, 1326, 1116, 885, 743, 524, 537, 388, 367, 293, 246, 206, 199, 149, 108, 88, 79, 82, 55, 53, 50, 32, 37, 29, 25, 22, 18, 12, 14, 9, 15, 16, 15, 13, 11, 12, 12, 10, 4, 7, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    datas_road[30] = sum(datas_road[30:])
    datas_road = datas_road[:31]
    datas_fast[30] = sum(datas_fast[30:])
    datas_fast = datas_fast[:31]
    label_list =  range(len(datas_road))
    x = range(len(datas_fast))
    rects_road = plt.bar(left=x,height = datas_road,width = 0.4,color = 'blue',label = 'road')
    rects_fast = plt.bar(left=[i+0.4 for i in x],width = 0.4,height = datas_fast,color = 'orange',label = 'fast')
    for rect in rects_road:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
    for rect in rects_fast:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
    plt.ylabel('总帧数')
    plt.xlabel('单帧RoI数量')
    plt.title('路面过滤-单帧提取RoI数量统计')
    plt.legend()
#visual_thresh_recall()
visual_frames_RoI_nums()
#visual_surf_RoI_nums()
