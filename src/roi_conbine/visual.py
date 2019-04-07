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
    datas = [34619, 18607, 17094, 16119, 13945, 12254, 11639, 10182, 8852, 7897, 7496, 6982, 6422, 5833, 5137, 4304, 3864, 3298, 2729, 2315, 1941, 1646, 1335, 1114, 884, 749, 529, 532, 389, 373, 291, 252, 207, 196, 152, 107, 87, 82, 83, 55, 53, 49, 32, 38, 29, 25, 21, 18, 13, 14, 7, 16, 16, 15, 14, 9, 14, 10, 10, 6, 7, 1, 2, 0]
    datas[20] = sum(datas[20:])
    datas = datas[:21]
    label_list =  range(len(datas))
    num_list = datas
    x = range(len(num_list))
    rects = plt.bar(left=x,height = num_list,color = 'green',label = 'surf')
    plt.xlabel('总帧数')
    plt.ylabel('单帧RoI数量')
    plt.title('surf+dts单帧提取RoI数量统计')
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
#visual_thresh_recall()
visual_frames_RoI_nums()
#visual_surf_RoI_nums()
