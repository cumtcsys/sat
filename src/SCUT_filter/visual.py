# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Song Yang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################import matplotlib.pyplot as plt
'''
    DESCRIPTION: parse the last line of the eval file which contains the recall information of the whole SCUT
                 total:[827 3502 0.236151] 40240 -> [827 3502 0.236151]
    PARAMETERS:
        path: where the eval path located
    RETURN:
        
'''
def parse_eval(path):
    #total:[827 3502 0.236151] 40240
    lastline = readlastline(path)
    content = lastline[lastline.find('[')+1:lastline.find(']')]
    recall = float(content.split(' ')[-1])
    return recall
'''
    DESCRIPTION:
        parse all eval result file to dict (slide_label)
    RETURN: a dict with all recall data
'''
def get_slide_label_data():
    recall_data = {}
    for cs in ['sate','common']:
        if not recall_data.has_key(cs):recall_data[cs] = {}
        for label in ['label']:
            if not recall_data[cs].has_key(label):recall_data[cs][label] = {}
            for withoccl in [True,False]:
                if not recall_data[cs][label].has_key('withoccl'):recall_data[cs][label]['withoccl'] = {}
                if not recall_data[cs][label].has_key('withoutoccl'):recall_data[cs][label]['withoutoccl'] = {}
                for ious in [0.3,0.4,0.5,0.6,0.7]:
                    if withoccl == True:
                        recall_data[cs][label]['withoccl'][ious] = parse_eval('./data/%s/slide/eval_withoccl_%s_%01.01f_0.8_.txt'%(cs,label,ious))
                    else:
                        recall_data[cs][label]['withoutoccl'][ious] = parse_eval('./data/%s/slide/eval_%s_%01.01f_0.8_.txt'%(cs,label,ious))            
    return recall_data  
'''
    DESCRIPTION:
        parse all eval result file to dict
    RETURN: a dict with all recall data
'''
def get_recall_data():
    recall_data = {}

    for cs in ['sate','common']:
        if not recall_data.has_key(cs):recall_data[cs] = {}
        for label in ['label','addROI','simpleTrace','sizeFilter','positionFilter','recycleProcess','haarLike','hogLBP','final']:
            if not recall_data[cs].has_key(label):recall_data[cs][label] = {}
            for withoccl in [True,False]:
                if not recall_data[cs][label].has_key('withoccl'):recall_data[cs][label]['withoccl'] = {}
                if not recall_data[cs][label].has_key('withoutoccl'):recall_data[cs][label]['withoutoccl'] = {}
                for ious in [0.3,0.4,0.5,0.6,0.7]:
                    if withoccl == True:
                        recall_data[cs][label]['withoccl'][ious] = parse_eval('./data/%s/eval_withoccl_%s_%01.01f_0.8_.txt'%(cs,label,ious))
                    else:
                        recall_data[cs][label]['withoutoccl'][ious] = parse_eval('./data/%s/eval_%s_%01.01f_0.8_.txt'%(cs,label,ious))            
    return recall_data  

#def visual_recall(recall_data):
#    for cs in ['sate','common']:
#        for label in ['label','addROI','simpleTrace','sizeFilter','positionFilter','recycleProcess','haarLike','hogLBP','final']:
#            for withoccl in [True,False]:
#                if withoccl == True:
#                    x,y = [],[]
#                    for k_iou in sorted(recall_data[cs][label]['withoccl'].keys()):
#                        x.append(k_iou)
#                        y.append(recall_data[cs][label]['withoccl'][k_iou])
#                    plt.plot(x,y,linestyle = '--',marker = 'x')
#                else:
#                    for k_iou in sorted(recall_data[cs][label]['withoutoccl'].keys()):
#                        x.append(k_iou)
#                        y.append(recall_data[cs][label]['withoutoccl'][k_iou])
#                    plt.plot(x,y)  
#    plt.show()

'''
    DESCRIPTION:
        draw several eval method(label,addROI,recycleProcess,final) result. iou = 0.5
    PARAMERTERS:
        recal_data: parsed eval file last line
        typestr: title of the figure
'''
def visual_iou_0_5(recall_data,typestr = 'iou = 0.5'):    
    plt.figure(figsize = (8,5))
    plt.title('%s recall'%(typestr))
    plt.xlabel('step')
    plt.ylabel('recall')
    plt.xticks([0.3,0.4,0.5,0.6,0.7])
    llist = []
    x,y = ['label','addROI','recycleProcess','final'],[]
    for cs in ['sate','common']:
        for withoccl in [True,False]:
            y = []
            if withoccl == True:
                for label in ['label','addROI','recycleProcess','final']:                           
                    y.append(recall_data[cs][label]['withoccl'][0.5])                    
            #plt.xticks([0.2,0.4,0.6,0.8],x)
                l1, = plt.plot([0.2,0.4,0.6,0.8],y,linestyle = '--',linewidth = 3,marker = 'x')
                llist.append(l1)
                print x,y
            y = []
            if withoccl == False:
                for label in ['label','addROI','recycleProcess','final']:       
                    y.append(recall_data[cs][label]['withoutoccl'][0.5])
                plt.xticks([0.2,0.4,0.6,0.8],x)
                l2, = plt.plot([0.2,0.4,0.6,0.8],y,linewidth = 3,marker = 'o')  
                llist.append(l2)
                print x,y
    plt.legend(handles = llist,labels = ['sate withoccl','sate withoutoccl','common withoccl','common withoutoccl'],loc = 'best')
    plt.show()
'''
    DESCRIPTION:
        draw single eval method result (x:iou = 0.3->0.7   ;   y:recall[0.3]->recall[0.7])
    PARAMERTERS:
        recal_data: parsed eval file last line
        typestr: eval method name
'''
def visual_single(recall_data,typestr = 'final'):    
    plt.figure(figsize = (8,5))
    plt.title('%s recall'%(typestr))
    plt.xlabel('iou')
    plt.ylabel('recall')
    plt.xticks([0.3,0.4,0.5,0.6,0.7])
    llist = []
    for cs in ['sate','common']:
        for label in [typestr]:
            for withoccl in [True,False]:
                if withoccl == True:
                    x,y = [],[]
                    for k_iou in sorted(recall_data[cs][label]['withoccl'].keys()):
                        x.append(k_iou)
                        y.append(recall_data[cs][label]['withoccl'][k_iou])
                    l1, = plt.plot(x,y,linestyle = '--',linewidth = 3,marker = 'x')
                    llist.append(l1)
                    print x,y
                else:
                    x,y = [],[]
                    for k_iou in sorted(recall_data[cs][label]['withoutoccl'].keys()):
                        x.append(k_iou)
                        y.append(recall_data[cs][label]['withoutoccl'][k_iou])
                    l2, = plt.plot(x,y,linewidth = 3,marker = 'o')  
                    llist.append(l2)
                    print x,y
    plt.legend(handles = llist,labels = ['sate withoccl','sate withoutoccl','common withoccl','common withoutoccl'],loc = 'best')
    plt.show()
'''
    DESCRIPTION: effective way for reading the last line of a file
    PARAMETERS:
        path: file path
    RETURN:
        last line of the file
'''
def readlastline(filepath):
    with open(filepath,'r') as f:
        offset = -20      
        while True:
            f.seek(offset,2)
            lines = f.readlines()
            if len(lines)>=2:
                return lines[-1]
            offset *=2
if __name__ == '__main__':
    #print parse_eval('./data/sate/eval_final_0.1_.txt')
    #visual_single(get_recall_data(),'addROI')
    #visual_single(get_slide_label_data(),'label')
    visual_iou_0_5(get_recall_data())
