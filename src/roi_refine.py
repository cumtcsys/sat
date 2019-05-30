# -*- coding: utf-8 -*-
from src.utils import datatools
from src.utils import evaltools
from src import global_data
'''找最优的a,b,c,d转化参数'''
def findBestRecallABCD(typestr,iou_threshold_list,withoccl = True,ptype='walk_ride',psize = 'near_mid_far'):
    all_ann_bboxes = datatools.parse_whole_dataset_ann_filtered()#所有GT
    all_detect_bboxes = datatools.parse_whole_dataset_detected_box(typestr)#所有检测框
    GTs = len(all_ann_bboxes)
#    alist = [1,2,3]
#    blist = [0.5,1,1.5]
#    clist = [3,4,5]
#    dlist = [5,7,9,11]
#    alist = [0.1*item for item in range(20,30,1)]
#    blist = [0.1*item for item in range(5,15,1)]
#    clist = [0.1*item for item in range(40,60,1)]
#    dlist = [2.17*item for item in clist]
    minloss = len(all_ann_bboxes)
#    alist = [2.5]
#    blist = [1]
#    clist = [5]
#    dlist = [11]2.700000 1.400000 5.800000 12.152000
    alist = [2.7]
    blist = [1.4]
    clist = [5.8]
    dlist = [12.15]
    for a in alist:
        for b in blist:
#            if b<1.2:continue
            for c in clist:
#                if b==1.2 and c<5:continue
                for d in dlist:
                    refined_bboxes = kps2rois(all_detect_bboxes,a,b,c,d)
                    loss = computeloss(all_ann_bboxes,refined_bboxes,iou_threshold_list)
                    if loss< minloss:
                        minloss = loss
                    recall = (GTs - minloss)/GTs
                    print('model = [%lf %lf %lf %lf] loss = %d recall = %lf'%(a,b,c,d,loss,recall))
'''计算头部点转化loss(匹配loss=0,不匹配loss=1)'''
def computeloss(all_ann_bboxes,all_surf_bboxes,iou_threshold_list):
    match_cnt = 0
    for framestr,ann_bboxes in all_ann_bboxes.items():#遍历每一帧GoundTruth
        for ann_box in ann_bboxes:
            for detect_box in all_surf_bboxes[framestr]:
                if evaltools.iou(ann_box[:4],detect_box[:4],iou_threshold_list) :
                    match_cnt = match_cnt + 1
                    break
    GT_cnt = datatools.countBoxes(all_ann_bboxes)
    loss = GT_cnt - match_cnt
    dr = match_cnt / GT_cnt
    print('检测率',dr)
    return loss
def avgBestIou():
    all_ann_bboxes = datatools.parse_whole_dataset_ann_filtered()#所有GT
    all_detect_bboxes = datatools.parse_whole_dataset_detected_box('surf')#所有检测框
    cnt = 0
    avgiou = 0
    for framestr,ann_bboxes in all_ann_bboxes.items():#遍历每一帧GoundTruth
        for ann_box in ann_bboxes:
            for detect_box in all_detect_bboxes[framestr]:
                tempiou = evaltools.iou_1(ann_box[:4],detect_box[:4])
                if tempiou >=0.5:
                    cnt = cnt + 1
                    avgiou += tempiou
    print('avgiou',avgiou/cnt)
'''将单个pd提取到的surfRoI转化成surf特征点'''
def surfRoI2kp(box):
    x,y,w,h = box[0],box[1],box[2],box[3]
    kpr = (w/5)
    kpx = (x + 2.5*kpr) 
    kpy = (y + kpr)
    return (kpx,kpy,kpr)
'''将surf特征点转化成RoI'''
def kp2roi(kp,a,b,c,d):
    kpx,kpy,kpr = kp[0],kp[1],kp[2]
    x = kpx - a*kpr
    y = kpy - b*kpr
    w = c*kpr
    h = d*kpr
    return [x,y,w,h]
'''将所有surfbox调整为新RoI'''
def kps2rois(all_surf_bboxes,a,b,c,d):
    refined_bboxes = {}
    for framestr,bboxes in all_surf_bboxes.items():
        framerebboxes = []
        for box in bboxes:
            kp = surfRoI2kp(box)
            rebox = kp2roi(kp,a,b,c,d)
            framerebboxes.append(rebox)
        refined_bboxes[framestr] = framerebboxes 
    return refined_bboxes
findBestRecallABCD('surf',[0.5,1.5],True)       
#avgBestIou()