# -*- coding: utf-8 -*-
from src.utils import datatools
from src.utils import evaltools
from src import global_data


'''虚警评估'''
def eval_common_fp(typestr,iou_threshold_list,withoccl = True,ptype='walk_ride',psize = 'near_mid_far'):
    all_ann_bboxes = datatools.parse_whole_dataset_ann_filtered()#所有GT
    all_detect_bboxes = datatools.parse_whole_dataset_detected_box(typestr)#所有检测框
    filteredBoxes = filterAnnBoxes(all_ann_bboxes,typestr,withoccl,ptype,psize)#根据行人类别，尺度过滤
    match_cnt = 0
    for framestr,ann_bboxes in filteredBoxes.items():#遍历每一帧GoundTruth
        for detect_box in all_detect_bboxes[framestr]:
            for ann_box in ann_bboxes:
                if evaltools.iou(ann_box[:4],detect_box[:4],iou_threshold_list) :
                    match_cnt = match_cnt + 1
                    break
    detect_cnt = datatools.countBoxes(all_detect_bboxes)
    fp = detect_cnt - match_cnt
    return fp
'''过滤GroundTruth'''
def filterAnnBoxes(annbboxes,typestr,withoccl = True,ptype='walk_ride',psize = 'near_mid_far'):
    filteredBoxes = {}
    for framestr,frameboxes in annbboxes.items():#处理所有帧
        framefilteredBox = []
        for box in frameboxes:#处理单帧所有bbox
            flag = True
            if ptype == 'walk' and box[6] != 1:#box[6]=1表示walk_person
                flag = False
            if ptype == 'ride' and box[6] != 0:#box[6]=0表示ride_person
                flag = False
            if withoccl == False and box[5] == 1:#过滤遮挡
                flag = False
            if psize == 'near' and box[3] < 90:#过滤不合要求的尺度
                flag = False
            if psize == 'mid' and (box[3] >= 90 or box[3] <48):
                flag = False
            if psize == 'far' and box[3] > 48:#过滤不合要求的尺度
                flag = False
            if flag == True:
                framefilteredBox.append(box)
        filteredBoxes[framestr] = framefilteredBox
    return filteredBoxes


'''common标准评估'''
def eval_common(typestr,iou_threshold_list,withoccl = True,ptype='walk_ride',psize = 'near_mid_far'):
    all_ann_bboxes = datatools.parse_whole_dataset_ann_filtered()#所有GT
    all_detect_bboxes = datatools.parse_whole_dataset_detected_box(typestr)#所有检测框
    fAnnBoxes = filterAnnBoxes(all_ann_bboxes,typestr,withoccl,ptype,psize)
    match_cnt = 0
    for framestr,ann_bboxes in fAnnBoxes.items():#遍历每一帧GoundTruth
        for ann_box in ann_bboxes:
            for detect_box in all_detect_bboxes[framestr]:
#                if detect_box[4]>0:continue#只评估segs
#                if detect_box[4]<0:continue#只评估SURF
                if evaltools.iou(ann_box[:4],detect_box[:4],iou_threshold_list) :
                    match_cnt = match_cnt + 1
                    break
    detect_cnt = datatools.countBoxes(all_detect_bboxes)
    GT_cnt = datatools.countBoxes(fAnnBoxes)
    fp = eval_common_fp(typestr,iou_threshold_list,withoccl,ptype,psize)
    print( ('匹配成功:%d GT总数:%d 输出框总数:%d 检测率：%lf 虚警：%d iou = [%.2f,%.2f]') % (match_cnt,GT_cnt,detect_cnt,match_cnt/GT_cnt,fp,iou_threshold_list[0],iou_threshold_list[1]))
'''
    DESCRIPTION: eval the detected bbox and write the result to file
                 adapt the common styles
    PARAM:
        typestr: label addROI,positionFilter,sizeFilter,recycleprocess,simpleTrace,haarlike,hogLBP
        iou_threshold: iou threshold,usually equals 0.5
        withoccl: if need exclude occl object,set withoccl = False
'''
def write_eval_common(typestr,iou_threshold_list,withoccl = True):
    print('common')
    total_ann_box = 0# how many boxes(filtered) scut have
    matched_box = 0#how many boxes iou greater than iou_threshold
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]#scut[0] = 2 means set00/ have V000.txt,V001.txt,V002.txt
    total_detect_num,temp_detect_num = 0,0
    #destiguish withoccl and not
    if withoccl == True:
        eval_path = './../../data/eval/common/eval_'+'withoccl_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_''.txt'
    else: eval_path = './../../data/eval/common/eval_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_'+'.txt'
    fobj = open(eval_path,'w')
    cnt = 0
    #process set by set
    for set_num in range(21):
        for v_num in range(scut[set_num]+1):    
            tempcnt = 0
            ann_dict = datatools.parse_ann_filtered(set_num,v_num)
            detect_dict = datatools.parse_detected_box(set_num,v_num,typestr)
            temp_total_box = 0
            temp_matched_box = 0
            temp_detect_num = 0
            #process all frames
            for framestr,box_ann_list in ann_dict.items():  
                #process all box in a frame
                for ann_box in box_ann_list:
                    if withoccl == False and ann_box[5] == 1:
                        continue
                    total_ann_box = total_ann_box + 1
                    temp_total_box = temp_total_box + 1
                    for detect_box in detect_dict[framestr]:
                        if evaltools.iou(ann_box[:4],detect_box[:4],iou_threshold_list) :
                            matched_box = matched_box + 1
                            temp_matched_box = temp_matched_box + 1
                            break
                tempcnt = tempcnt+len(box_ann_list)
                temp_detect_num = temp_detect_num + len(detect_dict[framestr])
            cnt = cnt + tempcnt    
            total_detect_num = total_detect_num + temp_detect_num
            if temp_total_box==0:fobj.writelines('%02d%03d[%d %d %f] %d\n' % (set_num,v_num,temp_matched_box,temp_total_box,0,temp_detect_num))
            else:fobj.writelines('%02d%03d[%d %d %f] %d\n' % (set_num,v_num,temp_matched_box,temp_total_box,temp_matched_box*1.0/temp_total_box,temp_detect_num))
    if total_ann_box == 0:fobj.writelines('total:[%d %d %f] %d' % (matched_box,total_ann_box,0,total_detect_num))        
    else:fobj.writelines('total:[%d %d %f] %d' % (matched_box,total_ann_box,matched_box*1./total_ann_box,total_detect_num))        
    print(cnt)
    resstr = 'common'
    if withoccl == True:resstr = resstr + '带遮挡'
    else:resstr = resstr + '忽略遮挡'
    print(resstr)
    print( ('匹配成功:%d GT总数:%d 输出框总数:%d iou = [%.2f,%.2f]') % (matched_box,total_ann_box,total_detect_num,iou_threshold_list[0],iou_threshold_list[1]))
    if total_ann_box == 0:
        print(0)
    else:
        print(matched_box*1./total_ann_box)
    fobj.flush()
    fobj.close()
'''
    DESCRIPTION: eval the detected bbox and write the result to file
                 adapt the sate styles
    PARAM:
        typestr: label addROI,positionFilter,sizeFilter,recycleprocess,simpleTrace,haarlike,hogLBP
        iou_threshold: iou threshold,usually equals 0.5
        withoccl: if need exclude occl object,set withoccl = False
'''
def write_eval_sate(typestr,iou_threshold_list ,withoccl = True):
    total_id = 0# how many boxes(filtered) scut have
    matched_id = 0#how many boxes iou greater than iou_threshold
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]#scut[0] = 2 means set00/ have V000.txt,V001.txt,V002.txt
    #destiguish withoccl and not
    if withoccl == True:
        eval_path = './../../data/eval/sate/eval_'+'withoccl_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_'+'.txt'
    else: eval_path = './../../data/eval/sate/eval_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_'+'.txt'
    fobj = open(eval_path,'w')
    ann_id_dict = {}
    detect_id_dict = {}
    total_detect_num,temp_detect_num = 0,0
    total_matched_num,temp_matched_num = 0,0
    #process set by set
    for set_num in range(21):
        #process v by v
        for v_num in range(scut[set_num]+1):  
            ann_dict = datatools.parse_ann_filtered(set_num,v_num)
            for framestr,box_ann_list in ann_dict.items():     
                #process all box in a frame
                for ann_box in box_ann_list:
                    if withoccl == False and ann_box[5] == 1:
                        continue
                    ann_key = '%02d%03d%05d' % (set_num,v_num,ann_box[4])
                    detect_key = '%02d%03d%05d' % (set_num,v_num,ann_box[4])
                    if ann_key not in ann_id_dict:ann_id_dict[ann_key] = 0
                    else:ann_id_dict[ann_key] = ann_id_dict[ann_key]+1
                    if detect_key not in detect_id_dict:detect_id_dict[detect_key] = 0
            
                
    print(len(ann_id_dict),len(detect_id_dict))            
    #process set by set
    for set_num in range(21):
        #process v by v
        for v_num in range(scut[set_num]+1):  
            ann_dict = datatools.parse_ann_filtered(set_num,v_num)
            detect_dict = datatools.parse_detected_box(set_num,v_num,typestr)
            temp_detect_num = 0
            temp_matched_num = 0
            #process all frames
            for framestr,box_ann_list in ann_dict.items():     
                #process all box in a frame
                for ann_box in box_ann_list:
                    if withoccl == False and ann_box[5] == 1:
                        continue
                    first_match_flag = False;
                    for detect_box in detect_dict[framestr]:
                        if evaltools.iou(ann_box[:4],detect_box[:4],iou_threshold_list):
                            if first_match_flag == False:
                                first_match_flag == True
                                detect_key = '%02d%03d%05d' % (set_num,v_num,ann_box[4])
                                detect_id_dict[detect_key] = detect_id_dict[detect_key] + 1
                            temp_matched_num = temp_matched_num + 1
            
                # counter how many box in detect file
                temp_detect_num = temp_detect_num + len(detect_dict[framestr])
            total_matched_num = total_matched_num + temp_matched_num
            temp_total_id = 0
            temp_matched_id = 0
            total_detect_num = total_detect_num + temp_detect_num
            for k,v in detect_id_dict.items():
                if k[:5] == ('%02d%03d' % (set_num,v_num)) : temp_total_id = temp_total_id + 1
                if k[:5] == ('%02d%03d' % (set_num,v_num)) and v != 0: temp_matched_id = temp_matched_id + 1
            if temp_total_id==0:fobj.writelines('%02d%03d[%d %d %f] %d\n' % (set_num,v_num,temp_matched_id,temp_total_id,0,temp_detect_num))
            else:fobj.writelines('%02d%03d[%d %d %f] %d\n' % (set_num,v_num,temp_matched_id,temp_total_id,temp_matched_id*1.0/temp_total_id,temp_detect_num))
            total_id = total_id + temp_total_id
            matched_id = matched_id + temp_matched_id
    resstr = 'sate'
    if withoccl == True:resstr = resstr + '带遮挡'
    else:resstr = resstr + '忽略遮挡'
    print(resstr)
    print( ('匹配行人数:%d 行人总数:%d 输出框总数:%d 匹配框总数:%d iou=[%.2f,%.2f]') % (matched_id,total_id,total_detect_num,total_matched_num,iou_threshold_list[0],iou_threshold_list[1]))
    if total_id == 0:
        fobj.writelines('total:[%d %d %f] %d' % (matched_id,total_id,0,total_detect_num))    
        print(0)
    else:
        fobj.writelines('total:[%d %d %f] %d' % (matched_id,total_id,matched_id*1.0/total_id,total_detect_num))        
        print(matched_id*1.0/total_id)
    fobj.flush()
    fobj.close()

if __name__ == '__main__':

#    eval_common(typestr,iou_threshold_list,withoccl = True,ptype='walk_ride',psize = 'near_mid_far')
    pstr = 'final'
    iou_list = [0.5,1.5]
#    eval_common(pstr,iou_list,withoccl =True,ptype='walk',psize = 'near')
#    eval_common('recycleprocess',iou_list,withoccl =True,psize = 'near')
#    eval_common('recycleprocess',iou_list,withoccl =True,psize = 'mid')
#    eval_common('recycleprocess',iou_list,withoccl =True,psize = 'far')
    
#    eval_common(pstr,iou_list,withoccl =True)
#    eval_common(pstr,iou_list,withoccl =True,psize = 'near')
#    eval_common(pstr,iou_list,withoccl =True,psize = 'mid')
#    eval_common(pstr,iou_list,withoccl =True,psize = 'far')
    
    
    
#    
    
#    eval_common('final',iou_list,withoccl =True)
    write_eval_common(pstr,iou_list,True)
    write_eval_sate(pstr,iou_list,True)
#    write_eval_common(pstr,iou_list,False)
#    write_eval_sate(pstr,iou_list,True)
#    write_eval_sate(pstr,iou_list,False)
#    eval_common_fp(pstr,iou_list,True)
#    eval_common_fp(pstr,iou_list,False)