# -*- coding: utf-8 -*-
import numpy as np
from src.utils import datatools
from src.utils import evaltools
from src import global_data

'''
    DESCRIPTION: parse detected_box file
    PARAMETERS:
        dbpath: where the detected_box file located
        set_num: set00->set20
        v_num: such as set00 -> v000.txt,v001.txt,v002.txt 
        typestr: label,addROI,simpleTrace,sizeFilter,positionFilter,recycleProcess,hogLBP,haarLike,final
'''
def parse_detected_box(dbpath,set_num,v_num,typestr):
    parsed_box_dict = {}
    path_ann_prefix = dbpath
    path_ann = path_ann_prefix + ('set%02d/V%03d_%s.txt' % (set_num,v_num,typestr))
    fobj = open(path_ann,'r')
    for line in fobj.readlines():
        box_start_index = line.find('[')
        box_end_index = line.rfind(']')
        framestr = line[:box_start_index].strip()
        # 0100000000[]
        if box_start_index + 1 == box_end_index:
            parsed_box_dict[framestr] = []
            continue        
        # box_end_index -2 remove '[12,13,40,80; ]'
        recstr = line[box_start_index+1:box_end_index -2]
        rec_list = [rec_4.split(' ') for rec_4 in recstr.split('; ')]
        for ii in range(len(rec_list)):
            for jj in range(len(rec_list[ii])):
                rec_list[ii][jj] = int(rec_list[ii][jj])
        parsed_box_dict[framestr] = rec_list
    return parsed_box_dict
'''
    计算各阈值对应的surf框数量
'''
def cal_surf_bbox_num(surf_response_threshold = 126):
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]
    path_detect_prefix = 'H:/飒特项目/输出文件/行人检测ROI输出/recall13/'
    out_path = "./../../data/surf_threshold_finetune_result/count_surf.txt"
    outobj = open(out_path,'a')
    surf_bbox_num = 0
    for set_num in range(21):
        for v_num in range(scut[set_num]+1):  
            detect_dict = parse_detected_box(path_detect_prefix,set_num,v_num,'surf')
            for framestr in detect_dict.keys():
                for detect_box in detect_dict[framestr]:
                    if detect_box[4]>=surf_response_threshold:
                        surf_bbox_num +=1
#        print("set_num:%d",set_num)
    print("surf bbox:%d",surf_bbox_num)
    outobj.write("%d %d\n"%(surf_response_threshold,surf_bbox_num))
    outobj.flush()
    outobj.close()
    return surf_bbox_num
'''
    计算SCUT总帧数、平均每帧RoI数
'''
def cal_frames_RoIs():
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]
    path_detect_prefix = 'H:/飒特项目/输出文件/行人检测ROI输出/recall25/'
    roi_nums = [0 for i in range(101)]
    frame_nums = 0
    thresh = 126
    for set_num in range(21):
        for v_num in range(scut[set_num]+1):  
            detect_dict = parse_detected_box(path_detect_prefix,set_num,v_num,'recycleProcess')
            frame_nums +=len(detect_dict)
            for framestr,detected_box in detect_dict.items():
                nums_per_frame = 0
                for box in detected_box:
                    nums_per_frame += 1
#                    if box[4] == -1 :nums_per_frame += 1
#                    if box[4]>= thresh:nums_per_frame += 1
                index = nums_per_frame
                if index > 100: 
                    roi_nums[100] = roi_nums[100] + 1
                    continue
                roi_nums[index] = roi_nums[index] + 1
            
    print('scut总帧数:%d'%(frame_nums))
    print('单帧RoIs数量统计：',roi_nums)


'''
    DESCRIPTION: eval the detected bbox and write the result to file
                 adapt the common styles
    PARAM:
        typestr: label addROI,positionFilter,sizeFilter,recycleprocess,simpleTrace,haarlike,hogLBP
        iou_threshold: iou threshold,usually equals 0.5
        withoccl: if need exclude occl object,set withoccl = False
'''
def write_eval_common(typestr,iou_threshold_list,withoccl,surf_response_threshold):
    total_ann_box = 0# how many boxes(filtered) scut have
    matched_box = 0#how many boxes iou greater than iou_threshold
    path_detect_prefix = 'H:/飒特项目/输出文件/行人检测ROI输出/recall18/'#detect bbox path
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]#scut[0] = 2 means set00/ have V000.txt,V001.txt,V002.txt
    total_detect_num,temp_detect_num = 0,0
    #destiguish withoccl and not
    if withoccl == True:
        eval_path = './../../data/surf_threshold_finetune_result/eval_'+'withoccl_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_''.txt'
    else: eval_path = './../../data/surf_threshold_finetune_result/eval_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_'+'.txt'
    fobj = open(eval_path,'a')
    #process set by set
    for set_num in range(21):
        #process v by v
        for v_num in range(scut[set_num]+1):  
#            if v_num != 10:continue
            ann_dict = datatools.parse_ann_filtered(set_num,v_num)
            detect_dict = parse_detected_box(path_detect_prefix,set_num,v_num,typestr)
            temp_total_box,temp_matched_box,temp_detect_num = 0,0,0
            #process all frames
            for framestr,box_ann_list in ann_dict.items():  
                count_detect = 0#统计单帧总检测框数量
                #process all box in a frame
                for ann_box in box_ann_list:
#                    if ann_box[3] >= 90 or ann_box[3] <48:continue 
#                    if ann_box[3] < 90:continue
                    if ann_box[3] > 48:continue
                    if withoccl == False and ann_box[5] == 1:
                        continue
                    total_ann_box = total_ann_box + 1
                    temp_total_box = temp_total_box + 1
                    
                    for detect_box in detect_dict[framestr]:
                        #如果surf响应值阈值小于设定的阈值则将该surf提出的RoI过滤掉,-1代表阈值分割提出的RoI,
                        #双阈值分割贡献
#                        if detect_box[4] == -1:continue
                        #头部贡献
#                        if detect_box[4]<surf_response_threshold:continue
                        #总召回率，忽略surf得分低的RoI
                        if detect_box[4]<surf_response_threshold and detect_box[4]!= -1 :continue
                        if evaltools.iou(ann_box[:4],detect_box[:4],iou_threshold_list) :
                            matched_box = matched_box + 1
                            temp_matched_box = temp_matched_box + 1
                            break
#                count_detect +=len(detect_dict[framestr] )
#                count_detect += len([box for box in detect_dict[framestr] if (box[4]>surf_response_threshold or box[4]== -1)])
#                count_detect += len([box for box in detect_dict[framestr] if box[4] == -1])
#                temp_detect_num = temp_detect_num + count_detect
                temp_detect_num = temp_detect_num + len(detect_dict[framestr])
            total_detect_num = total_detect_num + temp_detect_num
    if total_ann_box == 0:fobj.writelines('surf_response_threshold = %d-- total:[%d %d %f] %d' % (surf_response_threshold,matched_box,total_ann_box,0,total_detect_num))        
    else:fobj.writelines('surf_response_threshold = %d total:[%d %d %f] %d\n' % (surf_response_threshold,matched_box,total_ann_box,matched_box*1./total_ann_box,total_detect_num))        
    print(('surf_threshold = %d,matech:%d total_ann:%d total_surf__detect:%d iou_threshold[ %.2f , %.2f]') % (surf_response_threshold,matched_box,total_ann_box,total_detect_num,iou_threshold_list[0],iou_threshold_list[1]))
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
def write_eval_sate(typestr,iou_threshold_list ,withoccl,surf_response_threshold):
    total_id = 0# how many boxes(filtered) scut have
    matched_id = 0#how many boxes iou greater than iou_threshold
    path_detect_prefix = 'H:/飒特项目/输出文件/行人检测ROI输出/recall21/'#detect bbox path
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]#scut[0] = 2 means set00/ have V000.txt,V001.txt,V002.txt
    #destiguish withoccl and not
    if withoccl == True:
        eval_path = './../../data/surf_threshold_finetune_result/eval_'+'withoccl_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_'+'.txt'
    else: eval_path = './../../data/surf_threshold_finetune_result/eval_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_'+'.txt'
    fobj = open(eval_path,'a')
    ann_id_dict = {}
    detect_id_dict = {}
    total_detect_num,temp_detect_num = 0,0
    temp_matched_box,total_matched_box = 0,0
    total_matched_num,temp_matched_num = 0,0
    #process set by set
    for set_num in range(21):
        #process v by v
        for v_num in range(scut[set_num]+1):  
            ann_dict = parse_ann_filtered(path_ann_prefix,set_num,v_num)
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
            detect_dict = parse_detected_box(path_detect_prefix,set_num,v_num,typestr)
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
                        #双阈值分割贡献
#                        if detect_box[4] != -1:continue
                        #头部贡献
#                        if detect_box[4]<surf_response_threshold:continue
                        #总召回率
                        if detect_box[4]<surf_response_threshold and detect_box[4]!= -1 :continue
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
            total_id = total_id + temp_total_id
            matched_id = matched_id + temp_matched_id
    print(('surf_response_threshold=%d matech:%d total_ann:%d total_surf__detect:%d total_matched_num:%d iou_threshold[0] = %.2f iou_threshold[1] = %.2f') % (surf_response_threshold,matched_id,total_id,total_detect_num,total_matched_num,iou_threshold_list[0],iou_threshold_list[1]))
    if total_id == 0:
        fobj.writelines('surf_response_threshold = %d total:[%d %d %f] %d\n' % (surf_response_threshold,matched_id,total_id,0,total_detect_num))    
        print(0)
    else:
        fobj.writelines('surf_response_threshold = %d total:[%d %d %f] %d\n' % (surf_response_threshold,matched_id,total_id,matched_id*1.0/total_id,total_detect_num))        
        print(matched_id*1.0/total_id)
    fobj.flush()
    fobj.close()

if __name__ == '__main__':

#    write_eval_common('recycleProcess',[0.5,1.5],False,-1)
#    write_eval_common('addROI',[0.5,1.5],True,0)
#    write_eval_sate('addROI',[0.5,1.5],True,0)
#    write_eval_common('addROI',[0.5,1.5],False,0)
#    write_eval_sate('addROI',[0.5,1.5],False,0)
    cal_frames_RoIs()

