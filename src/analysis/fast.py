# -*- coding: utf-8 -*-
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
def fastFilter(annbox,kpboxes,fastthresh,numsThresh):
    num = 0
    for kpbox in kpboxes:
        if fastKPinAnn(annbox,kpbox,fastthresh):
            num+=1
        if num >= numsThresh:
            return True
#    print('num = %d\n'%(num))
    return False
def fastKPinAnn(annbox,kpbox,fastthresh):
    kpx,kpy,fastScore = kpbox[0],kpbox[1],kpbox[4]
    annx,anny,annw,annh = annbox[0],annbox[1],annbox[2],annbox[3]
    if kpx>=annx and kpx<=annx+annw and kpy>=anny and kpy<=anny+annh and fastScore>=fastthresh:return True
    return False
'''
    DESCRIPTION: eval the detected bbox and write the result to file
                 adapt the common styles
    PARAM:
        typestr: label addROI,positionFilter,sizeFilter,recycleprocess,simpleTrace,haarlike,hogLBP
        iou_threshold: iou threshold,usually equals 0.5
        withoccl: if need exclude occl object,set withoccl = False
'''
def statistic_fast(typestr,iou_threshold_list,fastthresh,numthresh,withoccl = True):
    total_ann_box = 0# how many boxes(filtered) scut have
    matched_box = 0#how many boxes iou greater than iou_threshold
    path_ann_prefix = global_data.ann_filter_path#parsed scut ann path
    path_detect_prefix = 'H:/飒特项目/输出文件/行人检测ROI输出/recall27/'#detect bbox path
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]#scut[0] = 2 means set00/ have V000.txt,V001.txt,V002.txt
    total_detect_num,temp_detect_num = 0,0
    temp_fast_match_num = 0
    for set_num in range(21):
        for v_num in range(scut[set_num]+1):  
            ann_dict = datatools.parse_ann_filtered(set_num,v_num)
            addROI_dict = parse_detected_box('H:/飒特项目/输出文件/行人检测ROI输出/recall26/',set_num,v_num,'addROI')
            detect_dict = parse_detected_box(path_detect_prefix,set_num,v_num,typestr)
            temp_total_box = 0
            temp_matched_box = 0
            temp_detect_num = 0
            #process all frames
            for framestr,box_ann_list in ann_dict.items():  
                #process all box in a frame
                for ann_box in box_ann_list:
#                    if ann_box[3] >= 90 or ann_box[3] <48:continue 
#                    if ann_box[3] < 90:continue
#                    if ann_box[3] > 48:continue
                    if withoccl == False and ann_box[5] == 1:
                        continue
                    total_ann_box = total_ann_box + 1
                    temp_total_box = temp_total_box + 1
                    for detect_box in addROI_dict[framestr]:#RoI提取的框
                        if not evaltools.iou(ann_box[:4],detect_box[:4],iou_threshold_list):continue
#                        temp_fast_match_num+=1#统计有多少是与
                        if fastFilter(detect_box,detect_dict[framestr],fastthresh,numthresh):
                            matched_box = matched_box + 1
                            temp_matched_box = temp_matched_box + 1
                            break
#                    for detect_box in detect_dict[framestr]:
#                        if fastKPinAnn(ann_box[:4],detect_box[:5],fastthresh) :
#                            matched_box = matched_box + 1
#                            temp_matched_box = temp_matched_box + 1
#                            break
                
                temp_detect_num = temp_detect_num + len(detect_dict[framestr])
            total_detect_num = total_detect_num + temp_detect_num
    print( ('matech:%d total_ann:%d total_detect:%d iou_threshold[0] = %.2f iou_threshold[1] = %.2f') % (matched_box,total_ann_box,total_detect_num,iou_threshold_list[0],iou_threshold_list[1]))
    if total_ann_box == 0:
        print(0)
        return 0
    else:
        print(matched_box*1./total_ann_box)
        return matched_box*1./total_ann_box 
if __name__ == '__main__':
    outfile = open('./fast.txt','w')
    outfile.writelines('[')
    for i in range(30,51,5):
        for numthresh in [1]:
            recall = statistic_fast('fastPoint',[0.5,1.5],i,numthresh,True)
            outfile.writelines('['+str(numthresh)+','+str(i)+','+str(recall)+']'+',')
    outfile.writelines(']')
    outfile.flush()
    outfile.close()
