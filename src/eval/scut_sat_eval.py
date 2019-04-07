# -*- coding: utf-8 -*-
'''
    DESCRIPTION: compute the two input rectangle IOU(area(ann&&detected)/(area(ann) || area(detected))))
    PARAM:
        rec1: x,y,w,h
        rec2: x,y,w,h      
'''
def iou_1(rec1,rec2,iou_threshold):
    x1,y1,width1,height1 = rec1[0],rec1[1],rec1[2],rec1[3]
    x2,y2,width2,height2 = rec2[0],rec2[1],rec2[2],rec2[3]

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0:
        ratio = 0 # iou = 0 
    else:
        Area = width*height; # 
        Area1 = width1*height1; 
        Area2 = width2*height2;
        ratio = Area*1./(Area1+Area2-Area);
    if ratio >= iou_threshold: return True
    return False
'''
    DESCRIPTION: compute the two input rectangle IOU(area(ann&&detected)/area(ann))
    PARAM:
        rec1: x,y,w,h (ann)
        rec2: x,y,w,h (detected)  
'''
def iou_2(rec1,rec2,iou_threshold):
    x1,y1,width1,height1 = rec1[0],rec1[1],rec1[2],rec1[3]
    x2,y2,width2,height2 = rec2[0],rec2[1],rec2[2],rec2[3]

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0:
        ratio = 0 # iou = 0 
    else:
        Area = width*height; # 
        Area1 = width1*height1; 
        ratio = Area*1./(Area1);
    # return IOU
    if ratio >= iou_threshold: return True
    return False
def iou(ann,detected,ioulist):
    if iou_1(ann,detected,ioulist[0]) or  iou_2(ann,detected,ioulist[1]):return True
    return False
'''
    DESCRIPTION: parse ann_filtered file
    PARAMETERS:
        dbpath: where the ann_filtered file located
        set_num: set00->set20
        v_num: such as set00 -> v000.txt,v001.txt,v002.txt       
'''
def parse_ann_filtered(dbpath,set_num,v_num):
    parsed_box_dict = {}
    path_ann_prefix = dbpath
    path_ann = path_ann_prefix + ('set%02d/V%03d_ann.txt' % (set_num,v_num))
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
    DESCRIPTION: eval the detected bbox and write the result to file
                 adapt the common styles
    PARAM:
        typestr: label addROI,positionFilter,sizeFilter,recycleprocess,simpleTrace,haarlike,hogLBP
        iou_threshold: iou threshold,usually equals 0.5
        withoccl: if need exclude occl object,set withoccl = False
'''
def write_eval_common(typestr,iou_threshold_list,withoccl = True):
    total_ann_box = 0# how many boxes(filtered) scut have
    matched_box = 0#how many boxes iou greater than iou_threshold
    path_ann_prefix = './data/ann_filter/'#parsed scut ann path
    path_detect_prefix = 'H:/飒特项目/输出文件/行人检测ROI输出/recall23/'#detect bbox path
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]#scut[0] = 2 means set00/ have V000.txt,V001.txt,V002.txt
    total_detect_num,temp_detect_num = 0,0
    #destiguish withoccl and not
    if withoccl == True:
        eval_path = './data/common/eval_'+'withoccl_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_''.txt'
    else: eval_path = './data/common/eval_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_'+'.txt'
    fobj = open(eval_path,'w')
    #process set by set
    for set_num in range(21):
#        if set_num != 5:continue
        #process v by v
        for v_num in range(scut[set_num]+1):  
#            if v_num != 10:continue
            ann_dict = parse_ann_filtered(path_ann_prefix,set_num,v_num)
            detect_dict = parse_detected_box(path_detect_prefix,set_num,v_num,typestr)
            temp_total_box = 0
            temp_matched_box = 0
            temp_detect_num = 0
            
            #process all frames
            for framestr,box_ann_list in ann_dict.items():  
#                print(framestr)

                #process all box in a frame
                for ann_box in box_ann_list:
#                    if ann_box[3] >= 90 or ann_box[3] <48:continue 
                    if ann_box[3] < 90:continue
#                    if ann_box[3] > 48:continue
                    if withoccl == False and ann_box[5] == 1:
                        continue
                    total_ann_box = total_ann_box + 1
                    temp_total_box = temp_total_box + 1
                    for detect_box in detect_dict[framestr]:
                        if iou(ann_box[:4],detect_box[:4],iou_threshold_list) :
                            matched_box = matched_box + 1
                            temp_matched_box = temp_matched_box + 1
                            break

                
                temp_detect_num = temp_detect_num + len(detect_dict[framestr])
            total_detect_num = total_detect_num + temp_detect_num
            if temp_total_box==0:fobj.writelines('%02d%03d[%d %d %f] %d\n' % (set_num,v_num,temp_matched_box,temp_total_box,0,temp_detect_num))
            else:fobj.writelines('%02d%03d[%d %d %f] %d\n' % (set_num,v_num,temp_matched_box,temp_total_box,temp_matched_box*1.0/temp_total_box,temp_detect_num))
    if total_ann_box == 0:fobj.writelines('total:[%d %d %f] %d' % (matched_box,total_ann_box,0,total_detect_num))        
    else:fobj.writelines('total:[%d %d %f] %d' % (matched_box,total_ann_box,matched_box*1./total_ann_box,total_detect_num))        
    print( ('matech:%d total_ann:%d total_detect:%d iou_threshold[0] = %.2f iou_threshold[1] = %.2f') % (matched_box,total_ann_box,total_detect_num,iou_threshold_list[0],iou_threshold_list[1]))
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
    path_ann_prefix = './data/ann_filter/'#parsed scut ann path
    path_detect_prefix = 'H:/飒特项目/输出文件/行人检测ROI输出/recall23/'#detect bbox path
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]#scut[0] = 2 means set00/ have V000.txt,V001.txt,V002.txt
    #destiguish withoccl and not
    if withoccl == True:
        eval_path = './data/sate/eval_'+'withoccl_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_'+'.txt'
    else: eval_path = './data/sate/eval_'+typestr+'_'+str(iou_threshold_list[0])+'_'+str(iou_threshold_list[1])+'_'+'.txt'
    fobj = open(eval_path,'w')
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
            ann_dict = parse_ann_filtered(path_ann_prefix,set_num,v_num)
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
                        if iou(ann_box[:4],detect_box[:4],iou_threshold_list):
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
    print( ('matech:%d total_ann:%d total_detect:%d total_matched_num:%d iou_threshold[0] = %.2f iou_threshold[1] = %.2f') % (matched_id,total_id,total_detect_num,total_matched_num,iou_threshold_list[0],iou_threshold_list[1]))
    if total_id == 0:
        fobj.writelines('total:[%d %d %f] %d' % (matched_id,total_id,0,total_detect_num))    
        print(0)
    else:
        fobj.writelines('total:[%d %d %f] %d' % (matched_id,total_id,matched_id*1.0/total_id,total_detect_num))        
        print(matched_id*1.0/total_id)
    fobj.flush()
    fobj.close()
#write_eval_common('addROI',0.3)
#write_eval_common('addROI',0.5,True)

if __name__ == '__main__':
    #write_eval_common('final',0.5,True)
    #write_eval_common('final',0.3,True)
#    write_eval_sate('final',0.3,True)
#    for ious in [0.3,0.4,0.5,0.6,0.7]:
#        for label in ['label','addROI','simpleTrace','sizeFilter','positionFilter','recycleProcess','haarLike','hogLBP','final']:
#            for withoccl in [True,False]:
#                write_eval_sate(label,[ious,0.8],withoccl)
#                write_eval_common(label,[ious,0.8],withoccl)
    
#    write_eval_common('addROI',[0.5,1.5],True)
#    write_eval_common('addROI',[0.5,1.5],False)
#    
#    write_eval_common('final',[0.5,1.5],True)
#    write_eval_common('final',[0.5,1.5],False)
#    
    write_eval_common('addROI',[0.5,1.5],True)
#    write_eval_common('recycleProcess',[0.5,1.5],True)
#    write_eval_common('final',[0.5,1.5],False)
    
    
#    write_eval_sate('addROI',[0.5,1.5],True)
#    write_eval_sate('addROI',[0.5,1.5],False)
    
#    write_eval_sate('final',[0.5,1],True)
#    write_eval_sate('final',[0.5,1],False)
    
#    write_eval_sate('surf',[0.5,1.5],True)
#    write_eval_sate('surf',[0.5,1.5],False)
    
    
    
    
    
    
    
#    write_eval_sate('final',[0.5,0.6],False)
#    write_eval_common('final',[0.5,0.8],False)
#    write_eval_sate('final',[0.5,0.8],False)
#    write_eval_common('final',[0.5,0.9],False)
#    write_eval_sate('final',[0.5,0.9],False)
#    write_eval_common('final',[0.5,1.0],False)
#    write_eval_sate('final',[0.5,1.0],False)
