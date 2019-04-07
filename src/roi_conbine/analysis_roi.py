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
    if ratio > iou_threshold: return True
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
def get_overlap_roi():
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]
    path_detect_prefix = 'H:/飒特项目/输出文件/行人检测ROI输出/recall15/'

    iou_thresh = 0.5
    overlap_surfRoIs = []
    overlap_dtsRoIs = []
    overlapRoIs = []
    for set_num in range(21):
        for v_num in range(scut[set_num]+1):  
            detect_dict = parse_detected_box(path_detect_prefix,set_num,v_num,'addROI')
            for framestr in detect_dict.keys():
                frame_dts_rois = [detect_box for detect_box in detect_dict[framestr] if detect_box[4] == -1]
                frame_surf_rois = [detect_box for detect_box in detect_dict[framestr] if detect_box[4] != -1]
                for dts_roi in frame_dts_rois:
                    for surf_roi in frame_surf_rois:
                        if iou_1(dts_roi[:4],surf_roi[:4],iou_thresh):
                            overlap_surfRoIs.append(surf_roi)
                            overlap_dtsRoIs.append(dts_roi)
#                            overlapRoIs.append(overlapRoIs)
#        print('%d'%(set_num))
    return overlap_surfRoIs,overlap_dtsRoIs
def statistic_roi():
    overlap_surfRoIs,overlap_dtsRoIs = get_overlap_roi()
    statistic = [0 for w in range(0,800)]
    print(overlap_surfRoIs)
    for roi in overlap_surfRoIs:
        statistic[roi[2]] = statistic[roi[2]] + 1
    print(statistic)
def write_overlap_roi(outpath,rois):
    out_surf_path = "./../../data/surf_threshold_finetune_result/count_surf.txt"
    outobj = open(out_path,'a')
    print("surf bbox:%d",surf_bbox_num)
    outobj.write("%d %d\n"%(surf_response_threshold,surf_bbox_num))
    outobj.flush()
    outobj.close()
statistic_roi()