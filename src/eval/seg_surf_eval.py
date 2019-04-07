import glob
import scut_sat_eval as sateval
# -*- coding: utf-8 -*-
def parse_detected_box():
    detectpathes = getDetectPath()
    detectinfo = {}
    for setnum in range(len(detectpathes)):#处理所有set
        for dvpath in detectpathes[setnum]:#处理set中所有特征点文件
            print(dvpath)
            dfile = open(dvpath,'r')                                
            for line in dfile.readlines():
                box_start_index = line.find('[')
                box_end_index = line.rfind(']')
                framestr = line[:box_start_index].strip()
                # 0100000000[]
                if box_start_index + 1 == box_end_index:
                    detectinfo[framestr] = []
                    continue        
                # box_end_index -2 remove '[12,13,40,80; ]'
                recstr = line[box_start_index+1:box_end_index -2]
                rec_list = [rec_4.split(' ') for rec_4 in recstr.split('; ')]
                for ii in range(len(rec_list)):
                    for jj in range(len(rec_list[ii])):
                        rec_list[ii][jj] = int(rec_list[ii][jj])
                detectinfo[framestr] = rec_list
    print(detectinfo.keys())
    return detectinfo
def getSURFKPPath():
    pathes = []
    prepath = 'E:\\workspace\\spyder_workspace\\sat\\data\\surf'
    for setnum in range(21):
        setpath = '%s\\set%02d\\*.txt'%(prepath,setnum)
        pathes.append(glob.glob(setpath))
    return pathes
def getDetectPath():
    pathes = []
    prepath = 'H:\\human_dataset\\scut_result\\'
    for setnum in range(21):
        setpath = '%s\\set%02d\\*label.txt'%(prepath,setnum)
        pathes.append(glob.glob(setpath))
    return pathes
'''
    Description: 解析所有特征点文件，获取特征中心点，半径信息,以字典形式返回
    return: 包含所有set,所有特征点位置以及尺度信息
'''
def parseKPFile():
    kppathes = getSURFKPPath()
    kpsinfo = {}
    for setnum in range(len(kppathes)):#处理所有set
        for kpvpath in kppathes[setnum]:#处理set中所有特征点文件
            kpfile = open(kpvpath,'r')
            indrs = kpvpath.rfind('\\')
            indp = kpvpath.rfind('.')
            vstr = kpvpath[indrs+2:indp]#获取不带后缀的视频名，并去掉文件名首字母V
            for line in kpfile.readlines():
                #line示例 00004:527 264 22;81 26 22;22 34 18;277 245 17;71 25 22;484 261 18;428 266 34;
                line = line.strip()
                framestr = line[:5]#帧号
                key = str(setnum) + vstr + framestr
                if len(line) == 6:#当前帧没有特征点，
                    kpsinfo[key] = []
                else:
                    kpstrs = line[6:-1]#去掉最后分号
                    kps = kpstrs.split(';')
                    kplist = []
                    for kp in kps:
                        kpstr = kp.split(' ')
                        x,y,r = int(kpstr[0]),int(kpstr[1]),int(kpstr[2])
                        kplist.append([x,y,r])
                    kpsinfo[key] = kplist
            kpfile.close()
    return kpsinfo
'''
    Description: 读取标注和
    return: 包含所有set,所有特征点位置以及尺度信息
'''
def kpAnnMatch():
    kppathes = getSURFKPPath()
    kpsinfo = {}
    anninfo = {}
    matchednum = 0#匹配上的bbox个数，用于计算召回率
    bboxesnum = 0#标注框总数量
    seginfo = parse_detected_box()
    for setnum in range(len(kppathes)):#处理所有set
        for kpvpath in kppathes[setnum]:#处理set中所有特征点文件
            print(kpvpath)
            vmatched = 0
            vbboxnum = 0
            #解析surf特征点
            kpfile = open(kpvpath,'r')
            indrs = kpvpath.rfind('\\')
            indp = kpvpath.rfind('.')
            vstr = kpvpath[indrs+2:indp]#获取不带后缀的视频名，并去掉文件名首字母V
            for line in kpfile.readlines():
                #line示例 00004:527 264 22;81 26 22;22 34 18;277 245 17;71 25 22;484 261 18;428 266 34;
                line = line.strip()
                framestr = line[:5]#帧号
                key = '%02d'%(setnum) + vstr + framestr
                if len(line) == 6:#当前帧没有特征点，
                    kpsinfo[key] = []
                else:
                    kpstrs = line[6:-1]#去掉最后分号
                    kps = kpstrs.split(';')
                    kplist = []
                    for kp in kps:
                        kpstr = kp.split(' ')
                        x,y,r = int(kpstr[0]),int(kpstr[1]),int(kpstr[2])
                        kplist.append([x,y,r])
                    kpsinfo[key] = kplist
            kpfile.close()
            #解析标注文件
            anninfo = parse_ann_filtered(setnum,int(vstr))
            for framekey in anninfo.keys():#处理每一帧
                for annbbox in anninfo[framekey]:#处理单帧中所有bbox
                    combinedinfo = combine_seg_surf_roi(kpsinfo,seginfo,framekey)
                    vbboxnum +=1#记录标记框数量
                    for cbox in combinedinfo:#处理单帧中所有bbox
#                        if cbox[3] < 90: continue#值统计大目标 
#                        if cbox[3] >= 90 or cbox[3] < 48: continue#只统计中目标 
                        if cbox[3] >= 48: continue#只统计小目标                        
                        if sateval.iou(annbbox,cbox,[0.3,0.8]):#对应frame的surf特征点
                            vmatched +=1
                            break
            if vbboxnum == 0:print('%02d%s: %lf'%(setnum,vstr,10000))#当前视频没有符合条件的标注，bbox数量为0
            else:print('%02d%s: %lf'%(setnum,vstr,vmatched*1.0/vbboxnum))
            bboxesnum += vbboxnum
            matchednum += vmatched
    print('%d/%d recall:%lf'%(matchednum,bboxesnum,matchednum*1.0/bboxesnum))
def combine_seg_surf_roi(kpsinfo,seginfo,framekey):
    
    combinedbbox = []
    segbbox = seginfo[framekey] 
    kpsinfo =kpsinfo[framekey] 
    combinedbbox = [bbox for bbox in segbbox]
    for kp in kpsinfo:
        kpbbox = surfKP2ROIs(kp)
        for box in kpbbox:
            combinedbbox.append(box)
    return combinedbbox
        
'''
    Description: 判断surf特征点kp是否与bbox匹配
    return: 包含所有set,所有特征点位置以及尺度信息
'''
def isMatched(kp,bbox):
#    kpx,kpy,kpr = kp[0]-104,kp[1]-105,kp[2]#annbox 为相对中间区域，w需要减104，h需要减105
#    bx,by,bw,bh = bbox[0],bbox[1],bbox[2],bbox[3]
    #surf特征点位于bbox上半区域,surf尺度小于bbox宽
#    if kpx < bx + bw and kpx > bx and kpy > by and kpy < by + bh/2 and kpr < bw:
#        return True
#    else: return False
    for k in surfKP2ROIs(kp):
        if sateval.iou(k,bbox,[0.3,0.8]): return True
    return False
'''
    DESCRIPTION: parse ann_filtered file
    PARAMETERS:
        set_num: set00->set20
        v_num: such as set00 -> v000.txt,v001.txt,v002.txt       
'''
def parse_ann_filtered(set_num,v_num):
    
    parsed_box_dict = {}
    path_ann_prefix = './data/ann_filter/'
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
    Description: 将surf特征点转化为ROI    
'''
def surfKP2ROIs(kp):
    kpx,kpy,kpr = kp[0]-104,kp[1]-105,kp[2]#annbox 为相对中间区域，w需要减104，h需要减105
    by = kpy - int(kpr)
    bx = kpx - int(kpr*1.5)
    bw = 3*kpr
    bh = 6*kpr
    xlist = [1,1.5,2,4]
    wlist = [1,2,3]
    hlist = [3,6,8]
    rois = []
    for x_ in xlist:
        for w_ in wlist:
            for h_ in hlist:
                by = kpy - int(kpr)
                bx = kpx - int(kpr*x_)
                bw = w_*kpr
                bh = h_*kpr
                rois.append([bx,by,bw,bh])
#    rois = [[bx,by,bw,bh],[bx,by,bw,bh]]
    return rois
#parseKPFile()
#parse_detected_box()
kpAnnMatch()