# -*- coding: utf-8 -*-
import glob
from src import global_data
PYTHON = False
'''
    Description: get all scut video absolute path
'''
def getVideoPath():
    pathes = []
    prepath = 'H:\\human_dataset\\scut_avi\\videos'
    for setnum in range(21):
        setpath = '%s\\set%02d\\*.avi'%(prepath,setnum)
        pathes.append(glob.glob(setpath))
    return pathes
def getSURFKPPath():
    pathes = []
    #python opencv自带surf算法提取的surf特征点
    #prepath = 'E:\\workspace\\spyder_workspace\\sat\\data\\surf'
    prepath = 'H:\\2018秋季学期\\工作记录\\SURF特征点\\C语言SURF实现\\recall2'
    for setnum in range(21):
        setpath = '%s\\set%02d\\*.txt'%(prepath,setnum)
        pathes.append(glob.glob(setpath))
    return pathes
def getAnnFilteredPath():
    pathes = []
    prepath = './../../data/bbox/ann_filter/'
    for setnum in range(21):
        setpath = '%s\\set%02d\\*.txt'%(prepath,setnum)
        pathes.append(glob.glob(setpath))
    return pathes
def parseKPFile(kpvpath,setnum,vnum):
    kpfile = open(kpvpath,'r')
    kpsinfo = {}
    for line in kpfile.readlines():
        #line示例 00004:527 264 22;81 26 22;22 34 18;277 245 17;71 25 22;484 261 18;428 266 34;
        line = line.strip()
        framestr = line[:5]#帧号
        key = '%02d%03d'%(setnum,vnum) + framestr
        if len(line) == 6 or len(line) == 0:#当前帧没有特征点，
            kpsinfo[key] = []
        else:
            kpstrs = line[6:-1]#去掉最后分号
            kps = kpstrs.split(';')
            kplist = []
            for kp in kps:
                kpstr = kp.split(' ')
                x,y,r=0,0,0
                if PYTHON:
                    x,y,r = int(kpstr[0]),int(kpstr[1]),int(kpstr[2])
                else:
                    x,y,r = int(kpstr[0]),int(kpstr[1]),float(kpstr[2])
                    kplist.append([x,y,r])
                    kpsinfo[key] = kplist
    kpfile.close()
    return kpsinfo
def parseAllKPFile():
    kpsinfos = {}
    kppathes = getSURFKPPath()
    for setnum in range(len(kppathes)):#处理所有set
        for kpvpath in kppathes[setnum]:#处理set中所有特征点文件
            indrs = kpvpath.rfind('\\')
            indp = kpvpath.rfind('.')
            vstr = kpvpath[indrs+2:indp]#获取去掉"V"和后缀的视频名
            kpsinfo = parseKPFile(kpvpath,setnum,int(vstr))
            svkey = "%02d"%(setnum) + vstr
            kpsinfos[svkey] = kpsinfo
    return kpsinfos
def parse_all_ann_filtered():
    anninfos = {}
    annpathes = getAnnFilteredPath()
    for setnum in range(len(annpathes)):#处理所有set
        for annpath in annpathes[setnum]:#处理set中所有标注文件
            indrs = annpath.rfind('\\')
            indp = annpath.rfind('_')
            vstr = annpath[indrs+2:indp]#获取不带后缀的视频名，并去掉文件名首字母V
#            svkey = '%02d%s'%(setnum,vstr)
            v_ann_filtered = parse_ann_filtered(setnum,int(vstr))
#            anninfos[svkey] = parse_ann_filtered(setnum,int(vstr))
            for svfkey in v_ann_filtered.keys():
                anninfos[svfkey] = v_ann_filtered[svfkey]
    return anninfos
            
'''
    DESCRIPTION: parse ann_filtered file
    PARAMETERS:
        set_num: set00->set20
        v_num: such as set00 -> v000.txt,v001.txt,v002.txt       
'''
def parse_ann_filtered(set_num,v_num):
    
    parsed_box_dict = {}
    path_ann_prefix = global_data.ann_filter_path
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
    DESCRIPTION: parse ann_filtered file
    PARAMETERS:
        set_num: set00->set20
        v_num: such as set00 -> v000.txt,v001.txt,v002.txt       
'''
def parse_ann_forFPEval(set_num,v_num):
    
    parsed_box_dict = {}
    path_ann_prefix = global_data.ann_forFPEval_path
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