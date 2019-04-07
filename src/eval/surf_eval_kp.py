import glob 
import cv2
import numpy as np
import scut_sat_eval as sateval
from visual import Visual
import roiextract
'''
    File Description:根据提取的SURF特征点来评估SURF召回率（SURF未嵌入到DSP版本之前，嵌入之后本文程序弃用）
'''
#python 版本提取出的surf特征点的对象768*576原始图像，c++版本surf提取的surf特征点的对象是512*300中心区域
#PYTHON = True测试python版本surf特征点匹配召回率，=False测试c++版本特征点匹配召回率
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
    prepath = 'E:\\workspace\\spyder_workspace\\sat\\data\\ann_filter'
    for setnum in range(21):
        setpath = '%s\\set%02d\\*.txt'%(prepath,setnum)
        pathes.append(glob.glob(setpath))
    return pathes
'''
    Description: 灰度映射
'''
def grayMap(im):
    k1,k2 = 3,10
    for ci in range(3):
        imc = im[:,:,ci]
        mu = np.mean(imc)
        sigma = np.std(imc)
        for ii in range(imc.shape[0]):
            for jj in range(imc.shape[1]):
                x = imc[ii,jj]
                if x < mu - k1*sigma:
                    im[ii,jj,ci] = 0
                elif x >= mu + k2*sigma: 
                    im[ii,jj,ci] = 255
                else:
                    im[ii,jj,ci] = 255 * (x - (mu -k1*sigma)) * 1.0 /((mu + k2*sigma)-(mu - k1*sigma))
    return im

'''
    Description: 对scut所有视频执行一次surf特征点提取，并与scut数据集一样的目录结构写入提取到的surf特征点    
'''
def surfGrayMap():
    pathes = getVideoPath()
    for setnum in range(len(pathes)):
        for vpath in pathes[setnum]:
           indrs = vpath.rfind('\\')
           indp = vpath.rfind('.')
           vstr = vpath[indrs+1:indp]#获取不带后缀的视频名
#           outpath = './data/surf/set%02d/%s.txt'%(setnum,vstr)
#           outfile = open(outpath,'w+')
           
           cap = cv2.VideoCapture(vpath)
           ii = 1
           while True:
#               line = ''
               ret ,frame = cap.read()
               grayMap(frame)
               if ret == False:break
               #一行写一帧的结果，先先帧号
#               line += '%05d:'%(ii)
               surf = cv2.xfeatures2d.SURF_create(1000)
               kp,des = surf.detectAndCompute(frame,None) 
               surf_im = cv2.drawKeypoints(frame,kp,None,(255,0,0),4)
               cv2.imshow('set%02d/%s'%(setnum,vstr),surf_im)
               cv2.waitKey(1000)
               
               gray_frame = grayMap(frame)
               surf = cv2.xfeatures2d.SURF_create(1000)
               kp,des = surf.detectAndCompute(gray_frame,None) 
               surf_gray_im = cv2.drawKeypoints(gray_frame,kp,None,(255,0,0),4)
               cv2.imshow('set%02d/%s graymap'%(setnum,vstr),surf_gray_im)
               cv2.waitKey(1000)
               
#               for point in kp:
#                   x,y = point.pt
#                   x,y = int(x),int(y)
#                   r = int(point.size)
#                   line += '%d %d %d;'%(x,y,r)
#               line += '\n'
#               outfile.write(line)
#               outfile.flush()
#               ii += 1
#           outfile.close()
           cv2.destroyAllWindows()
           
'''
    Description: 对scut所有视频执行一次surf特征点提取，并与scut数据集一样的目录结构写入提取到的surf特征点    
'''
def surf():
    pathes = getVideoPath()
    for setnum in range(len(pathes)):
        for vpath in pathes[setnum]:
           indrs = vpath.rfind('\\')
           indp = vpath.rfind('.')
           vstr = vpath[indrs+1:indp]#获取不带后缀的视频名
           outpath = './data/surf/set%02d/%s.txt'%(setnum,vstr)
           outfile = open(outpath,'w+')
           
           cap = cv2.VideoCapture(vpath)
           ii = 1
           while True:
               line = ''
               ret ,frame = cap.read()
               if ret == False:break
               #一行写一帧的结果，先先帧号
               line += '%05d:'%(ii)
               surf = cv2.xfeatures2d.SURF_create(1000)
               kp,des = surf.detectAndCompute(frame,None) 
               surf_im = cv2.drawKeypoints(frame,kp,None,(255,0,0),4)
               cv2.imshow('set%02d/%s'%(setnum,vstr),surf_im)
               cv2.waitKey(1)
               for point in kp:
                   x,y = point.pt
                   x,y = int(x),int(y)
                   r = int(point.size)
                   line += '%d %d %d;'%(x,y,r)
               line += '\n'
               outfile.write(line)
               outfile.flush()
               ii += 1
           outfile.close()
           cv2.destroyAllWindows()


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
'''
    Description: 解析所有特征点文件，获取特征中心点，半径信息,以字典形式返回
    return: 包含所有set,所有特征点位置以及尺度信息
'''
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
'''
    Description: 解析标注，kps文件，并可视化显示kp特征点和标注匹配情况
    return: 包含所有set,所有特征点位置以及尺度信息
'''
def kpAnnShow():
    kpsinfos = parseAllKPFile()#解析所有kp文件
    anninfos = parse_all_ann_filtered()#解析所有标注
    videopathes = getVideoPath()#读取所有视频路径
    for setnum in range(len(videopathes)):#处理所有set
        for vpath in videopathes[setnum]:#处理set中所有特征点文件
            indrs = vpath.rfind('\\')
            indp = vpath.rfind('.')
            vstr = vpath[indrs+2:indp]#获取去掉"V"和后缀的视频名
            cap = cv2.VideoCapture(vpath)
            frameindex = 0
            vkey = '%02d'%(setnum)+vstr
            anninfo = anninfos[vkey]
            kpsinfo = kpsinfos[vkey]
            while True:
               frameindex +=1
               ret,frame = cap.read()
               if ret == False:break
               #kpskey = '%05d'%(frameindex)
               framekey = vkey+'%05d'%(frameindex)
               kps = kpsinfo[framekey]
               ann = anninfo[framekey]
               ceterframe = frame[105:105 + 300,104:104 + 512]
               threshimg,segrois = roiextract.segmetation_roi(ceterframe)
               kprois = cKPS2ROIs(kps)
               Visual.showResult(ceterframe,ann,'annotation',(255,0,0))
               Visual.showResult(ceterframe,kprois,'surf_roi',(0,0,255))
               Visual.showResult(ceterframe,segrois,winname = 'seg',color = 200)#color值越大越线条越偏白
#               cv2.waitKey(50)
    return None
    
'''
    Description: 计算surf特征点提roi的召回率
    return: 包含所有set,所有特征点位置以及尺度信息
'''
def kpAnnMatch():
    kppathes = getSURFKPPath()
    kpsinfo = {}
    anninfo = {}
    matchednum = 0#匹配上的bbox个数，用于计算召回率
    bboxesnum = 0#标注框总数量
    kpnum = 0;
    for setnum in range(len(kppathes)):#处理所有set
        for kpvpath in kppathes[setnum]:#处理set中所有特征点文件
            vmatched = 0
            vbboxnum = 0
            kpvideonum = 0
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
            #解析标注文件
            anninfo = parse_ann_filtered(setnum,int(vstr))
            for framekey in anninfo.keys():#处理每一帧
                for bbox in anninfo[framekey]:#处理单帧中所有bbox
#                    if bbox[3] > 48: continue#只统计远距离目标（小目标）
#                    if bbox[3]<=48 or bbox[3]>=90:continue#只统计中距离目标（中目标）
#                    if bbox[3] < 90:continue#只统计近距离目标（大目标）  
                    vbboxnum +=1#记录标记框数量
                    kpvideonum += len(kpsinfo[framekey])#累加每帧中的surf特征点数
                    for kp in kpsinfo[framekey]:#对应frame的surf特征点
                        if isMatched(kp,bbox):
                            vmatched +=1
                            break
            if vbboxnum == 0:
                print('%02d%s: %lf'%(setnum,vstr,10000))#当前视频没有符合条件的标注，bbox数量为0
            else:
                print('%02d%s:kps = %d %d/%d %lf'%(setnum,vstr,kpvideonum,vmatched,vbboxnum,vmatched*1.0/vbboxnum))
                
            bboxesnum += vbboxnum
            matchednum += vmatched
            kpnum +=kpvideonum
    print('%d/%d %d recall:%lf'%(matchednum,bboxesnum,kpnum,matchednum*1.0/bboxesnum))

'''
    Description: 判断surf特征点kp是否与bbox匹配
    return: 包含所有set,所有特征点位置以及尺度信息
'''
def isMatched(kp,bbox):
    if PYTHON:
        for k in surfKP2ROIs(kp):
            if sateval.iou(bbox,k,[0.3,0.8]): return True
    else:
        for k in cSurfKP2ROIs(kp):
            if sateval.iou(bbox,k,[0.7,1.5]): return True
    return False
def parse_all_ann_filtered():
    anninfos = {}
    annpathes = getAnnFilteredPath()
    for setnum in range(len(annpathes)):#处理所有set
        for annpath in annpathes[setnum]:#处理set中所有标注文件
            indrs = annpath.rfind('\\')
            indp = annpath.rfind('_')
            vstr = annpath[indrs+2:indp]#获取不带后缀的视频名，并去掉文件名首字母V
            svkey = '%02d%s'%(setnum,vstr)
            anninfos[svkey] = parse_ann_filtered(setnum,int(vstr))
    return anninfos
            
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
    kpx,kpy,kpr = 0,0,0
    if PYTHON:
        kpx,kpy,kpr = kp[0]-104,kp[1]-105,kp[2]#annbox 为相对中间区域，w需要减104，h需要减105
    else:
        kpx,kpy,kpr = kp[0],kp[1],kp[2]#c++提取的版本直接为中心区域坐标
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
'''
    Description: 将C版本surf特征点转化为ROI    
'''
def cSurfKP2ROIs(kp):
    kpx,kpy,kpr = 0,0,0
    if PYTHON:
        kpx,kpy,kpr = kp[0]-104,kp[1]-105,kp[2]/2#annbox 为相对中间区域，w需要减104，h需要减105,Python版本中r为直径
    else:
        kpx,kpy,kpr = kp[0],kp[1],kp[2]#c++提取的版本直接为中心区域坐标
    xlist,ylist = [2.5],[1]
    wlist,hlist = [11],[6]
    rois = []
    for y_ in ylist:
        for x_ in xlist:
            for w_ in wlist:
                for h_ in hlist:
                    by = int(kpy - kpr*y_)
                    bx = int(kpx - kpr*x_)
                    bw = int(w_*kpr)
                    bh = int(h_*kpr)
                    rois.append([bx,by,bw,bh])
    return rois  
def cKPS2ROIs(kps):
    rois = []
    for kp in kps:
        kprois = cSurfKP2ROIs(kp)
        for roi in kprois:
            rois.append(roi)
    return rois

if __name__ == '__main__':
    kpAnnMatch()  
#    kpAnnShow()
#surfKP2ROI()
#surfGrayMap()