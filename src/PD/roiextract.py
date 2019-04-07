# -*- coding: utf-8 -*-
import numpy as np
import cv2
'''
    ====================================================
    File Description:阈值分割提取RoI,滑窗提取RoI,SURF提取RoI
    ====================================================
'''


'''
    Description:双阈值分割提ROI
    @im 512—w*300-h图像
    @return 阈值分割结果图，联通区域ROIs
'''
def segmetation_roi(im):
    threshim = __doDualThreshold(im)
    rois = __labelConnectedComponent(threshim)
    return threshim,rois

'''
    Description:滑窗法提ROI
    @param img_width: 图像宽
    @param img_height:图像高
    @param max_h:最大ROI高度
    @param min_h:最小ROI高度
    @param w_step:横向滑动步长
    @param h_step:纵向滑动步长
    @return 滑窗ROIs
'''
def slide_roi(self,img_width,img_height,max_h = 42,min_h = 24,w_step = 6,h_step = 6):
    bbox = []
    box = []
    MAXHEIGHT = max_h
    MINHEIGHT = min_h
        
    width_step = w_step
    height_step = h_step
    
    cur_height = MINHEIGHT
    while cur_height <= MAXHEIGHT :
        for x in range(1,img_width - cur_height,width_step):
            for y in range(1 , img_height - cur_height , height_step):
                box = []
                width = cur_height
                height = cur_height
                box.append(x)
                box.append(y)
            box.append(width)
            box.append(height)
            bbox.append(box)
        cur_height = cur_height + 5
    return bbox
'''
    Description:surf 头部检测提ROI
    @param img: 待提取ROI的图像
    @param width:图像宽
    @param height:图像高
    @return SURF kp点与ROIs
'''
def surf_roi(img,width = 512,height = 300):
    surf = cv2.xfeatures2d.SURF_create(5000)   
    print('NOctaveLayers:%d'%(cv2.xfeatures2d_SURF.getNOctaveLayers(surf)))
    print('NOctaves:%d'%(cv2.xfeatures2d_SURF.getNOctaves(surf)))
    kps, des = surf.detectAndCompute(img,None)
    rois = []#返回提取到的roi
    filteredkps = __surfKPFilter(kps)#过滤kp
    for point in filteredkps:#遍历所有surf点，去掉较大的点和非常小的点
        x_center,y_center = point.pt
        r = point.size
        kp = [int(x_center),int(y_center),int(r)]
        framerois = __surfKP2ROIs(kp,width,height)
        for roi in framerois:
            rois.append(roi)
    print(len(rois))
    return filteredkps,rois
def __surfKPFilter(kps):
    fiteredkps = []
    for kp in kps:
        if kp.size > 40:continue#过滤大框
        fiteredkps.append(kp)
    return fiteredkps
'''
    Description: 将surf特征点转化为ROI    
    @param kp:surf特征点，[center_x,center_y,r]
    @ width,height:原图像宽高
'''
def __surfKP2ROIs(kp,width,height):
    kpx,kpy,kpr = kp[0],kp[1],kp[2]
#    xlist = [1,1.5,2,4]
#    wlist = [1,2,3]
#    hlist = [3,6,8]
    xlist = [1,1.5]
    wlist = [2,3]
    hlist = [ w* 1.41 for w in wlist]
    rois = []
    for x_ in xlist:
        for w_ in wlist:
            for h_ in hlist:
                by = int(kpy) - int(kpr)
                bx = int(kpx) - int(kpr*x_)
                bw = int(w_*kpr)
                bh = int(h_*kpr)
                rois.append(__validROI(bx,by,bw,bh,width,height))
#    rois = [[bx,by,bw,bh],[bx,by,bw,bh]]
    return rois
'''
    Description:验证box = [x,y,w,h]是否在im_w*im_h图像内部
    @param x,y,w,h： x,y为ROI左顶点坐标，w,h为roi宽高
'''
def __validROI(x,y,w,h,im_w,im_h):
    if x < 0: x = 0
    if y < 0: y = 0
    if w < 0: w = 0
    if h < 0: h = 0
    if x > im_w: x = im_w
    if y > im_h: y = im_h
    if x + w > im_w: w = im_w - x
    if y + h > im_h: h = im_h - y
    return [x,y,w,h]
'''
    Description:葛俊峰论文中自适应双阈值分割
    @return rois = [[x,y,w,h],[x,y,w,h]]
'''
def __doDualThreshold(im):
    OMEGA = 13
    ALPHA = 1
    KAY = 1.06
    EPSILON = 4
    GAMMA = 5
    BETA = 220
    grayim = im
    sum_ = 0
    height,width = im.shape[0],im.shape[1]
    if im.shape[2] != 1:#将彩色图像转换为灰度图像
        grayim = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresim = np.zeros((im.shape[0],im.shape[1]),dtype = np.uint8)
    Yd = None
    TL,TH,T1,T2,T3 = None,None,None,None,None
    Iij = None
    for row in range(0,height,2):
        sum_ = sum(grayim[row,:2*OMEGA+1])
        for cols in range(OMEGA+1,width - OMEGA,1):
            sum_ += grayim[row,cols + OMEGA]
            sum_ -= grayim[row,cols - OMEGA - 1]
            TL = sum_ / ((OMEGA*2) + 1 ) + ALPHA
            T3 = max((KAY*(TL - ALPHA)),TL + EPSILON)
            T2 = min(T3,TL + GAMMA)
            T1 = min(T2,BETA)
            TH = max(T1,TL)
            Iij = grayim[row,cols]
            if Iij > TH:
                if grayim[row,cols -1]>0 or grayim[row,cols + 1] > TL:Yd = 255
                else: Yd = 0
            elif Iij < TL:Yd = 0
            else: Yd = thresim[row,cols - 1]
            thresim[row,cols] = Yd
        thresim[row + 1] = thresim[row]
#        for i in range(width):
#            thresim[row + 1,i] = thresim[row,i]
    return thresim
'''
    Description:联通区域标记，参考飒特项目ExtractModule.c中labelConnectedComponent
'''
def __labelConnectedComponent(threshim):
    PADDING = 3
    DST_WIDTH = threshim.shape[1]
    DST_HEIGHT = threshim.shape[0]
    rois = []#联通区域
    c_c_img = np.zeros(threshim.shape,int)

    label_set = np.zeros(5270,dtype = int)
    used_label = 1
    rows = threshim.shape[0] - 1
    cols = threshim.shape[1] - 1 
    neighbor_labels = [0,0]
    neighbor_num,left_neighbor,up_neighbor = 0,0,0
    min_label,max_label,temp_min,temp_label = 0,0,0,0
    old_min_label,pixel_label,roi_num = 0,0,0
    cur_label,pre_label = 0,0
    img_rois = np.zeros(51200,dtype = int)
    roi_x,roi_y,roi_width,roi_height = 0,0,0,0
    label_set[1] = 1
    
    for i in range(1,rows):
#        label_pre_row = c_c_img[i-1,:]
#        label_cur_row = c_c_img[i,:]
#        data_cur_row = threshim[i,:]
        for j in range(1,cols):
            if threshim[i,j] == 255:
                neighbor_num = 0
                left_neighbor = c_c_img[i,j-1]#左邻居label
                up_neighbor = c_c_img[i-1,j]#上邻居label
                if left_neighbor > 1:
                    neighbor_labels[neighbor_num] = left_neighbor
                    neighbor_num +=1
                if up_neighbor > 1:
                    neighbor_labels[neighbor_num] = up_neighbor
                    neighbor_num += 1
                if neighbor_num < 1:
                    used_label +=1
                    label_set[used_label] = used_label
                    c_c_img[i,j] = used_label
                elif neighbor_num == 1:
                    min_label = neighbor_labels[0]
                    c_c_img[i,j] = min_label
                else:
                    if neighbor_labels[0] > neighbor_labels[1]:
                        temp_min = neighbor_labels[0]
                        neighbor_labels[0] = neighbor_labels[1]
                        neighbor_labels[1] = temp_min
                    min_label = neighbor_labels[0]
                    c_c_img[i,j] = min_label
                    
                    temp_label = neighbor_labels[1]
                    old_min_label = label_set[temp_label]
                    if old_min_label > min_label:
                        label_set[old_min_label] = min_label
                        label_set[temp_label] = min_label
                    elif old_min_label < min_label:
                        label_set[min_label] = old_min_label
    for i in range(2,used_label+1):
        cur_label = label_set[i]
        pre_label = label_set[cur_label]
        while pre_label != cur_label:
            cur_label = pre_label
            pre_label = cur_label
        label_set[i] = cur_label
    max_label = 0
    for i in range(2,used_label+1):
        if label_set[i] > max_label:
            max_label = label_set[i]
    for i in range(max_label+1):
        img_rois[i * 5 ] = 0
        img_rois[i * 5 + 1] = 10000
        img_rois[i * 5 + 2] = 10000
        img_rois[i * 5 + 3] = -1
        img_rois[i * 5 + 4] = -1
    for i in range(1,rows):
        for j in range(1,cols):
            pixel_label = c_c_img[i,j]
            c_c_img[i,j] = label_set[pixel_label]
            pixel_label = c_c_img[i,j]
            if pixel_label > 1:
                if j < img_rois[pixel_label * 5 + 1]:
                    img_rois[pixel_label * 5] = pixel_label
                    img_rois[pixel_label * 5 + 1] = j
                if j > img_rois[pixel_label * 5 + 3]:
                    img_rois[pixel_label * 5] = pixel_label
                    img_rois[pixel_label * 5 + 3] = j
                if i < img_rois[pixel_label * 5 + 2]:
                    img_rois[pixel_label * 5] = pixel_label
                    img_rois[pixel_label * 5 + 2] = i
                if i > img_rois[pixel_label * 5 + 4]:
                    img_rois[pixel_label * 5] = pixel_label
                    img_rois[pixel_label * 5 + 4] = i
    roi_num = 0
    for i in range(2,max_label + 1):
        if img_rois[i*5] > 1:
            roi_num += 1
            roi_x = img_rois[i*5 + 1] - PADDING
            roi_y = img_rois[i*5 + 2] - PADDING
            roi_width = img_rois[i * 5 + 3] - img_rois[i * 5 + 1] + 1 + PADDING * 2
            roi_height = img_rois[i * 5 + 4] - img_rois[i * 5 + 2] + 1 + PADDING * 2
            if roi_x < 0:
                roi_x = 0
            if roi_y < 0:
                roi_y = 0
            if roi_x + roi_width >= DST_WIDTH:
                roi_width = DST_WIDTH - roi_x
            if roi_y + roi_height >= DST_HEIGHT:
                roi_height = DST_HEIGHT - roi_y
            rois.append([roi_x,roi_y,roi_width,roi_height])  
#    print(rois)
    return rois