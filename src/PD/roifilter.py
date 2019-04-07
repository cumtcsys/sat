# -*- coding: utf-8 -*-
def getFilteredROI(rois):
    filteredrois = __sizeFilter(rois)
    return filteredrois
def __sizeFilter(rois):
    filteredrois = []
    for roi in rois:
        x,y,w,h = roi[0],roi[1],roi[2],roi[3]
        if h < 30 or h >280:continue#去掉太远太近的目标
        if h < w*1.5 or h > w*4:continue#高/宽在（1.5,4）之间
        filteredrois.append(roi)
    return filteredrois