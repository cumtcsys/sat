# -*- coding: utf-8 -*-
import cv2
import PD.roiextract as roiextract

'''可视化surf特征点提取效果'''
def show_surf_point():
    cap = cv2.VideoCapture(r'H:\human_dataset\3_shenle.avi')
    ii = 0
    while True:
        ii +=1
    #    if ii >100:break
        ret , frame = cap.read()
        if ret == False:break
        centerframe = frame[104:104+300,105:105+512]
        #记录surf特征点提取时间
        import time
        s_time = time.time()
        surf = cv2.xfeatures2d.SURF_create(1000,nOctaves = 2,nOctaveLayers = 3,upright = True)   
        kp, des = surf.detectAndCompute(centerframe,None)
        img2 = cv2.drawKeypoints(centerframe,kp,None,(255,0,0),4)
        print(time.time()-s_time)
        cv2.imshow('surf',img2)
        if cv2.waitKey(1) == 32:
            print('space')
            while True:
                if(cv2.waitKey(1) == 32):break
'''可视化orb特征点提取效果'''
def show_orb_point():
    cap = cv2.VideoCapture(r'H:\human_dataset\3_shenle.avi')
    ii = 0
    while True:
        ii +=1
    #    if ii >100:break
        ret , frame = cap.read()
        if ret == False:break
        centerframe = frame[104:104+300,105:105+512]
        #记录surf特征点提取时间
        import time
        s_time = time.time()
        orb = cv2.ORB_create(10)
        kp, des = orb.detectAndCompute(centerframe,None)
        img2 = cv2.drawKeypoints(centerframe,kp,None,(255,0,0),4)
        print(time.time()-s_time)
        cv2.imshow('surf',img2)
        if cv2.waitKey(1) == 32:
            print('space')
            while True:
                if(cv2.waitKey(1) == 32):break            
def show_fast_point():
    cap = cv2.VideoCapture(r'H:\human_dataset\3_shenle.avi')
    ii = 0
    while True:
        ii +=1
        ret , frame = cap.read()
        if ret == False:break
        centerframe = frame[104:104+300,105:105+512]
        #记录surf特征点提取时间
        import time
        s_time = time.time()
        surf = cv2.xfeatures2d.SURF_create(1000,nOctaves = 2,nOctaveLayers = 3,upright = True)   
        fastdetector = cv2.FastFeatureDetector_create()
        fastdetector.setThreshold(20)
        kp_fast = fastdetector.detect(centerframe,None)
        kp_surf, des = surf.detectAndCompute(centerframe,None)
        
        img2 = cv2.drawKeypoints(centerframe,kp_fast,None,(255,0,0),2)
        fastboxes = kpfast2box(kp_fast)
        for box in fastboxes:
            img2 = cv2.rectangle(img2,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,0,255))
            img2 = cv2.drawKeypoints(img2,kp_surf,None,(0,255,0),2)
        print(time.time()-s_time)
        cv2.imshow('surf',img2)
        cv2.waitKey(50)
        if cv2.waitKey(1) == 32:
            print('space')
            while True:
                if(cv2.waitKey(1) == 32):break
'''将kp特征点转化为RoIs'''
def kpfast2box(kp_fast):
    bbox = []
    xalign = 6
    yalign = 3
    w = 12
    h = 30
    for kp in kp_fast:
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        bbox.append([x - xalign,y - yalign,w,h])
    return bbox
'''
    Description:在联通区域内提surf特征点(以减少分割断裂与分割粘连的情况)
'''
def seg_surf():
    cap = cv2.VideoCapture(r'H:\human_dataset\3_shenle.avi')
    ii = 0
    while True:
        ii +=1
    #    if ii >100:break
        ret , frame = cap.read()
        if ret == False:break
        centerframe = frame[104:104+300,105:105+512]
        #记录surf特征点提取时间
        import time
        s_time = time.time()
        surf = cv2.xfeatures2d.SURF_create(1000,)    
        
        img2 = centerframe
        threshim,rois = roiextract.segmetation_roi(centerframe)
        t_seg = time.time() - s_time
        print('阈值分割时间:%lf'%(t_seg))
        print(rois)
        for roi in rois:
            x,y,w,h = roi
            kp, des = surf.detectAndCompute(centerframe[y:y+h,x:x+w],None)
            img2 = cv2.drawKeypoints(img2,kp,None,(255,0,0),4)
        t_seg_surf = time.time()-s_time
        print('阈值分割+联通区域surf时间:%lf'%(t_seg_surf))
        print('surf时间:%lf',t_seg_surf - t_seg)
        cv2.imshow('surf',img2)
        if cv2.waitKey(1) == 32:
            print('space')
            while True:
                if(cv2.waitKey(1) == 32):break
if __name__ == "__main__":
    show_fast_point()




