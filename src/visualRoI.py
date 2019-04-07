# -*- coding: utf-8 -*-
import cv2
from visual import Visual
import PD.hogsvm as hogsvm
import PD.roiextract as roiextract
import PD.roifilter as roifilter
''''''
def surfDetect(outPath = None):
    if outPath != None:#将结果写出到视频
        fps = 25   #视频帧率
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D') 
        videoWriter = cv2.VideoWriter(outPath, fourcc, fps, (768,576))   #(1360,480)为视频大小
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(r'H:\human_dataset\3_shenle.avi')
    ii = 0
    while True:
        ret , frame = cap.read()
        ii +=1     
        grayim = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret == False:break
        cv2.imshow('origin',frame)
        cv2.waitKey(10)
        ceterframe = frame[105:105 + 300,104:104 + 512]
        cetergrayim = grayim[105:105 + 300,104:104 + 512]
        filteredkps,surfrois = roiextract.surf_roi(cetergrayim,512,300)
        kpimg = cv2.drawKeypoints(ceterframe,filteredkps,None,(255,0,0),4)
        personroi = []
        filteredrois = roifilter.getFilteredROI(surfrois)
        for roi in filteredrois:
            if hogsvm.svmPredict(cetergrayim,roi) == 1:
                personroi.append(roi)
                print('person')
        cv2.imshow('surf keypoint',kpimg)
        cv2.waitKey(10)
        Visual.showResult(cetergrayim,personroi,winname = 'person roi',color = 200)#color值越大越线条越偏白
        if cv2.waitKey(1) == 32:
            print('space')
            while True:
                if(cv2.waitKey(1) == 32):break
        if outPath != None:#将结果写出到视频
            videoWriter.write(kpimg)
    cv2.destroyAllWindows()
'''
    Description:阈值分割提ROI，svm分类器检测
'''
def segDetect():
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(r'H:\human_dataset\3_shenle.avi')
    ii = 0
    while True:
        ret , frame = cap.read()
        ii +=1        
        grayim = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret == False:break
        cv2.imshow('origin',frame)
        cv2.waitKey(10)
        ceterframe = frame[105:105 + 300,104:104 + 512]
        cetergrayim = grayim[105:105 + 300,104:104 + 512]
        threshimg,segrois = roiextract.segmetation_roi(ceterframe)
        personroi = []
        filteredrois = roifilter.getFilteredROI(segrois)
        print(len(filteredrois))
        for roi in filteredrois:
            if hogsvm.svmPredict(cetergrayim,roi) == 1:
                personroi.append(roi)
                print('person')
        Visual.showResult(threshimg,segrois,winname = 'seg roi',color = 200)#color值越大越线条越偏白
#        for roi in segrois:
#            hog.svmPredict(cetergrayim,roi)
        Visual.showResult(cetergrayim,personroi,winname = 'person roi',color = 200)#color值越大越线条越偏白
        if cv2.waitKey(1) == 32:
            print('space')
            while True:
                if(cv2.waitKey(1) == 32):break
    cv2.destroyAllWindows()
'''
    Description:可视化双阈值分割图像
'''
def showSegImg():
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(r'H:\human_dataset\3_shenle.avi')
    ii = 0
    while True:
        ret , frame = cap.read()
        ii +=1        
        grayim = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret == False:break
        ceterframe = frame[105:105 + 300,104:104 + 512]
#        cetergrayim = grayim[105:105 + 300,104:104 + 512]
        threshimg,segrois = roiextract.segmetation_roi(ceterframe)
        count = 0
        for row in range(threshimg.shape[0]):
            for col in range(threshimg.shape[1]):
                if threshimg[row,col] > 0:
                    count += 1
        print('前景点数：%d',count)
        cv2.imshow('seg',threshimg)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
'''
    Description:可视化SURF生成的ROI
'''
def showSURFROI():
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(r'H:\human_dataset\3_shenle.avi')
    ii = 0
    while True:
        ret , frame = cap.read()
        ii +=1        
        grayim = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret == False:break
        cv2.imshow('origin',frame)
        cv2.waitKey(10)
        ceterframe = frame[105:105 + 300,104:104 + 512]
        cetergrayim = grayim[105:105 + 300,104:104 + 512]
        filteredkps,surfrois = roiextract.surf_roi(cetergrayim,512,300)
        kpimg = cv2.drawKeypoints(ceterframe,filteredkps,None,(255,0,0),4)
        cv2.imshow('surf keypoint',kpimg)
        cv2.waitKey(10)
#        for roi in segrois:
#            hog.svmPredict(cetergrayim,roi)
        Visual.showResult(cetergrayim,surfrois,winname = 'surf',color = 200)#color值越大越线条越偏白
        if cv2.waitKey(1) == 32:
            print('space')
            while True:
                if(cv2.waitKey(1) == 32):break
    cv2.destroyAllWindows()

'''
    Description:可视化双阈值分割提取的ROI
'''
def showSegROI():
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(r'H:\human_dataset\3_shenle.avi')
    ii = 0
    while True:
        ret , frame = cap.read()
        ii +=1        
        grayim = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret == False:break
        cv2.imshow('origin',frame)
        cv2.waitKey(10)
        ceterframe = frame[105:105 + 300,104:104 + 512]
        cetergrayim = grayim[105:105 + 300,104:104 + 512]
        threshimg,segrois = roiextract.segmetation_roi(ceterframe)
#        dimg = cv2.adaptiveThreshold(src = graim,maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType = cv2.THRESH_BINARY_INV,blockSize = 5,C = 3)
#        dimg = doDualThreshold()#statrow = 105,statcol = 104
#        labelrois = labelConnectedComponent(dimg)
#        for roi in segrois:
#            hog.svmPredict(cetergrayim,roi)
        Visual.showResult(cetergrayim,segrois,winname = 'seg',color = 200)#color值越大越线条越偏白
        if cv2.waitKey(1) == 32:
            print('space')
            while True:
                if(cv2.waitKey(1) == 32):break
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    pass
#    showSegImg()
#    showSegROI()
#    showSURFROI()
#    segDetect()
#    segXGBoostDetect()
