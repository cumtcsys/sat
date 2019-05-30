# -*- coding: utf-8 -*-
'''统计宽、高、中心点、左上角点关系'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils import datatools
'''
    =============================================
    统计SCUT数据集头部位置、宽高分布并可视化
    =============================================
'''
'''
    @Description:将所有的GT转化为[[左上点，中心点，宽，高],...,]
'''
def __box2whc():
    whclist = []
    ann_filtered_dict = datatools.parse_all_ann_filtered()
    for key in ann_filtered_dict:
        for gt in ann_filtered_dict[key]:
            x,y,w,h = gt[0],gt[1],gt[2],gt[3]
#            if h >200:print(gt)
            ltp = tuple([x,y])
            cp = tuple([int(x+w/2),int(y+h/2)])
            whclist.append([ltp,cp,w,h])
    return whclist
'''将PD程序检测出的kp（与检测框一样的格式）转化成点'''
def __surfbox2kp():
    kp = []
    detect_bboxes = datatools.parse_whole_dataset_detected_box('surf')
    for framestr,frameboxes in detect_bboxes.items():
        for box in frameboxes:
            x,y,w = box[0],box[1],box[2]
            kpr = w/5
            kpx = int(x + 2.5*kpr) 
            kpy = int(y + kpr)
            kp.append([kpx,kpy,kpr])
    
    for y in range(300):
        cnt = 0
        for item in kp:
            if item[1] == y:cnt = cnt + 1
        print('%d:%d',y,cnt)
                
    return kp


'''统计scut数据集宽高分布，以三维图可视化'''
def line3d(whclist):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x = [0 for whc in whclist]
    y = [whc[1][1] for whc in whclist]
    z = [whc[3] for whc in whclist]
#    ax.plot(x,y,z,label = 'ceter point(x,y) h')
    ax.scatter(x, y, z, s=0.5, c=None, depthshade=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('h')
#    ax.legend()
    plt.show()
'''统计scut数据集宽高分布，以二维图进行可视化'''
def line2d(whclist):
    fig = plt.figure()
    y = [whc[1][1] for whc in whclist if whc[3]<60 and whc[3] > 48]
    h = [whc[3] for whc in whclist if whc[3]<60 and whc[3] > 48]
    w = [whc[2] for whc in whclist if whc[3]<60 and whc[3] > 48]
    plt.subplot(1,2,1)
    plt.scatter(h,y,s=0.1)
    plt.ylabel('cpoint y')
    plt.xlabel('h')   
    
    plt.subplot(1,2,2)
    plt.scatter(w,y,s=0.1)
    plt.ylabel('cpoint y')
    plt.xlabel('w')    
'''统计scut数据集BBox中心线位置并可视化'''
def plot_center_line_position():
    whclist = __box2whc()
    plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    x_all = [whc[0][0] for whc in whclist]
    y_all = [300 - (whc[0][1] + whc[3]/2) for whc in whclist]
    x_far = [whc[0][0] for whc in whclist if whc[3]<=48]
    y_far = [300 - (whc[0][1] + whc[3]/2) for whc in whclist if whc[3]<=48]
    x_mid = [whc[0][0] for whc in whclist if whc[3]<=90 and whc[3] > 48]
    y_mid = [300 - (whc[0][1] + whc[3]/2) for whc in whclist if whc[3]<=90 and whc[3] > 48]
    x_near = [whc[0][0] for whc in whclist if whc[3]>=90]
    y_near = [300 - (whc[0][1] + whc[3]/2) for whc in whclist if whc[3]>=90]
    plt_far = plt.subplot(2,3,1)
    plt_mid = plt.subplot(2,3,2)
    plt_near = plt.subplot(2,3,3)
    plt_all = plt.subplot(2,1,2)
    plt.sca(plt_all)
    plt.title('Center Line - All')
    plt.scatter(x_all,y_all,s=0.1,color = 'orange')
    plt.ylabel('y')
    plt.xlabel('x')   
    
    plt.sca(plt_far)
    plt.title('Center Line - far')
    plt.scatter(x_far,y_far,s=0.1,color = 'r')
    plt.ylabel('y')
    plt.xlabel('x')   
    
    plt.sca(plt_mid)
    plt.title('Center Line - mid')
    plt.scatter(x_mid,y_mid,s=0.1,color = 'g')
    plt.ylabel('y')
    plt.xlabel('x')  
    
    plt.sca(plt_near)
    plt.title('Center Line - near')
    plt.scatter(x_near,y_near,s=0.1,color = 'b')
    plt.ylabel('y')
    plt.xlabel('x')  
    plt.show()
'''统计scut数据集BBox左上角点下移1/5h位置并可视化'''
def plot_head_position(whclist):
    #行人头部被认为在RoI上1/5区域
    plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    x_all = [whc[0][0] for whc in whclist]
    y_all = [300 - (whc[0][1] + whc[3]/5) for whc in whclist]
    x_far = [whc[0][0] for whc in whclist if whc[3]<=48]
    y_far = [300 - (whc[0][1] + whc[3]/5) for whc in whclist if whc[3]<=48]
    x_mid = [whc[0][0] for whc in whclist if whc[3]<=90 and whc[3] > 48]
    y_mid = [300 - (whc[0][1] + whc[3]/5) for whc in whclist if whc[3]<=90 and whc[3] > 48]
    x_near = [whc[0][0] for whc in whclist if whc[3]>=90]
    y_near = [300 - (whc[0][1] + whc[3]/5) for whc in whclist if whc[3]>=90]
    plt_far = plt.subplot(2,3,1)
    plt_mid = plt.subplot(2,3,2)
    plt_near = plt.subplot(2,3,3)
    plt_all = plt.subplot(2,1,2)
    plt.sca(plt_all)
    plt.title('Head Position - All')
    plt.scatter(x_all,y_all,s=0.1,color = 'orange')
    plt.ylabel('y')
    plt.xlabel('x')   
    
    plt.sca(plt_far)
    plt.title('Head Position - far')
    plt.scatter(x_far,y_far,s=0.1,color = 'r')
    plt.ylabel('y')
    plt.xlabel('x')   
    
    plt.sca(plt_mid)
    plt.title('Head Position - mid')
    plt.scatter(x_mid,y_mid,s=0.1,color = 'g')
    plt.ylabel('y')
    plt.xlabel('x')  
    
    plt.sca(plt_near)
    plt.title('Head Position - near')
    plt.scatter(x_near,y_near,s=0.1,color = 'b')
    plt.ylabel('y')
    plt.xlabel('x')  
    plt.show()
    
'''统计PD输出的surfbox的surf点位置并可视化'''
def plot_surfkp_position():
    surfkps = __surfbox2kp()
    plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    x_all = [kp[0] for kp in surfkps]
    y_all = [300 - kp[1] for kp in surfkps]
    x_far = [kp[0] for kp in surfkps if kp[2]*11<=48]
    y_far = [300 - kp[1] for kp in surfkps if kp[2]*11<=48]
    x_mid = [kp[0] for kp in surfkps if kp[2]*11<=90 and kp[2]*11 > 48]
    y_mid = [300 - kp[1] for kp in surfkps if kp[2]*11<=90 and kp[2]*11 > 48]
    x_near = [kp[0] for kp in surfkps if kp[2]*11>=90]
    y_near = [300 - kp[1] for kp in surfkps if kp[2]*11>=90]
    plt_far = plt.subplot(2,3,1)
    plt_mid = plt.subplot(2,3,2)
    plt_near = plt.subplot(2,3,3)
    plt_all = plt.subplot(2,1,2)
    plt.sca(plt_all)
    plt.title('Head Position - All')
    plt.scatter(x_all,y_all,s=0.1,color = 'orange')
    plt.ylabel('y')
    plt.xlabel('x')   
    
    plt.sca(plt_far)
    plt.title('Head Position - far')
    plt.scatter(x_far,y_far,s=0.1,color = 'r')
    plt.ylabel('y')
    plt.xlabel('x')   
    
    plt.sca(plt_mid)
    plt.title('Head Position - mid')
    plt.scatter(x_mid,y_mid,s=0.1,color = 'g')
    plt.ylabel('y')
    plt.xlabel('x')  
    
    plt.sca(plt_near)
    plt.title('Head Position - near')
    plt.scatter(x_near,y_near,s=0.1,color = 'b')
    plt.ylabel('y')
    plt.xlabel('x')  
    plt.show()    
    

'''寻找scut数据集BBox头部集中区域'''
def head_position_cut(whclist):
    x_all = [whc[0][0] for whc in whclist]
    y_all = [300 - (whc[0][1] + whc[3]/5) for whc in whclist]
    x_far = [whc[0][0] for whc in whclist if whc[3]<=48]
    y_far = [300 - (whc[0][1] + whc[3]/5) for whc in whclist if whc[3]<=48]
    x_mid = [whc[0][0] for whc in whclist if whc[3]<=90 and whc[3] > 48]
    y_mid = [300 - (whc[0][1] + whc[3]/5) for whc in whclist if whc[3]<=90 and whc[3] > 48]
    x_near = [whc[0][0] for whc in whclist if whc[3]>=90]
    y_near = [300 - (whc[0][1] + whc[3]/5) for whc in whclist if whc[3]>=90]
    #y下限
    for low in range(100,200+1):
        y_all_num = len([ item for item in y_all if item < low])
        print('all -> thresh = %d:%d/%d  %lf'%(low,y_all_num,len(y_all),y_all_num/len(y_all)))
        y_far_num = len([ item for item in y_far if item < low])
        print('far -> thresh = %d:%d/%d  %lf'%(low,y_far_num,len(y_far),y_far_num/len(y_far)))
        y_mid_num = len([ item for item in y_mid if item < low])
        print('mid -> thresh = %d:%d/%d  %lf'%(low,y_mid_num,len(y_mid),y_mid_num/len(y_mid)))
        y_near_num = len([ item for item in y_near if item < low])
        print('near -> thresh = %d:%d/%d  %lf'%(low,y_near_num,len(y_near),y_near_num/len(y_near)))
        print()
    #y上限
    for low in range(250,300+1):
        y_all_num = len([ item for item in y_all if item > low])
        print('all -> thresh = %d:%d/%d  %lf'%(low,y_all_num,len(y_all),y_all_num/len(y_all)))
        y_far_num = len([ item for item in y_far if item > low])
        print('far -> thresh = %d:%d/%d  %lf'%(low,y_far_num,len(y_far),y_far_num/len(y_far)))
        y_mid_num = len([ item for item in y_mid if item > low])
        print('mid -> thresh = %d:%d/%d  %lf'%(low,y_mid_num,len(y_mid),y_mid_num/len(y_mid)))
        y_near_num = len([ item for item in y_near if item > low])
        print('near -> thresh = %d:%d/%d  %lf'%(low,y_near_num,len(y_near),y_near_num/len(y_near)))
        print()
if __name__ == '__main__':
    
    #head_position_cut(whclist)
    #plot_head_position(whclist)
    plot_center_line_position()
    plot_surfkp_position()
    #line2d(whclist)