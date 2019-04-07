# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
    DESCRIPTION: compute the two input rectangle IOU(area(ann&&detected)/(area(ann) || area(detected))))
    PARAM:
        rec1: x,y,w,h
        rec2: x,y,w,h      
'''
def iou(rec1,rec2):
    x1,y1,width1,height1 = rec1[0],rec1[1],rec1[2],rec1[3]
    x2,y2,width2,height2 = rec2[0],rec2[1],rec2[2],rec2[3]

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);
    ratio = 0
    if width <=0 or height <= 0:
        ratio = 0 # iou = 0 
    else:
        Area = width*height; # 
        Area1 = width1*height1; 
        Area2 = width2*height2;
        ratio = Area*1./(Area1+Area2-Area);
    return ratio
'''
    DESCRIPTION: compute the two input rectangle IOU(area(ann&&detected)/area(ann))
    PARAM:
        rec1: x,y,w,h (ann)
        rec2: x,y,w,h (detected)  
'''
def ioa(rec1,rec2):
    x1,y1,width1,height1 = rec1[0],rec1[1],rec1[2],rec1[3]
    x2,y2,width2,height2 = rec2[0],rec2[1],rec2[2],rec2[3]

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);
    ratio = 0
    if width <=0 or height <= 0:
        ratio = 0 # iou = 0 
    else:
        Area = width*height; # 
        Area1 = width1*height1; 
        ratio = Area*1./(Area1);
    # return IOU
    return ratio
def iouAndioa(ann,detected,ioulist):
    if iou(ann,detected) > ioulist[0] or ioa(ann,detected) > ioulist[1]:return True
    return False
'''
    DESCRIPTION: parse ann_filtered file
    PARAMETERS:
        dbpath: where the ann_filtered file located
        set_num: set00->set20
        v_num: such as set00 -> v000.txt,v001.txt,v002.txt       
'''
