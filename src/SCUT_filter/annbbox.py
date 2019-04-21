# -*- coding: utf-8 -*-
# Copyright (c) 2018-present, Song Yang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import utils
#height: 20 - 280 
# small: 30-48 midder: 49-90 big: 91-280
# dst_start row: 105 dst_start_col:104
# dst_width: 512 dst_height: 300
#for i in range(1800):
#    print bbox[i]
#print '######'
#print bbox
'''
    DESCRIPTION:
        statistic the number of  objects and bboxs before and after filte in SCUT 
    PARAM:
         vbbs: parse scut datasets vbb annotation files by zhewei' toolbox;
               see /home/xuzhewei/code/pydatatool/pydatatool/scut/utils.py for more information
'''
def count_annbbox(vbbs):
    #record set00,set01,...,set20 info;scut[0] = 2 means set00 has V000,V001,V002 three vbb files
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]
    num = 0
    num_withoutoccl = 0
    id_dict,id_dict_withoutoccl = {},{}
    #process set by set
    for set_num in range(21):
        #process v by v
        for v_num in range(scut[set_num]+1): 
            # bbox stores objLists in one vbb files
            bbox = vbbs['set%02d' % set_num]['V%03d' % v_num]['objLists']
            #process frame by frame,bbox contains one video annotation
            for ii in range(len(bbox)):               
                #process box by box,
                #process one frame's box by box
                for jj in range(len(bbox[ii])):
                    box = bbox[ii][jj]
                    num = num + 1
                    if box['occl'] == 0:num_withoutoccl = num_withoutoccl + 1
                    if not id_dict.has_key('%02d%03d%05d'%(set_num,v_num,box['id'])):
                        id_dict['%02d%03d%05d'%(set_num,v_num,box['id'])] = 0
                    if not id_dict_withoutoccl.has_key('%02d%03d%05d'%(set_num,v_num,box['id'])):
                        if box['occl'] == 0:
                            id_dict_withoutoccl['%02d%03d%05d'%(set_num,v_num,box['id'])] = 0
    print num,num_withoutoccl
    print len(id_dict),len(id_dict_withoutoccl)
'''
    DESCRIPTION: if x lower than low and great than high return true
'''
def inrange(low,high,x):
    if x >=low and x<=high:
        return True
    return False
'''
    DESCRIPTION: if x lower than low and great than high return true
    PARAM:
         vbbs: parse scut datasets vbb annotation files by zhewei' toolbox;
               see /home/xuzhewei/code/pydatatool/pydatatool/scut/utils.py for more information
         outpath: output the filtered annbbox to the outpath
'''
def write_annbbox(vbbs,outpath):
    #record set00,set01,...,set20 info;scut[0] = 2 means set00 has V000,V001,V002 three vbb files
    scut = [2,3,1,2,11,10,6,1,2,1,0,3,3,1,2,11,9,7,1,2,1]
    #process set by set
    for set_num in range(21):
        #process v by v
        for v_num in range(scut[set_num]+1): 
            # bbox stores objLists in one vbb files
            bbox = vbbs['set%02d' % set_num]['V%03d' % v_num]['objLists']
            filepath = outpath + '/set%02d/V%03d_ann.txt' % (set_num,v_num)
            fobj = open(filepath,'w')
            #process frame by frame,bbox contains one video annotation
            for ii in range(len(bbox)):
                frame_str = '%02d%03d%05d' % (set_num,v_num,ii+1)
                lineprefix = frame_str + '['
                fobj.write(lineprefix)
                #process box by box,
                #process one frame's box by box
                for jj in range(len(bbox[ii])):
                    box = bbox[ii][jj]
                    #only cares ride_person and walk_person
#                    if box['lbl'] not in ['ride_person','walk_person']:
                    if box['lbl'] !='ride_person' and box['lbl'] !='walk_person':
#                    if  box['lbl'] !='walk_person':
                        continue
                    rec = box['pos']
                    #to check if there has type of object such as people in the output annbbox file
                    walk_person = -1
                    #filte by xrang,yrang,height
                    if box['lbl'] == 'ride_person':walk_person = 0
                    if box['lbl'] == 'walk_person':walk_person = 1
                    if inrange(104,616,rec[0]) and inrange(105,405,rec[1]) and inrange(104,616,rec[0]+rec[2]) and inrange(105,405,rec[1]+rec[3]) and inrange(30,280,rec[3]):
                        # x,y,w,h,id,occl,1 walk_person 0 ride_person
                        fobj.write(('%d %d %d %d %d %d %d; ' % (rec[0]-104,rec[1]-105,rec[2],rec[3],box['id'],box['occl'],walk_person)))
                fobj.writelines(']\n')
            print('set_num:%d;v_num%d' % (set_num,v_num))
            fobj.flush()
            fobj.close()
if __name__ == '__main__':
    vbbspath = "/home/all/datasets/SCUT_FIR_101/annotations_vbb"
    outpath = '/home/yangsong/workspace/spyder_workspace/sate_eval/data/ann_filtered/scut_result'
    vbbs = utils.load_vbbs(vbbspath)
    write_annbbox(vbbs,outpath)
    print 'finished'