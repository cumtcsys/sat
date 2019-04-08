import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_importance():
    pt = get_data()
    f, (ax1,ax2) = plt.subplots(figsize = (3,7),nrows=2)
     
    # cmap用cubehelix map颜色
    cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    sns.heatmap(pt, linewidths = 0.05, ax = ax1, vmax=np.amax(pt), vmin=np.amin(pt), cmap=cmap)
    ax1.set_title('cubehelix map')
    ax1.set_xlabel('')
    ax1.set_xticklabels([]) #设置x轴图例为空值
    ax1.set_ylabel('kind')
    # cmap用matplotlib colormap
    sns.heatmap(pt,annot=True, linewidths = 0.05, ax = ax2, vmax=np.amax(pt), vmin=np.amin(pt), cmap='rainbow') 
    # rainbow为 matplotlib 的colormap名称
    ax2.set_title('cell importance --far')
    ax2.set_xlabel('width')
    ax2.set_ylabel('height')
    
#    f, (ax1, ax2) = plt.subplots(figsize=(10,4),nrows=2)
#    sns.heatmap(pt, annot=True, ax=None,linewidths = 0.05, vmax=np.amax(pt), vmin=np.amin(pt), cmap='rainbow')
#    sns.heatmap(pt, annot=True, ax=ax2, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
    
def get_data():
    path = './data/cellimportanc.txt'
    fobj = open(path,'r')
    data = []
    block = []
    CELL_ROW = 8
    CELL_COL = 4
    importance = np.zeros((CELL_ROW,CELL_COL))

    for ii,line in enumerate(fobj.readlines()):
        line = line.strip()
        value = int(line)#特征在模型中出现的次数
        block.append(value)
        if (ii + 1) % 4 == 0:
            data.append(block)#将特征以block（36维）进行划分
            block=[]
    for row in range(CELL_ROW):
        for col in range(CELL_COL):
            if row == 6 and col == 3:
                print('debug')
            print('row:%d col:%d'%(row,col))
            if row == 0 and col == 0 :#左上角cell
                importance[row,col] = data[0][0]
            elif row == 0 and col == CELL_COL - 1:#右上角cell
                importance[row,col] = data[col - 1][1]
            elif row == CELL_ROW - 1 and col == 0:#左下角cell
                importance[row,col] = data[(row-1)*(CELL_COL-1)+col][2]
            elif row == CELL_ROW - 1 and col == CELL_COL - 1:#右下角cell
                importance[row,col] = data[(row-1)*(CELL_COL-1)+col-1][3]
            elif row == 0:#第一行cell只会重复一次
                importance[row,col] = max(data[col-1][1],data[col][0])
            elif row == CELL_ROW - 1:#最后一行cell只会重复一次
                importance[row,col] = max(data[(row-1)*(CELL_COL-1)+col-1][3],data[(row-1)*(CELL_COL-1)+col][2])
            elif col == 0:#第一列cell只会重复一次
                importance[row,col] = max(data[(row-1)*(CELL_COL-1)][2],data[(row)*(CELL_COL-1)][0])
            elif col == CELL_COL - 1:#最后一列cell只会重复一次
                importance[row,col] = max(data[(row)*(CELL_COL-1)-1][3],data[(row+1)*(CELL_COL-1)-1][1])
            else:#中间cell会重复4次
                importance[row,col] = max(data[(row-1)*(CELL_COL-1)+col-1][3]
                                                ,data[(row-1)*(CELL_COL-1)+col][2]
                                                ,data[(row)*(CELL_COL-1)+col-1][1]
                                                ,data[(row)*(CELL_COL-1)+col][0]
                                                )
    return importance
                
    return data
plot_importance()