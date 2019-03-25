from pylab import *


RED_COLOR = '\x1b[31m'
BLACK_COLOR = '\x1b[0m'

#print('Black color '+RED_COLOR+'red color '+BLACK_COLOR+'black color!')
#print('Again black color')



def get_distinct_color(i):
    mnColors=np.array([
                [0,0,255],      # 0
                [255,0,0],      # 1
                [0,255,0],      # 2
                [178,178,178],  # 3
                [0,255,255],    # 4
                [255,0,255],    # 5
                [36,185,137],   # 6
                [128,128,255],  # 7
                [237,166,18],   # 8
                [204,208,27],   # 9
                [120,120,120],    # 10
                [160,116,95],   # 11
                ], dtype=np.float)

    mnColors=mnColors/255

    if i<mnColors.shape[0]:
        return mnColors[i,:]
    else:
        return np.random.rand(3)



def get_gradient_color(i,MaxColor):
    '''Get gradient color.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    

    x=i/MaxColor
    blue = min(max(4*(0.75-x), 0.), 1.)
    red  = min(max(4*(x-0.25), 0.), 1.)
    #green= min(max(4*(x-0.75), 0.), 1.)
    green= min(4*max((x-0.5)**2, 0.), 1.)
    return [red,green,blue]
    #return [min(1,i*4/MaxColor),max(0,1.9*i/MaxColor-0.9),0]