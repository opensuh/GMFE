import os
import numpy as np
import cv2


def scatterPlot(x, y, gmin, gmax, size=1000):
    ''' 
    make scatter plot image
    args:
        x = channel 1 input list or 1D-array
        y = channel 2 input list or 1D-array
        gmin = minimum value to draw for pixel (1,1)
        gmax = maximum value to draw for pixel (size, size)
        size = image size = (size, size)
    return:
        splot = scatter plot image (numpy.ndarray)
    '''
    if type(x)==np.ndarray and type(y)==np.ndarray:
        pass
    elif type(x)==list and type(y)==list and len(x)*len(y)>0:
        x, y = np.array(x), np.array(y)
    else:
        print('ERROR: invalid type input x,y!\n       x,y must be list or numpy.ndarray')
        return None

    if gmin >= gmax:
        print('ERROR: invalid min, max bounds')
        return None

    splot = np.zeros([size, size])
    # normalize x and y
    x, y = x-gmin, y-gmin
    x, y = x/(gmax-gmin), y/(gmax-gmin)
    x, y = x*size, y*size
    x, y = x.astype(int), y.astype(int)
    cnt = 0
    for i in range(len(x)):
        if x[i]>=0 and y[i]>=0 and x[i] < size and y[i] < size:
            splot[x[i], y[i]] += 1
            cnt+=1
        
    splot /= splot.max() 
    splot *= 255 
    splot = np.uint8(splot)
    return splot, cnt


def toRGB(red, green, blue):
    ''' 
    Create image using R,G,B values of raw data that has passed the band pass filter
    args:
        red = Red value of raw data extracted through bandpass filter
        green = Green value of raw data extracted through bandpass filter
        blue = Blue value of raw data extracted through bandpass filter
    return:
        image = Generated NSP image result
    '''
    if len(red) != len(green) or len(red) != len(blue):
        print('ERROR: different color channel image size')
        return None
    image = np.empty((len(red), len(red), 3), dtype=np.uint8)
    image[:,:,0] = red
    image[:,:,1] = green
    image[:,:,2] = blue
    return image
