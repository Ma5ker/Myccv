# -*- coding:utf-8 -*-
import sys
import numpy as np
import cv2
import os
import pandas as pd


def QuantizeColor(src):
    '''
    对RGB分散通道，均分成4等级，得到 4*4*4 = 64 个等级进行统计
    '''
    img = src.copy()
    B,G,R = cv2.split(img)
    div = 64   #256//4
    vf = np.vectorize(lambda x, div: int(x//div))
    quantized_b = vf(B, div)
    quantized_g = vf(G, div)
    quantized_r = vf(R, div)
    img = 16 * quantized_b + 4 * quantized_g + quantized_r
    return img

def getTZ(src):
    '''
    参考博客：https://blog.csdn.net/dyx810601/article/details/50520243
    '''
    img =src.copy()
    #高斯平滑滤波
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #量化,16*R + 4*G + B表示为一个一维向量
    img = QuantizeColor(img)
    #设置一个各个值的像素个数数组
    result = [0 for i in range(64)]
    for i in range(64):
        #统计每个等级的元素数
        flag = img ==i
        tar = img [flag]
        result[i] = tar.size
    return result

if __name__ == '__main__':
    filefolder = r"C:\Users\RaspB\Desktop\test"
    assert os.path.exists(filefolder)
    assert os.path.isdir(filefolder)
    #图像路径列表,用后缀jpg判断是否为图片
    imagelist=[ filefolder+'\\'+filename for filename in os.listdir(filefolder) if filename.endswith('jpg') ]
    for imagepath in imagelist:
        img=cv2.imread(imagepath)
        #RGB三个通过B*16+ G *4 + R
        #得到统计的颜色直方图的64个等级的数据
        result=getTZ(img)
