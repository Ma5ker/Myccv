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


#求出对应级数下的二值图像
def getBImg(img,n):
    img1=img.copy()
    #当像素值为n时其对应处为255，否则为0，传入矩阵比较得到对应二值图
    if(n==0):
        #如果要求的是0，则将0变255，其他变0
        img1[img1!=n]=255
        img1 = 255-img1
    else:
        #如果要求的是其他阶，则将除本阶外的所有阶设为0
        img1[img1!=n] = 0
        img1[img1==n] = 255
    return img1

def getCCV(src,tau,n):
    img = src.copy()
    row,col,channels =img.shape
    #高斯平滑滤波
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #量化后形成一维向量
    img = QuantizeColor(img, n)
    #三通道分解
    bgr = cv2.split(img)
    #阈值根据阈值得到阈值面积占比   
    tau = row*col * tau
    #设置alpha 和 beta存放聚合与非聚合量,3表示三个通道，通道分离顺序为BGR
    alpha = np.zeros((3,n))
    beta = np.zeros((3,n))
    for i,ch in enumerate(bgr):
        for j in range(0,n):
            #取二值图
            th=getBImg(ch,j)
            #二值图连通处理
            ret, labeled, stat, centroids = cv2.connectedComponentsWithStats(th, None, None, None, connectivity=8)
            #返回结果参数含义
            #ret 标签数量   不需要
            #labeled 标签矩阵   不需要
            #stat统计矩阵:
            # cv2.CC_STAT_LEFT最左边的（x）坐标，它是水平方向上包围框的包含开始。
            # cv2.CC_STAT_TOP最上面的（y）坐标，它是垂直方向上边界框的包含开始。
            # cv2.CC_STAT_WIDTH边界框的水平大小
            # cv2.CC_STAT_HEIGHT边界框的垂直大小
            # cv2.CC_STAT_AREA连接组件的总面积（以像素为单位）
            #centroids 质心矩阵   不需要
            #赋值给两个矩阵
            #areas为统计矩阵的序号及面积
            #coord为统计矩阵的左上角坐标 判断矩阵的像素类型
            areas = [[v[4], label_idx] for label_idx, v in enumerate(stat)]# [index,area]
            coord = [[v[0], v[1]] for label_idx, v in enumerate(stat)]# [index,[x,y]]
            for a, c in zip(areas, coord):#area->a     coord -> c
                #分区大小
                area_size = a[0]
                #分区左上角坐标
                x, y = c[0], c[1]
                #判断是否越边界
                if (x < th.shape[1]) and (y < th.shape[0]): 
                    if index!=0:
                        bin_idx=j#当前级数
                        chnl=i #当前通道
                        if area_size >= tau:
                            #如果大于阈值面积，此区域算作聚合
                            #并对连通域中像素值相同的进行累加
                            alpha[chnl,bin_idx] = alpha[chnl,bin_idx] + area_size
                        else:
                            #否则，此区域算作非聚合
                            beta[chnl,bin_idx] = beta[chnl,bin_idx] + area_size
    return alpha,beta

if __name__ == '__main__':
    #参数
    n = 16
    tau = 0.001
    #alpha, beta=getCCV(img,tau,n)
