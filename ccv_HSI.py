# -*- coding:utf-8 -*-
import sys
import numpy as np
import cv2
import os
import pandas as pd

def RGB2HSI(imgRGB):
    '''
    参考：https://blog.csdn.net/lsh_2013/article/details/45245865
         https://blog.csdn.net/lwplwf/article/details/77494072
    '''
    img = imgRGB.copy()
    row,col,channels =img.shape
    #三通道分解
    B,G,R=cv2.split(img)
    #归一化
    [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
    imgHSI = img.copy()
    for i in range(row):
        for j in range(col):
            fenzi = 0.5*((R[i,j]-G[i,j])+(R[i,j]-B[i,j]))
            fenmu = np.sqrt((R[i, j]-G[i, j])**2+(R[i, j]-B[i, j])*(G[i, j]-B[i, j]))
            sum = R[i,j]+G[i,j]+B[i,j]
            I=sum/3.0
            #如果R=G=B,S=0,H=0
            if fenmu ==0 :
                H = 0
                S = 0
            else:
                #计算S
                if sum == 0:
                    S=0
                else:
                    minRGB = min(min(R[i,j],G[i,j]),B[i,j])
                    S = 1-3*minRGB/sum
                #计算H
                theta =  float(np.arccos(fenzi /fenmu))
                if B[i,j] < G[i,j]:
                    H = theta
                else:
                    H = 2*np.pi - theta
                H = H / 2*np.pi
            imgHSI[i,j]=H*255,S*255,I*255
    return imgHSI 

def QuantizeColor(src):
    '''
    量化,并用G = 9H + 3S + I计算，H 8等分，S 3等分，I 
    '''
    img = src.copy()
    H,S,I = cv2.split(img)
    #H通道
    div = 32
    vf = np.vectorize(lambda x, div: int(x//div))
    quantized_h = vf(H, div)
    #S通道
    div = 86
    vf = np.vectorize(lambda x, div: int(x//div))
    quantized_s = vf(S, div)
    #I通道
    div = 86
    vf = np.vectorize(lambda x, div: int(x//div))
    quantized_i = vf(I, div)
    #G = 9H + 3S + I
    imgG = 9 * quantized_h + 3 * quantized_s + quantized_i
    return imgG

#将图像中指定元素值的元素作为白色形成二值图
def getBImg(img,n):
    img1=img.copy()
    #当像素值为n时其对应处为255，否则为0，传入矩阵比较得到对应二值图
    if(n==0):
        #如果要求的是0，则将0变255，其他变0
        img1[img1!=n]=255
        img1= 255-img1
    else:
        #如果要求的是其他阶，则将除本阶外的所有阶设为0
        img1[img1!=n] = 0
        img1[img1==n] = 255
    return img1


def getCCV(src,tau):
    img =src.copy()
    row,col,channels =img.shape
    tau = tau * row * col
    cv2.imshow('RGB',src)
    #转HSI
    img = RGB2HSI(img)
    cv2.imshow('HSI',img)
    cv2.waitKey(0)
    #高斯平滑滤波
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #量化，并返回计算好的G
    imgG = QuantizeColor(img)
    #创建alpha和beta的列表
    alpha = [0 for i in range(72)]
    beta = [0 for i in range(72)]
    #总共有72种G值
    for i in range(72):
        #求此G值的聚合与非聚合面积
        #先转为当前值的二值图
        th=getBImg(imgG,i).astype(np.uint8)
        #二值图连通处理
        ret, labeled, stat, centroids = cv2.connectedComponentsWithStats(th, None, None, None, connectivity=8)
        areas = [[v[4], label_idx] for label_idx, v in enumerate(stat)]# [index,area]
        coord = [[v[0], v[1]] for label_idx, v in enumerate(stat)]# [index,[x,y]]
        for a, c in zip(areas, coord):#area->a     coord -> c
            #分区大小
            area_size = a[0]
            #分区左上角坐标
            x, y = c[0], c[1]
            #判断是否越边界
            if (x < th.shape[1]) and (y < th.shape[0]):
                #背景判断
                if index!=0:
                    #阈值判断
                    if area_size >= tau:
                        alpha[i] += area_size
                    else:
                        beta[i] += area_size
    return alpha,beta





if __name__ == '__main__':
    #阈值
    tau=0.001
    #alpha,beta=getCCV(img,tau)
