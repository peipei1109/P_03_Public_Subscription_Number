# -*- encoding: utf-8 -*-
'''
Created on 2016年5月22日

@author: LuoPei
'''
#http://mp.weixin.qq.com/s?__biz=MzA3MDg0MjgxNQ==&mid=2652389709&idx=1&sn=8845b9e32a3ad734f6ebe3ca6402fa4b&scene=4#wechat_redirect
from numpy import hstack,vstack,array,median, nan
from numpy.random import choice
from sklearn import datasets



#特征矩阵加工
#使用vstack 增加一行含缺失值的样本（nan,nan,nan,nan）
#使用hstack 增加一列花的颜色（0-白，1-黄，2-红），花的颜色是随机的，意味着颜色并不影响花的分类

iris = datasets.load_iris()
iris.data=hstack((choice([0,1,2],size=iris.data.shape[0]+1).reshape(-1,1),vstack((iris.data,array([nan,nan,nan,nan]).reshape(1,-1)))))

#目标值向量加工
#增加一个目标值，对应含缺失值的样本，值为众数
iris.tartget=hstack((iris.target,array([median(iris.target)])))

if __name__=="__main__":
    pass