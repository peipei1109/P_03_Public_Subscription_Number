# -*- encoding: utf-8 -*-
'''
Created on 2016年5月22日

@author: LuoPei
'''
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion

#新建将部分特征矩阵进行定性特征编码的对象
step2_1=('OneHotEncoder',OneHotEncoder(sparse=False))

#新建将部分特征矩阵进行对数函数转换的对象
step2_2=('ToLog',FunctionTransformer(log1p))

#新建将部分特征矩阵进行二值化类的对象
step2_3=('ToBinary',Binarizer())

#新建部分滨兴处理对象
#参数transform_list 为需要并行处理的对象李彪，该列表为二元组列表，第一元为对象名称，第二元为对象
#参数idx_list 为相应的需要读取的特征矩阵的列

step2=('FeatureUnionExt',FeatureUnion(transformer_list=[step2_1,step2_2,step2_3],idx_list=[[0],[1,2,3],[4]]))


if __name__=="__main__":
    pass