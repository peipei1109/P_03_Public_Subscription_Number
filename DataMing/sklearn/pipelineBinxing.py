# -*- encoding: utf-8 -*-
'''
Created on 2016年5月22日

@author: LuoPei
'''

#整体并行
from numpy import log1p
# from numpy import log
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import FeatureUnion

#新建将整体特征矩阵进行对数函数转换的对象
# step2_1=('ToLog',FunctionTransformer(loglp))
step2_1=('ToLog',FunctionTransformer(log1p))
#新建将整体特征矩阵进行二值化类的对象
step2_2=('ToBinary',Binarizer())

#新建整体并行处理对象

#该对象也有fit和transform 方法，fit和transform 方法均是并行地调用需要并行处理的对象的fit和transform 方法
#参数transformer_list为需要并行处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
step2=('FeatureUnion',FeatureUnion(transformer_list=[step2_1,step2_2]))




if __name__=="__main__":
    pass