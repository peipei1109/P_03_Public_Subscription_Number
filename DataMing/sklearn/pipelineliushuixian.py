# -*- encoding: utf-8 -*-
'''
Created on 2016年5月22日

@author: LuoPei
'''

from numpy import log1p
from sklearn.preprocessing import Imputer,OneHotEncoder,FunctionTransformer, Binarizer,MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

#新建计算缺失值的对象
step1 =('Imputer',Imputer())
#新建将部分特征矩阵进行定性特征编码的对象
step2_1=('OneHotEncoder',OneHotEncoder(sparse=False))
#新建将部分特征矩阵进行对数函数转换的对象
step2_2=('ToLog',FunctionTransformer(log1p))

#新建将部分特征矩阵进行二值化类的对象
step2_3=('ToBinary',Binarizer())

#新建部分并行处理对象，返回这为每个并行工作得输出的合并
step2=('FeatureUnion',FeatureUnion(transformer_list=[step2_1,step2_2,step2_3],idx_list=[[0],[1,2,3],[4]]))


#新建无量纲化对象
step3=('MinMaxScaler',MinMaxScaler())

#新建卡方校验选择特征的对象
step4=('SelectKBest',SelectKBest(chi2,k=3))

#新建PCA 降维的对象
step5=('PCA',PCA(n_components=2))


#新建逻辑回归的对象，其为待训练的模型作为流水线的最后一步
step6=('LogisticRegression',LogisticRegression(penalty='12'))
#新建流水线处理对象
#参数steps 为需要流水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
pipeline = Pipeline(steps=[step1,step2,step3,step4,step5,step6])

if __name__=="__main__":
    pass