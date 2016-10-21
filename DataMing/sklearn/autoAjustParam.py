# -*- encoding: utf-8 -*-
'''
Created on 2016年5月22日

@author: LuoPei
'''
from sklearn.grid_search import GridSearchCV
from pipelineliushuixian import *
from sklearnlearn import *
from sklearn.externals.joblib import dump,load
#新建网格搜索对象
#第一参数为待训练的模型
#param_grid为待调参数组成的网格，字典格式，键为参数名称（格式“对象名称__子对象名称__参数名称”),值为可取的参数值列表

grid_search=GridSearchCV(pipeline,param_grid={'FeatureUnion_ToBinary_threshold':[1.0,2.0,3.0,4.0],'LogisticRegression_C':[0.1,0.2,0.4,0.8]})

#训练以及调参
grid_search.fit(iris.data, iris.target)



#持久化
#持久化数据
#第一个参数为内存中的对象
#第二个参数为保存在文件系统中的名称
#第三个参数为压缩级别，0为不压缩，3位合适压缩级别
dump(grid_search,'grid_search.dump',compress=3)
#从文件系统中夹杂数据到内存中
grid_search=load('grid_search.dump')
if __name__=="__main__":
    pass