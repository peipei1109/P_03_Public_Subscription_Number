# -*- encoding: utf-8 -*-
'''
Created on 2016年7月5日

@author: LuoPei
'''
from sklearn.feature_extraction import DictVectorizer  


measurements = [  
    {'city': 'Dubai', 'temperature': 33.},  
    {'city': 'London', 'temperature': 12.},  
    {'city': 'San Fransisco', 'temperature': 18.},  
    ]  


vec = DictVectorizer() 
vec_array=vec.fit_transform(measurements).toarray()
print vec_array  
 
featruenames=vec.get_feature_names()  
print featruenames

measurements = [  
    {'city=Dubai': True, 'city=London': True, 'temperature': 33.},  
    {'city=London': True, 'city=San Fransisco': True, 'temperature': 12.},  
    {'city': 'San Fransisco', 'temperature': 18.},]

vec_array=vec.fit_transform(measurements).toarray()  
print vec_array

featruenames=vec.get_feature_names()  
print featruenames

D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]  
v = DictVectorizer(sparse=False) 
X = v.fit_transform(D)  
print X
print v.transform({'foo': 4, 'unseen_feature': 3})

print  v.transform({'foo': 4})  

print v.fit_transform({'foo': 4, 'unseen_feature': 3})  

#可见当使用transform之后，后面的那个总是可以实现同前面的一个相同的维度。当然这种追平可以是补齐，也可以是删减，所以通常，我们都是用补齐短的这样的方式来实现维度一致。如果你不使用transform，而是继续fit_transform，则会得到下面的结果（这显然不能满足我们的要求）
feature_dicts_tra={}
feature_dicts_dev={}
labels_t=[]
labels_d=[]
vec = DictVectorizer()  
sparse_matrix_tra = vec.fit_transform(feature_dicts_tra)  
sparse_matrix_dev = vec.transform(feature_dicts_dev)

from sklearn import linear_model  
  
logreg = linear_model.LogisticRegression(C = 1)  
logreg.fit(sparse_matrix_tra, labels_t)  
prediction = logreg.predict(sparse_matrix_dev)  
print(logreg)  
print("accuracy score: ")  
print(logreg.accuracy_score(labels_d, prediction))  
print(logreg.classification_report(labels_d, prediction))  
