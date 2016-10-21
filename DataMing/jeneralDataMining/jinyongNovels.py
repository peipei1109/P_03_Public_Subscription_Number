# -*- encoding: utf-8 -*-
'''
Created on 2016年7月4日

@author: LuoPei
'''
#http://mp.weixin.qq.com/s?__biz=MzA3MDg0MjgxNQ==&mid=2652389850&idx=1&sn=52ef6914d5d299e0306af3c123886071&scene=4#wechat_redirect
#用Python对金庸系列武侠小说进行文本挖掘
from __future__ import unicode_literals  #因为涉及中文字符，所以我们使用 __future__ 中 Python 3 的特性，将所有的字符串转为 unicode。
from matplotlib.font_manager import FontProperties
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import gensim
import jieba



#下面是一段测试代码，在可视化中加上英文标注~~~
font_peipei_consolas = FontProperties(fname="STSONG.TTF")
x = range(10)
plt.plot(x)
plt.title(u"中文",fontproperties=font_peipei_consolas, fontsize=14)#记得写中文的时候前面加上u，或者加上from __future__ import unicode_literals
plt.show()
with open('names.txt') as f:
    # 去掉结尾的换行符
    data = [line.strip().decode('utf8') for line in f.readlines()]

novels = data[::2]
names = data[1::2]

novel_names = {k: v.split() for k, v in zip(novels, names)}

for name in novel_names['天龙八部'][:20]:
    print name
    
def find_main_charecters(novel, num=10):
    with open('Novels/{}.txt'.format(novel)) as f:
        data = f.read().decode('utf-8')
    count = []
    for name in novel_names[novel]:
        count.append([name, data.count(name)])
    count.sort(key=lambda x: x[1])
    _, ax = plt.subplots()
    
    numbers = [x[1] for x in count[-num:]]
    names = [x[0] for x in count[-num:]]
    ax.barh(range(num), numbers, color='red', align='center')
    ax.set_title(novel, 
                 fontsize=14, 
                 fontproperties=font_peipei_consolas)
    ax.set_yticks(range(num))
    ax.set_yticklabels(names, 
                       fontsize=14,
                       fontproperties=font_peipei_consolas)
    plt.show()
find_main_charecters("天龙八部") 

#jieba 包具有一定的识别新词的能力，不过为了得到更准确的分词结果，我们可以将人名导入 jieba 库的字典，除此之外，我们还加入门派和武功的专有名词：
for _, names in novel_names.iteritems():
    for name in names:
        jieba.add_word(name)
with open("kungfu.txt") as f:
    kungfu_names = [line.decode('utf8').strip() 
                    for line in f.readlines()]
with open("bangs.txt") as f:
    bang_names = [line.decode('utf8').strip() 
                  for line in f.readlines()]

for name in kungfu_names:
    jieba.add_word(name)

for name in bang_names:
    jieba.add_word(name)   
    
#按照行来处理文本，进行分词
novels = ["书剑恩仇录", 
          "天龙八部",
          "碧血剑",
          "越女剑",
          "飞狐外传",
          "侠客行",
          "射雕英雄传",
          "神雕侠侣",
          "连城诀",
          "鸳鸯刀",
          "倚天屠龙记",
          "白马啸西风",
          "笑傲江湖",
          "雪山飞狐",
          "鹿鼎记"]

sentences = []

for novel in novels:
    print "处理：{}".format(novel)
    with open('novels/{}.txt'.format(novel)) as f:
        data = [line.decode('utf8').strip() 
                for line in f.readlines() 
                if line.decode('utf8').strip()]
    for line in data:
        words = list(jieba.cut(line))
        sentences.append(words)
        
        
        
#使用 gensim 中的默认参数进行训练：

model = gensim.models.Word2Vec(sentences, 
                               size=100, 
                               window=5, 
                               min_count=5, 
                               workers=4)

#首先看与乔峰相似的人：
for k, s in model.most_similar(positive=["乔峰", "萧峰"]):
    print k, s

#再看看与阿朱相似的人：
for k, s in model.most_similar(positive=["阿朱"]):
    print k, s
    
#除了人物，我们可以看看门派：
for k, s in model.most_similar(positive=["丐帮"]):
    print k, s
    
#武功：
for k, s in model.most_similar(positive=["降龙十八掌"]):
    print k, s

#在 Word2Vec 的模型里，有过“中国-北京=法国-巴黎”的例子，这里我们也可以找到这样的例子：

def find_relationship(a, b, c):
    """
    返回 d 
    a与b的关系，跟c与d的关系一样    
    """
    d, _ = model.most_similar(positive=[c, b], negative=[a])[0]
    print "给定“{}”与“{}”，“{}”和“{}”有类似的关系".format(a, b, c, d)

find_relationship("段誉", "段公子", "乔峰")

#类似的：
# 情侣对
find_relationship("郭靖", "黄蓉", "杨过")
# 岳父女婿
find_relationship("令狐冲", "任我行", "郭靖")
# 非情侣
find_relationship("郭靖", "华筝", "杨过")

#以及，小宝你是有多爱康熙：
# 韦小宝
find_relationship("杨过", "小龙女", "韦小宝")
find_relationship("令狐冲", "盈盈", "韦小宝")
find_relationship("张无忌", "赵敏", "韦小宝")


#除了人物之间的关系，还可以看看人物与门派武功之间的关系：
find_relationship("郭靖", "降龙十八掌", "黄蓉")
find_relationship("武当", "张三丰", "少林")
find_relationship("任我行", "魔教", "令狐冲")

'''
 之前我们对文本进行 Word2Vec 的结果，是将一个中文词组，映射到了一个向量空间，因此，我们可以利用这个向量表示的空间，对这些词进行聚类分析。

因为全部小说中的人物太多，我们考虑从单本小说进行入手，先把天龙八部中的人物的词向量拿出来：
'''
all_names = []

word_vectors = None

for name in novel_names["天龙八部"]:
    if name in model:
        all_names.append(name)
        if word_vectors is None:
            word_vectors = model[name]
        else:
            word_vectors = np.vstack((word_vectors, model[name]))
            all_names = np.array(all_names)


#聚类我们可以使用很多方法，这里我们先考虑 Kmeans：
from sklearn.cluster import KMeans

N = 3

label = KMeans(N).fit(word_vectors).labels_

for c in range(N):
    print "\n类别{}：".format(c+1)
    for idx, name in enumerate(all_names[label==c]):
        print name,
        if idx % 10 == 9:
            print 
    print

#我们把众龙套去掉，再聚一次：  
N = 4

c = sp.stats.mode(label).mode

remain_names = all_names[label!=c]
remain_vectors = word_vectors[label!=c]
remain_label = KMeans(N).fit(remain_vectors).labels_

for c in range(N):
    print "\n类别{}：".format(c+1)
    for idx, name in enumerate(remain_names[remain_label==c]):
        print name,
        if idx % 10 == 9:
            print 
    print


#换一本小说：

# all_names = []
# word_vectors = None
# for name in novel_names["倚天屠龙记"]:
#     if name in model:
#         all_names.append(name)
#         if word_vectors is None:
#             word_vectors = model[name]
#         else:
#             word_vectors = np.vstack((word_vectors, model[name]))
#             all_names = np.array(all_names)
#     
#这次采用层级聚类的方式，调用的是 Scipy 中层级聚类的包：
import scipy.cluster.hierarchy as sch

Y = sch.linkage(word_vectors, method="ward")

_, ax = plt.subplots(figsize=(10, 40))

Z = sch.dendrogram(Y, orientation='right')
idx = Z['leaves']

ax.set_xticks([])
ax.set_yticklabels(all_names[idx], 
                  fontproperties=font_peipei_consolas)
ax.set_frame_on(False)

plt.show()

#除了人物，我们还可以考虑对武功进行聚类分析：
all_names = []

word_vectors = None

for name in kungfu_names:
    if name in model:
        all_names.append(name)
        if word_vectors is None:
            word_vectors = model[name]
        else:
            word_vectors = np.vstack((word_vectors, model[name]))
            
all_names = np.array(all_names)

Y = sch.linkage(word_vectors, method="ward")

_, ax = plt.subplots(figsize=(10, 35))

Z = sch.dendrogram(Y, orientation='right')

idx = Z['leaves']

ax.set_xticks([])

ax.set_yticklabels(all_names[idx], 
                   fontproperties=font_peipei_consolas)

ax.set_frame_on(False)

plt.show()


#最后是门派的聚类
all_names = []

word_vectors = None

for name in bang_names:
    if name in model:
        all_names.append(name)
        if word_vectors is None:
            word_vectors = model[name]
        else:
            word_vectors = np.vstack((word_vectors, model[name]))
            
all_names = np.array(all_names)

Y = sch.linkage(word_vectors, method="ward")

_, ax = plt.subplots(figsize=(10, 25))

Z = sch.dendrogram(Y, orientation='right')

idx = Z['leaves']

ax.set_xticks([])

ax.set_yticklabels(all_names[idx], 
                   fontproperties=font_peipei_consolas)

ax.set_frame_on(False)

plt.show()



