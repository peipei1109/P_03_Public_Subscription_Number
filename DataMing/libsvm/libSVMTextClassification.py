# -*- encoding: utf-8 -*-
'''
Created on 2016年7月5日

@author: LuoPei
'''
import os
import os.path
import jieba
import math


#读取数据，把gb2312编码的数据转成utf-8
def trans_to_utf8(rootdir):
    for parent,dirnames,files in os.walk(rootdir):
        print "".join(["Reduced_utf8\\",parent])
        if not os.path.exists("".join(["Reduced_utf8\\",parent])):
            os.makedirs("".join(["Reduced_utf8\\",parent]))
        for filename in files: 
            data=''           
            #print "the full name of the file is:" +os.path.join(parent,filename)
            with open(os.path.join(parent,filename),'rb') as f:
                try:
                    data=f.read().decode("gb2312").encode("utf-8")
                except UnicodeDecodeError:
                    #print u'文档中有  非 gb2312的字符，跳过~~', 有些文档中有非gb2312编码的字符，这种文档我们直接跳过去~
                    continue   
            with open(os.path.join("Reduced_utf8\\",parent,filename),'wb') as f:
                f.write(data)
                 
#中文分词，所有的文档~
def seg(rootfile):
    sentences=[]
    for parent,dirnames,files in os.walk(rootfile):
        for filename in files:
            with open(os.path.join(parent,filename),'rb') as f:
                data = [line.decode('utf8').strip() 
                        for line in f.readlines() 
                        if line.decode('utf8').strip()]
            for line in data:
                words = list(jieba.cut(line))
                sentences.extend(words)
    return  sentences           
          
#ignore some term  
def ignore(s):
    return s == 'nbsp' or s == ' ' or s == ' ' or s == '/t' or s == '/n' or s == '，' or s == '。' or s == '！' or s == '、' or s == '―'  or s == '？'  or s == '＠' or s == '：'   or s == '＃' or s == '%'  or s == '＆'    or s == '（' or s == '）' or s == '《' or s == '》'   or s == '［' or s == '］' or s == '｛' or s == '｝'   or s == '*' or s == ',' or s == '.'  or s == '&'   or s == '!' or s == '?' or s == ':' or s == ';' or s == '-' or s == '&'  or s == '<' or s == '>' or s == '(' or s == ')'  or s == '[' or s == ']' or s == '{' or s == '}'    


#add by myself
#中文分词，对单个文档进行
def seg_singlefile(file):
    segments=[]
    with open(file) as f:
        data = [line.decode('utf8').strip()
                for line in f.readlines() 
                if line.decode('utf8').strip()]
        for line in data:
            words = list(jieba.cut(line))
            segments.extend(words)
    return segments 

#自己写的一个小函数，统计每个类中，每个词出现的[文章数，次数]，格式为{"label1":{"item1":[文章数，总次数]，...}，"label2":{"item1":[文章数，总次数]，...}}
#并统计出每一类中，每一个文档的词-词频的字典，具体格式是{"label1":[{},{},{}],"lable2":[{},{},{}]}

def stat_class_item_bymyself(rootdir):
    class_fileCount={}
    class_item={} #{"label1":{"item1":[文章数，总次数]，...}，"label2":{"item1":[文章数，总次数]，...}}
    class_eachfile_item={}
    parent_list=[parent for parent,_,_ in os.walk(rootdir)]
    for index,parent in enumerate(parent_list[1:]):
        print "parent:", parent
        class_item[index]={} ;item_doc_count={}   #初始化每一个类的{"item1":[文章数，总次数]，...}  
        class_eachfile_item[index]=[]; file_item=[]   
        for _,_,filenames in os.walk(parent):
            print "filenames: " ,len(filenames)
            class_fileCount[index]=len(filenames)
            for filename in filenames:
                item_frequncey={}
                item_list=seg_singlefile("".join([parent,"\\"+filename]))
                item_list=[item for item in item_list if not ignore(item)]
                for item in set(item_list):
                    item_frequncey[item]=item_list.count(item)
                    item_value=item_doc_count.get(item)
                    #可以考虑吧[文章数，总次数]封装成一个类，如果是java的话，pthon的话，日后想想怎么简化该if else
                    if item_value:
                        item_doc_count[item][0] +=1  #出现的文章次数+1
                        item_doc_count[item][1] += item_list.count(item) #出现的词的总次数+item_list.count(item)
                    
                    else:
                        item_doc_count[item]=[]
                        item_doc_count[item].append(1) #出现的文章次数初始化为1
                        item_doc_count[item].append(item_list.count(item)) ##出现的词的总次数初始化为item_list.count(item)
                file_item.append(item_frequncey)
        class_item[index]= item_doc_count
        class_eachfile_item[index]= file_item 
            
    return class_item,class_eachfile_item,class_fileCount    



#在class_item的基础上，统计item_class
#{"item1":{"label1":文章数，"label2":文章数}，"item2":{"label1":文章数，"label2":文章数}}
def stat_item_class_bymyself(class_item):
    item_class={}
    for label in class_item.keys():
        for item, value_list in class_item[label].iteritems():
            item_class[item]=item_class.get(item)
            if item_class[item]:
                item_class[item][label]=class_item[label][item][0]
            else:
                item_class[item]={}
                item_class[item][label]=class_item[label][item][0]
    return item_class

 
#每一个类分开算的~对这个类算整体的tfidf 还没有合并特征~~~
def tfidf_class(class_item,class_fileCount):
    class_item_tf_idf={}
    print 'begin compute tf *idf'
    for label,item_dic in class_item.iteritems():
        totalFrequency=0
        item_tf_idf={}
        for item,item_value in item_dic.iteritems():
            totalFrequency +=item_value[1]
        for item,item_value in item_dic.iteritems():
            tf=item_value[1]+1/float(totalFrequency+len(item_dic.keys())) #加1平滑
            idf=math.log(class_fileCount[label]/float(item_value[0]+1))  #加1平滑 
            item_tf_idf[item]=tf*idf                        
        class_item_tf_idf[label]=item_tf_idf    
    
    return  class_item_tf_idf

#根据每个文档进行tf*idf计算
def tfidf_eachfile(item_times_frequncy,item_doc_frequency,fileCount):
   
    totalFrequency=0
    item_tf_idf={}
    for item,item_value in item_times_frequncy.iteritems():
        totalFrequency +=item_value
    for item,item_value in item_times_frequncy.iteritems():
        tf=(item_value+1)/float(totalFrequency+len(item_times_frequncy.keys())) #加1平滑
        idf=math.log(fileCount/float(item_doc_frequency[item][0]+1))  #加1平滑 
        item_tf_idf[item]=tf*idf    
    return  item_tf_idf


#计算某个类所有文档的
def tfidf_single_class(all_item_times_frequncy,item_doc_frequency,fileCount):
    tfidf_list=[]
    for item_times_frequncy in all_item_times_frequncy:
        item_tf_idf=tfidf_eachfile(item_times_frequncy,item_doc_frequency,fileCount)
        tfidf_list.append(item_tf_idf)
    return tfidf_list

#计算所有类的所有文档

def tfidf_all_class(class_eachfile_item,class_item,class_fileCount):
    all_tfidf={}
    for label,all_item_times_frequncy in class_eachfile_item.iteritems():
        tfidf_class_list=tfidf_single_class(all_item_times_frequncy,class_item[label],class_fileCount[label])
        all_tfidf[label]=tfidf_class_list
    return all_tfidf


def extract_features(class_item,class_fileCount):
  
    featureList=[]
    class_item_tf_idf=tfidf_class(class_item,class_fileCount)
    for index,item_tf_idf in class_item_tf_idf.iteritems():
        sortedItem=sorted(item_tf_idf.iteritems(),key=lambda x:x[1],reverse=True)
        itemlist=[item[0] for item in sortedItem][:1000]
        featureList.extend(itemlist)
       
    return  list(set(featureList))
    
def save_features(all_tfidf,featureList):
    with open("featurelist.scale",'wb') as f:
        for label,item_docs_tfidf in all_tfidf.iteritems():

            for item_tfidf in item_docs_tfidf:
                f.write(str(label)+" ")
                for index,item in enumerate(featureList):
                    tfidf= item_tfidf[item] if item_tfidf.get(item) else 0
                    f.write(str(index)+":"+str(tfidf)+" ")
                
                f.write("\n")
    
if __name__=="__main__":
    #test.....
    rootdir="TestCorpus"
    class_item,class_eachfile_item,class_fileCount=stat_class_item_bymyself(rootdir)  
    print class_item 
    print class_eachfile_item
    print class_fileCount
    item_class=stat_item_class_bymyself(class_item)
    print item_class
    all_tfidf=tfidf_all_class(class_eachfile_item,class_item,class_fileCount)
    print all_tfidf
    featurelist=extract_features(class_item,class_fileCount)
    print featurelist
    save_features(all_tfidf,featurelist)
    