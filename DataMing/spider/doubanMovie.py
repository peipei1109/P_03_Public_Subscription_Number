# -*- encoding: utf-8 -*-
'''
Created on 2016年9月10日

@author: LuoPei
'''



##从豆瓣网页中得到用户id 
 
##网页地址类型：http://movie.douban.com/subject/26289144/collections?start=0 
##http://movie.douban.com/subject/26289144/collections?start=20 
 
from bs4 import BeautifulSoup
import codecs 
import time
import urllib2


baseUrl='http://movie.douban.com/subject/25895276/collections?start='


# proxyInfo='127.0.0.1:8087'
# 
# proxySupport=urllib2.ProxyHandler({'http':proxyInfo})
# 
# opener=urllib2.build_opener(proxySupport)
# 
# urllib2.install_opener(opener)



#将用户信息（id，主页链接）保存至文件

def saveUserInfo(idList,linkList):

    if len(idList)!=len(linkList):

        print 'Error: len(idList)!=len(linkList) !'

        return

    writeFile=codecs.open('UserIdList3.txt','a','utf-8')

    size=len(idList)

    for i in range(size):

        writeFile.write(idList[i]+'\t'+linkList[i]+'\n')

    writeFile.close()
 

#从给定html文本中解析用户id和连接

def parseHtmlUserId(html):

    idList=[]   #返回的id列表

    linkList=[] #返回的link列表


    soup=BeautifulSoup(html)

    ##<td width="80" valign="top">

    ##<a href="http://movie.douban.com/people/liaaaar/">

    ##<img class="" src="/u3893139-33.jpg" alt="Liar." />

    ##</a>

    ##</td>

    td_tags=soup.findAll('td',width='80',valign='top')

    i=0

    for td in td_tags:

        #前20名用户是看过这部电影的，

        #而后面的只是想看这部电影的用户，因此舍弃

        if i==20:

            break

        a=td.a

        link=a.get('href')

        i_start=link.find('people/')

        id=link[i_start+7:-1]

        idList.append(id)
     
        linkList.append(link)
     
        i+=1
    return (idList,linkList)


#返回指定编号的网页内容

def getHtml(num):

    url=baseUrl+str(num)

    page=urllib2.urlopen(url)

    html=page.read()

    return html


def launch():

    #指定起始编号：20的倍数

    ques=raw_input('Start from number?（Multiples of 20） ')

    startNum=int(ques)

    if startNum%20 != 0:

        print 'Input number error!'

        return

    for i in range(startNum,200,20):

        print 'Loading page %d/200 ...' %(i+1)

        html=getHtml(i)

        (curIdList,curLinkList)=parseHtmlUserId(html)

        saveUserInfo(curIdList,curLinkList)
     
        print 'Sleeping.'

        time.sleep(5)


#下面是KNN算法~使用矢量化操作可以省很多时间~
from numpy import *

#打开数据文件，导出为矩阵，其中最后一列为类别
def fileToMatrix(filename, sep=','):
    f = open(filename)
    content = f.readlines()
    f.close()

    first_line_list = content[0].strip().split(sep)

    data_matrix = zeros( (len(content), len(first_line_list)-1) )
    label_vector = []

    index = 0
    for line in content:
        list_from_line = line.strip().split(sep)
        data_matrix[index,:] = list_from_line[0:-1]
        label_vector.append(int(list_from_line[-1]))
        index += 1

    return (data_matrix,label_vector)

def classify(inX, data_matrix, label_vector, k):
    diff_matrix = inX - data_matrix
    square_diff_matrix = diff_matrix ** 2
    square_distances = square_diff_matrix.sum(axis=1)

    sorted_indicies = square_distances.argsort()

    label_count = {}

    for i in range(k):
        cur_label = label_vector[ sorted_indicies[i] ]
        label_count[cur_label] = label_count.get(cur_label, 0) + 1

    max_count = 0
    nearest_label = None
    for label in label_count:
        count = label_count[label]
        if count > max_count:
            max_count = count
            nearest_label = label
    return nearest_label


def test(filename,k=3,sep=',',hold_ratio=0.3):
        data_matrix, label_vector = fileToMatrix(filename,sep=sep)

        data_num = data_matrix.shape[0]
        test_num = int(hold_ratio * data_num)
        train_num = data_num - test_num

        train_matrix = data_matrix[0:train_num,:]
        test_matrix = data_matrix[train_num:,:]

        train_label_vector = label_vector[0:train_num]
        test_label_vector = label_vector[train_num:]

        right_count = 0
        for i in range(test_num):
                inX = test_matrix[i,:]

                classify_result = classify(inX, train_matrix, train_label_vector, k)
                if classify_result == test_label_vector[i]:
                        right_count += 1
                print("  The classifier came back with: %d, the real answer is: %d" % (classify_result, test_label_vector[i]))

        accuracy = float(right_count)/float(test_num)
        print('The total accuracy is %f' % accuracy)
        

if __name__=="__main__":
    launch()
