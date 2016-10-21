# -*- encoding: utf-8 -*-
'''
Created on 2016年4月22日

@author: LuoPei
'''
#urllib2.urlopen 可以传入一个Request 实例,Request可以add自己想要的header，urllib只能请求url
#urllib可以retrieve或者quote，unquote等。所以这两个模块经常混合在一起使用
#beatifulsoup是一个解析xml的工具~

#以下是一个请求实例

import sys
import urllib
import urllib2
import json
import requests

url = 'http://apis.baidu.com/txapi/mvtp/meinv?num=10'
req = urllib2.Request(url)
req.add_header("apikey", "d3878a5246c4a33c850f185a7095313e")
print req
resp = urllib2.urlopen(req)
content = resp.read()
if(content):
   print(content)


#以下是正式代码
#写api地址，参数表
url = 'http://apis.baidu.com/txapi/mvtp/meinv'
headers = {'apikey':'d3878a5246c4a33c850f185a7095313e'}
params = {'num':'10'}

#发出请求，得到响应
r = requests.get(url,params = params,headers=headers)
r = r.json()


#定义一个存储图片的函数
def saveImage(imgUrl,imgName= 'default.jpg'):
  response = requests.get(imgUrl,stream = True)
  image = response.content
  dst = "f:baidu_img"
  path = dst+imgName
  print 'save the file:'+path+'n'
  with open(path,'wb') as img:
    img.write(image)
  img.close()
  
  
#开始获取图片
def run():
  for line in r['newslist']:
   title = line['title']
   picUrl = line['picUrl']
   saveImage(picUrl,imgName=title+'.jpg')
   

if __name__=="__main__":
    run()