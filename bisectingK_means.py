# -*- coding:utf-8 -*-  
"""
二分K-means算法
k-均值算法，二维实现，主要用于聚类，将相近的样本归为同一类
解决原始k均值方法容易收敛于局部最优解的问题
基本步骤：
首先将所有点作为一个簇，然后将该值一分为二，选择其中一个簇继续划分，直到满足用户指定的簇数目为止
度量指标：SSE（距离误差平方和）
选择哪一个簇继续划分：
（1）根据其划分是否可以最大程度降低整体SSE的值
（2）选择SSE最大的簇进行划分

"""

import os
from numpy import *
import matplotlib.pyplot as plt
from kMeans import loadDataSet,distEclud,kMeans
from compiler.ast import flatten

"""实现二分K均值算法"""
def biKmeans(dataSet,k,distMeas=distEclud):
	m = shape(dataSet)[0] #点的数目
	clusterAssment = mat(zeros((m,2))) # 状态矩阵，m*2的零矩阵，第一列记录该数据属于哪一簇，第二列存储误差值（与簇中心的距离）

	"""创建一个初始簇，包括整个数据集"""
	centroid0 = mean(dataSet,axis=0).tolist()[0] #初始簇的质心
	#print centroid0
	centList = [centroid0] #保存所有质心的集合
	#print centList,mat(centroid0)
	for j in range(m): #更新状态矩阵中的距离误差为各个点到初始质心的距离平方
		clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2

	"""当簇的数目不满足用户需要时，进行簇的划分"""
	while(len(centList) < k):
		lowestSSE = inf
		"""尝试对每一个簇进行划分，比较其误差找出最优的划分方案"""
		for i in range(len(centList)): #遍历当前的每一个簇
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]  #过滤出属于这一簇的数据
			centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas) #对这一簇进行2-均值划分
			sseSplit = sum(splitClustAss[:,1]) #计算划分之后的SSE误差值
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1]) #其他簇的误差
			print "sseSplit, and sseNotSplit",sseSplit,sseNotSplit
			
			"""当这一划分使得SSE的值减小时保存，最终找到的是使整体SSE降低最多的那一划分方案"""
			if (sseSplit + sseNotSplit)<lowestSSE:
				bestCentToSplit = i  #划分簇的序号
				bestNewCents = centroidMat #最优划分得到的质心
				bestClustAss = splitClustAss.copy() #最优划分得到的状态矩阵
				lowestSSE = sseSplit + sseNotSplit #最小ssE误差
		
		"""经过一次遍历确定了要划分的簇，则进行更新"""
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #对划分之后新得到的两个簇的编号进行更新
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
		print "the bestCentToSplit is:",bestCentToSplit
		print "the len of bestClustAss is:",len(bestClustAss)

		centList[bestCentToSplit]=bestNewCents[0,:] #更新划分之后的两个质心
		centList.append(bestNewCents[1,:])
		clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]=bestClustAss #更新状态矩阵

		print "centList",centList
		
		"""用matplotlib展示划分过程及结果"""		
		for t in range(len(centList)): #遍历当前的每一个簇
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == t)[0],:]  #过滤出属于这一簇的数据
			x=flatten(ptsInCurrCluster[:,0].tolist())
			y=flatten(ptsInCurrCluster[:,1].tolist())
			plt.scatter(x,y)   #将点展示出来
		xx=[]
		yy=[]
		for cent in centList: #标注质心
			xx.append(cent[0,0])
			yy.append(cent[0,1])
		plt.scatter(xx,yy,marker='*')
		plt.show()
		
	return centList,clusterAssment


if __name__ == '__main__':
	test = loadDataSet('./testSet2.txt')
	x=[]
	y=[]
	for t in test:
		x.append(t[0])
		y.append(t[1])
	plt.scatter(x,y)
	plt.show() #得到原始数据的散点图
	dataSet = mat(test)
	#print dataSet,dataSet[0,:]
	biKmeans(dataSet,3)
