# -*- coding:utf-8 -*-  
"""
K-means算法
k-均值算法，二维实现，主要用于聚类，将相近的样本归为同一类
基本步骤：计算质心-->根据距离进行分配-->重新更新

"""
import os
from numpy import *
import matplotlib.pyplot as plt
from compiler.ast import flatten

"""加载样本数据，从文件中读入"""
def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():   #按行读取文件
		curline = line.strip().split('\t')
		fltline = map(float,curline) #根据提供的函数对指定序列做映射
		dataMat.append(fltline)  #将文本文件导入到一个列表
	return dataMat

"""距离计算，这里用的是最简单的欧式距离"""
def distEclud(vecA,vecB):
	return sqrt(sum(power(vecA-vecB,2)))

"""随机生成初始的k个质心，也就是聚类后每一类的中心"""
def randCent(dataSet,k):
	n = shape(dataSet)[1] #n=2，数据是二维的点
	centroids = mat(zeros((k,n))) #k*2的零矩阵
	for j in range(n):
		minJ = min(dataSet[:,j]) #计算所有样本的范围，这是为了使随机生成的中心点位于数据集的范围之内
		rangeJ = float(max(dataSet[:,j])-minJ)
		centroids[:,j] = minJ + rangeJ * random.rand(k,1) #rand产生0到1范围的随机数
	return centroids  #返回了k个随机点

"""实现k均值算法，初始的k人为给定"""
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
	m = shape(dataSet)[0]  #数据集的数量
	clusterAssment = mat(zeros((m,2))) #m*2的零矩阵，第一列记录该数据属于哪一簇，第二列存储误差值（与簇中心的距离）
	centroids = createCent(dataSet,k) #随机生成初始质心
	clusterChanged = True  #判断是否收敛的标志，true为未收敛
	while clusterChanged: 
		clusterChanged = False
		for i in range(m): #依次遍历数据集中的所有点
			minDist = inf #inf代表正无穷大
			minIndex = -1
			for j in range(k): #将每一个数据点与质心依次比较距离，得出属于哪一簇
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if(distJI < minDist):
					minDist = distJI
					minIndex = j
			if clusterAssment[i,0] != minIndex: #如果原来不属于这一簇，则说明发生了更改，未收敛
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist ** 2 #记录对应的簇分配信息
		print centroids

		#遍历一次之后，依次更新质心所在的位置
		for cent in range(k): 
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #nonzero取非0值的下标，过滤出属于当前簇的点
			#print ptsInClust
			#展示结果
			#x=flatten(ptsInClust[:,0].tolist())
			#y=flatten(ptsInClust[:,1].tolist())
			#plt.scatter(x,y)   #将点展示出来
			centroids[cent,:] = mean(ptsInClust,axis=0) #属于本簇的点求均值，进行更新，其中，axis=0表示跨行
		#plt.scatter(flatten(centroids[:,0].tolist()),flatten(centroids[:,1].tolist()),marker='x')
		#plt.show()
	return centroids,clusterAssment

if __name__ == '__main__':
	test = loadDataSet('./testSet.txt')
	x=[]
	y=[]
	for t in test:
		x.append(t[0])
		y.append(t[1])
	plt.scatter(x,y)
	plt.show() #得到原始数据的散点图
	dataSet = mat(test)
	#print randCent(dataSet,4)
	k=4
	#调用kmenas算法
	centroids,clusterAssment = kMeans(dataSet,k)
	

	

