# k-邻近算法

# *使用Python实现kNN分类为算法*

## 计算距离函数classify0()如下：

~~~python
    def classify0(inX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        diffMat = tile(inX, (dataSetSize,1)) - dataSet
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndicies = distances.argsort()
        classCount={}
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
~~~

输入参数：用于分类的输入向量inX，训练样本dataSet，标签向量labels，k表示最近邻居数目。

tile()函数，构造矩阵，例：
~~~
    >>>numpy.tile([2,1],[2,3])
    array([[2, 1, 2, 1, 2, 1],
       [2, 1, 2, 1, 2, 1]])
~~~
argsort()函数，从小到大排序的索引值，例：
~~~
    >>>arr1
    array([3, 1, 2])
    >>> arr1.argsort()
    array([1, 2, 0], dtype=int32)
~~~

##示例一：约会匹配
从文本文件datingTestSet2.txt中读取约会数据

数据如图：
![dating](http://upload-images.jianshu.io/upload_images/6835596-18d7fc63f4453d89.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

###读取文件函数file2matrix()如下：

~~~python
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
~~~
line.strip()去掉回车，line.split('\t')用tab分割

###归一化数值函数autoNorm()

属性数字差值太大对计算结果影响最大，所以需要将取值范围处理为0到1利用公式
newValue = (oldValue-min)/(max-min)

~~~python
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
~~~

###测试分类结果以及错误率
~~~python
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
~~~

终端运行kNN.datingClassTest()得到如下结果:![结果如图](http://upload-images.jianshu.io/upload_images/6835596-b8eb6a262bc8a853.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

点击下载[项目代码](https://github.com/Vanvansama/Machine-Learning/tree/master/KNN)

---
## 示例二：手写识别
识别数字0到9，已将图像转换为文本格式如图：![手写数字](http://upload-images.jianshu.io/upload_images/6835596-26a0bb1dbf952dd4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### img2vector()函数：
为了使用分类器classify0，把32\*32的矩阵转换为1\*1024的向量

~~~python
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
~~~

###测试分类结果以及错误率

~~~python
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
~~~

终端运行kNN.handwritingClassTest()
结果如图：![image.png](http://upload-images.jianshu.io/upload_images/6835596-5194541f496a1f0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

trainingDigits中包含2000个例子，testDigits中包含大约900个例子，分类结果错误率为1.3%，实际使用这个算法时使用效率不高，因为为每个测试做2000次距离计算，还为测试向量准备2MB储存空间
